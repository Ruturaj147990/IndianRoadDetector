"""
Production Training Pipeline for IRD — IndianRoadDetection (V1).

Official Model Name: IRD (IndianRoadDetection)
Internal Model Class: IndianRoadDetector (pure PyTorch, ~4.24M parameters)

Features:
- Pure PyTorch training and validation pipeline (100% independent from Ultralytics).
- Custom multi-scale loss (IndianRoadLoss) with CIoU localization, focal objectness, and focal classification.
- Dynamic multi-scale spatial target assignment (MultiScaleSpatialMatcher).
- Automated mixed-precision training (AMP via torch.amp) when CUDA is available.
- Resumable checkpoints (restores model, optimizer, scheduler, scaler, epoch, and history).
- Transparent validation metrics: validation loss breakdown, mean match IoU, and top-1 class accuracy.
- Export of IRD-specific artifacts: ird_best.pt, ird_last.pt, ird_history.csv, ird_history.json, ird_config.json.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector, build_detector
from src.models.losses.custom_loss import (
    IndianRoadLoss,
    LossResult,
    bbox_ciou,
    build_loss,
    decode_boxes_at_indices,
)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class YOLODetectionDataset(Dataset):
    """
    Standard PyTorch Dataset for YOLO-format detection datasets.
    
    Reads images via PIL and converts to normalized float32 tensors [3, H, W].
    Reads corresponding YOLO text files: <class_id> <cx> <cy> <w> <h> in [0, 1].
    Includes validation safeguards to filter invalid or degenerate bounding boxes.
    """
    def __init__(
        self,
        img_dir: Path,
        lbl_dir: Path,
        img_size: Tuple[int, int] = (640, 640),
        max_samples: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.augment = augment

        # Find all valid image files
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.img_paths: List[Path] = []
        for ext in extensions:
            self.img_paths.extend(sorted(list(img_dir.glob(ext))))

        if max_samples is not None and max_samples > 0:
            self.img_paths = self.img_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        # 1. Parse YOLO label file: <class_id> <cx> <cy> <w> <h>
        boxes: List[List[float]] = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = float(parts[0])
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])

                            # Coordinate safeguards: clamp to [0.001, 0.999]
                            cx = max(0.001, min(0.999, cx))
                            cy = max(0.001, min(0.999, cy))
                            w = max(0.001, min(1.0, w))
                            h = max(0.001, min(1.0, h))

                            if w > 0.001 and h > 0.001:
                                boxes.append([cls_id, cx, cy, w, h])
                        except ValueError:
                            continue

        # 2. Load and augment image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            
            # Apply training augmentations
            if self.augment:
                # A. Horizontal Flip (p=0.5)
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    for b in boxes:
                        b[1] = 1.0 - b[1]  # Flip cx
                        
                # B. Photometric Jitter (p=0.6)
                if random.random() < 0.6:
                    b_factor = random.uniform(0.85, 1.15)
                    c_factor = random.uniform(0.85, 1.15)
                    img = ImageEnhance.Brightness(img).enhance(b_factor)
                    img = ImageEnhance.Contrast(img).enhance(c_factor)
                    
                # C. Subtle Gaussian Blur / Haze (p=0.15)
                if random.random() < 0.15:
                    blur_radius = random.uniform(0.3, 0.8)
                    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Resize to target input size
            img = img.resize(self.img_size)
            img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

            # D. Mild Cutout / Occlusion Erasing (p=0.25)
            if self.augment and random.random() < 0.25:
                H, W = self.img_size
                num_patches = random.randint(1, 2)
                for _ in range(num_patches):
                    ph = random.randint(16, 48)
                    pw = random.randint(16, 48)
                    py = random.randint(0, H - ph)
                    px = random.randint(0, W - pw)
                    img_tensor[:, py:py+ph, px:px+pw] = 0.45

        if boxes:
            target_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            target_tensor = torch.empty((0, 5), dtype=torch.float32)

        return img_tensor, target_tensor


def ird_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function converting batch targets into unified [N, 6] format:
    [batch_idx, class_id, cx, cy, w, h] as expected by IndianRoadLoss.
    """
    images = torch.stack([item[0] for item in batch], dim=0)

    target_list = []
    for b_idx, item in enumerate(batch):
        t = item[1]
        if t.numel() > 0:
            b_col = torch.full((t.shape[0], 1), b_idx, dtype=t.dtype)
            target_list.append(torch.cat([b_col, t], dim=1))

    if target_list:
        targets = torch.cat(target_list, dim=0)
    else:
        targets = torch.empty((0, 6), dtype=torch.float32)

    return images, targets


def resolve_dataset_dirs(data_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Resolves train and validation image/label directories for standard YOLO dataset layouts.
    
    Supports:
      1. <data_dir>/images/train, <data_dir>/labels/train
         <data_dir>/images/val,   <data_dir>/labels/val
      2. <data_dir>/train/images, <data_dir>/train/labels
         <data_dir>/val/images,   <data_dir>/val/labels
      3. Fallback to data/sample_16 if path is missing.
    """
    # Check layout 1: images/train
    if (data_dir / "images" / "train").exists() and (data_dir / "labels" / "train").exists():
        train_img = data_dir / "images" / "train"
        train_lbl = data_dir / "labels" / "train"
        val_img = data_dir / "images" / "val" if (data_dir / "images" / "val").exists() else train_img
        val_lbl = data_dir / "labels" / "val" if (data_dir / "labels" / "val").exists() else train_lbl
        return train_img, train_lbl, val_img, val_lbl

    # Check layout 2: train/images
    if (data_dir / "train" / "images").exists() and (data_dir / "train" / "labels").exists():
        train_img = data_dir / "train" / "images"
        train_lbl = data_dir / "train" / "labels"
        val_img = data_dir / "val" / "images" if (data_dir / "val" / "images").exists() else train_img
        val_lbl = data_dir / "val" / "labels" if (data_dir / "val" / "labels").exists() else train_lbl
        return train_img, train_lbl, val_img, val_lbl

    # Fallback to local sample dataset
    local_sample = Path(_project_root) / "data" / "sample_16"
    print(f"[IRD Dataset] Specified path '{data_dir}' not found. Falling back to '{local_sample}'...")
    train_img = local_sample / "images" / "train"
    train_lbl = local_sample / "labels" / "train"
    val_img = local_sample / "images" / "val" if (local_sample / "images" / "val").exists() else train_img
    val_lbl = local_sample / "labels" / "val" if (local_sample / "labels" / "val").exists() else train_lbl

    return train_img, train_lbl, val_img, val_lbl


def compute_lightweight_metrics(
    predictions: Any,
    targets: torch.Tensor,
    loss_fn: IndianRoadLoss,
    img_size: Tuple[int, int] = (640, 640),
) -> Dict[str, float]:
    """
    Computes transparent, honest validation metrics without claiming fake mAP:
      1. Mean Match IoU: Average IoU between predicted boxes and ground-truth boxes on positive locations.
      2. Class Top-1 Accuracy: Accuracy of predicted class logits on positive locations.
      3. Objectness Margin: Separation between foreground objectness scores and background scores.
    """
    with torch.no_grad():
        grid_shapes = [(p.shape[2], p.shape[3]) for p in predictions.box_preds]
        matches = loss_fn.matcher.match(targets, grid_shapes, img_size=img_size)

        all_ious = []
        correct_cls = 0
        total_cls = 0

        for s_idx, stride in enumerate(loss_fn.strides):
            match = matches[s_idx]
            n_pos = len(match["batch_idx"])
            if n_pos > 0:
                b_idx = match["batch_idx"]
                g_y = match["grid_y"]
                g_x = match["grid_x"]
                gt_boxes = match["gt_boxes"]
                gt_classes = match["gt_classes"]

                # 1. Box IoU
                raw_boxes = predictions.box_preds[s_idx][b_idx, :, g_y, g_x]
                dec_boxes = decode_boxes_at_indices(raw_boxes, g_x, g_y, stride)

                gt_cx, gt_cy, gt_w, gt_h = gt_boxes.unbind(-1)
                gt_dec = torch.stack([
                    gt_cx - gt_w / 2.0,
                    gt_cy - gt_h / 2.0,
                    gt_cx + gt_w / 2.0,
                    gt_cy + gt_h / 2.0,
                ], dim=-1)

                # Intersection over Union
                inter_x1 = torch.max(dec_boxes[:, 0], gt_dec[:, 0])
                inter_y1 = torch.max(dec_boxes[:, 1], gt_dec[:, 1])
                inter_x2 = torch.min(dec_boxes[:, 2], gt_dec[:, 2])
                inter_y2 = torch.min(dec_boxes[:, 3], gt_dec[:, 3])
                inter_w = (inter_x2 - inter_x1).clamp(min=0)
                inter_h = (inter_y2 - inter_y1).clamp(min=0)
                inter_area = inter_w * inter_h
                w1 = (dec_boxes[:, 2] - dec_boxes[:, 0]).clamp(min=0)
                h1 = (dec_boxes[:, 3] - dec_boxes[:, 1]).clamp(min=0)
                w2 = (gt_dec[:, 2] - gt_dec[:, 0]).clamp(min=0)
                h2 = (gt_dec[:, 3] - gt_dec[:, 1]).clamp(min=0)
                union_area = (w1 * h1) + (w2 * h2) - inter_area + 1e-7
                iou = inter_area / union_area
                all_ious.extend(iou.cpu().tolist())

                # 2. Class Accuracy
                cls_logits = predictions.cls_preds[s_idx][b_idx, :, g_y, g_x]
                pred_classes = cls_logits.argmax(dim=-1)
                correct_cls += (pred_classes == gt_classes).sum().item()
                total_cls += n_pos

        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        cls_acc = (correct_cls / total_cls * 100.0) if total_cls > 0 else 0.0

        return {
            "mean_iou": round(mean_iou, 4),
            "cls_accuracy_pct": round(cls_acc, 2),
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: IndianRoadLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool = False,
    clip_norm: float = 10.0,
    img_size: Tuple[int, int] = (640, 640),
) -> Dict[str, float]:
    """Train the IRD model for one full epoch."""
    model.train()
    total_loss_sum = 0.0
    box_loss_sum = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    total_positives = 0
    batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixed-precision context when enabled
        if use_amp:
            with torch.amp.autocast("cuda"):
                predictions = model(images)
                loss_result: LossResult = loss_fn(predictions, targets, img_size=img_size)
            scaler.scale(loss_result.total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss_result: LossResult = loss_fn(predictions, targets, img_size=img_size)
            loss_result.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        total_loss_sum += loss_result.total_loss.item()
        box_loss_sum += loss_result.box_loss.item()
        obj_loss_sum += loss_result.objectness_loss.item()
        cls_loss_sum += loss_result.classification_loss.item()
        total_positives += loss_result.number_of_positive_samples
        batches += 1

    n_b = max(batches, 1)
    return {
        "total_loss": total_loss_sum / n_b,
        "box_loss": box_loss_sum / n_b,
        "obj_loss": obj_loss_sum / n_b,
        "cls_loss": cls_loss_sum / n_b,
        "positives": total_positives,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: IndianRoadLoss,
    device: torch.device,
    img_size: Tuple[int, int] = (640, 640),
) -> Dict[str, float]:
    """Validate the IRD model without gradient computation."""
    model.eval()
    total_loss_sum = 0.0
    box_loss_sum = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    total_positives = 0
    all_ious: List[float] = []
    correct_classes = 0
    total_classes = 0
    batches = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss_result: LossResult = loss_fn(predictions, targets, img_size=img_size)

            total_loss_sum += loss_result.total_loss.item()
            box_loss_sum += loss_result.box_loss.item()
            obj_loss_sum += loss_result.objectness_loss.item()
            cls_loss_sum += loss_result.classification_loss.item()
            total_positives += loss_result.number_of_positive_samples
            batches += 1

            # Lightweight metrics
            m = compute_lightweight_metrics(predictions, targets, loss_fn, img_size=img_size)
            if m["mean_iou"] > 0.0:
                all_ious.append(m["mean_iou"])
            if m["cls_accuracy_pct"] > 0.0:
                correct_classes += m["cls_accuracy_pct"]
                total_classes += 1

    n_b = max(batches, 1)
    val_iou = float(np.mean(all_ious)) if all_ious else 0.0
    val_acc = (correct_classes / total_classes) if total_classes > 0 else 0.0

    return {
        "val_total_loss": total_loss_sum / n_b,
        "val_box_loss": box_loss_sum / n_b,
        "val_obj_loss": obj_loss_sum / n_b,
        "val_cls_loss": cls_loss_sum / n_b,
        "val_positives": total_positives,
        "val_mean_iou": round(val_iou, 4),
        "val_cls_acc": round(val_acc, 2),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
    use_amp: bool,
) -> None:
    """Save an IRD checkpoint containing full model and optimizer states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_name": "IRD (IndianRoadDetection)",
        "version": "1.0",
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp and scaler is not None else None,
        "config": config,
        "history": history,
        "num_classes": config.get("num_classes", 12),
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """Resume training from an IRD checkpoint."""
    print(f"[IRD Checkpoint] Resuming training from: {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if use_amp and scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    history = checkpoint.get("history", [])

    print(f"[IRD Checkpoint] Successfully restored state! Next epoch: {start_epoch}, Best Val Loss: {best_val_loss:.4f}")
    return start_epoch, best_val_loss, history


def export_history(
    history: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Exports training history to CSV and JSON formats."""
    csv_path = output_dir / "ird_history.csv"
    json_path = output_dir / "ird_history.json"

    # Export JSON
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # Export CSV
    if history:
        fieldnames = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)


def train_ird(
    data_dir: str = "/content/indian_road_yolo",
    output_dir: str = "experiments/custom_model/ird_v1",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    img_size: int = 640,
    num_classes: int = 12,
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    cls_weight: float = 1.0,
    decoder_version: str = "v2_smooth",
    quality_aware_obj: bool = True,
    matcher_version: str = "v1_spatial",
    class_balanced_loss: bool = False,
    small_obj_floor: bool = False,
    device_str: str = "auto",
    workers: int = 2,
    seed: int = 42,
    use_amp: bool = True,
    resume_path: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main training function for IRD (IndianRoadDetection).
    
    Args:
        data_dir: Path to YOLO-format dataset directory.
        output_dir: Output directory for checkpoints, history CSV/JSON, config.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Initial learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        img_size: Image input resolution (H=W).
        num_classes: Number of detection classes (default: 12).
        box_weight: Weight for CIoU box loss.
        obj_weight: Weight for focal objectness loss.
        cls_weight: Weight for focal classification loss.
        device_str: 'auto', 'cuda', or 'cpu'.
        workers: DataLoader worker count.
        seed: Random seed.
        use_amp: Whether to use mixed precision when CUDA is available.
        resume_path: Path to checkpoint to resume from.
        max_train_samples: Optional cap on training samples (useful for smoke tests).
        max_val_samples: Optional cap on validation samples (useful for smoke tests).
        
    Returns:
        Dictionary summarizing the training session.
    """
    set_seed(seed)

    # 1. Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    amp_active = (device.type == "cuda") and use_amp

    out_path = Path(_project_root) / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print(f"{'IRD — IndianRoadDetection (V1) Training Pipeline':^82}")
    print("=" * 82)
    print(f"Official Model Name:      IRD (IndianRoadDetection)")
    print(f"Internal Class Name:      IndianRoadDetector")
    print(f"Compute Device:           {device} (AMP active: {amp_active})")
    print(f"Dataset Path:             {data_dir}")
    print(f"Target Image Size:        {img_size} x {img_size}")
    print(f"Number of Classes:        {num_classes}")
    print(f"Total Epochs:             {epochs}")
    print(f"Batch Size:               {batch_size}")
    print(f"Learning Rate:            {lr}")
    print(f"Output Directory:         {out_path.resolve()}")
    print("-" * 82)

    # 2. Resolve Dataset Directories
    train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir = resolve_dataset_dirs(Path(data_dir))
    print(f"Train Images Directory:   {train_img_dir}")
    print(f"Val Images Directory:     {val_img_dir}")

    # 3. Create Datasets and DataLoaders
    train_dataset = YOLODetectionDataset(
        img_dir=train_img_dir,
        lbl_dir=train_lbl_dir,
        img_size=(img_size, img_size),
        max_samples=max_train_samples,
        augment=True,
    )
    val_dataset = YOLODetectionDataset(
        img_dir=val_img_dir,
        lbl_dir=val_lbl_dir,
        img_size=(img_size, img_size),
        max_samples=max_val_samples,
        augment=False,
    )

    print(f"Training Samples:         {len(train_dataset):,}")
    print(f"Validation Samples:       {len(val_dataset):,}")
    print("-" * 82)

    # Use specified workers
    num_workers = workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ird_collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ird_collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Initialize IRD Model & Loss
    model = IndianRoadDetector(num_classes=num_classes).to(device)
    loss_fn = IndianRoadLoss(
        num_classes=num_classes,
        box_weight=box_weight,
        obj_weight=obj_weight,
        cls_weight=cls_weight,
        decoder_version=decoder_version,
        quality_aware_obj=quality_aware_obj,
        matcher_version=matcher_version,
        class_balanced_loss=class_balanced_loss,
        small_obj_floor=small_obj_floor,
    ).to(device)

    param_counts = model.get_parameter_counts(only_trainable=True)
    print(f"IRD Model Parameter Count:")
    print(f"  Backbone:               {param_counts['backbone']:>10,} ({param_counts['backbone']/1e6:.2f}M)")
    print(f"  Neck:                   {param_counts['neck']:>10,} ({param_counts['neck']/1e6:.2f}M)")
    print(f"  Decoupled Head:         {param_counts['head']:>10,} ({param_counts['head']/1e6:.2f}M)")
    print(f"  Complete IRD Detector:  {param_counts['total']:>10,} ({param_counts['total']/1e6:.2f}M)")
    print(f"Matcher Version:          {matcher_version}")
    print(f"Class-Balanced Loss:      {class_balanced_loss}")
    print(f"Small-Object Floor (GSO): {small_obj_floor}")
    print("-" * 82)

    # 5. Optimizer, Scheduler, and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    if amp_active:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # 6. Save Configuration JSON
    config_record = {
        "model_name": "IRD (IndianRoadDetection)",
        "version": "1.0",
        "parameters": param_counts["total"],
        "num_classes": num_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "img_size": img_size,
        "box_weight": box_weight,
        "obj_weight": obj_weight,
        "cls_weight": cls_weight,
        "decoder_version": decoder_version,
        "quality_aware_obj": quality_aware_obj,
        "matcher_version": matcher_version,
        "class_balanced_loss": class_balanced_loss,
        "small_obj_floor": small_obj_floor,
        "device": str(device),
        "use_amp": amp_active,
        "seed": seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(out_path / "ird_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # 7. Checkpoint Resume Handling
    start_epoch = 1
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    if resume_path and Path(resume_path).exists():
        start_epoch, best_val_loss, history = load_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=amp_active,
        )

    # 8. Training Loop
    print(f"Starting IRD training from Epoch {start_epoch} to {epochs}...")
    pipeline_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Training phase
        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=amp_active,
            img_size=(img_size, img_size),
        )

        scheduler.step()

        # Validation phase
        val_res = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            img_size=(img_size, img_size),
        )

        epoch_time = time.time() - epoch_start

        # Record history
        record = {
            "epoch": epoch,
            "train_total_loss": round(train_res["total_loss"], 4),
            "train_box_loss": round(train_res["box_loss"], 4),
            "train_obj_loss": round(train_res["obj_loss"], 4),
            "train_cls_loss": round(train_res["cls_loss"], 4),
            "train_positives": train_res["positives"],
            "val_total_loss": round(val_res["val_total_loss"], 4),
            "val_box_loss": round(val_res["val_box_loss"], 4),
            "val_obj_loss": round(val_res["val_obj_loss"], 4),
            "val_cls_loss": round(val_res["val_cls_loss"], 4),
            "val_positives": val_res["val_positives"],
            "val_mean_iou": val_res["val_mean_iou"],
            "val_cls_acc": val_res["val_cls_acc"],
            "lr": round(current_lr, 6),
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(record)

        # Pretty console log
        is_best_str = ""
        if val_res["val_total_loss"] < best_val_loss:
            best_val_loss = val_res["val_total_loss"]
            is_best_str = " (BEST)"
            # Save ird_best.pt
            save_checkpoint(
                out_path / "ird_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config_record,
                history=history,
                use_amp=amp_active,
            )

        # Save ird_last.pt after every epoch
        save_checkpoint(
            out_path / "ird_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config_record,
            history=history,
            use_amp=amp_active,
        )

        # Export CSV/JSON history
        export_history(history, out_path)

        print(
            f"IRD | Epoch [{epoch:>3}/{epochs}] | "
            f"Train Loss: {train_res['total_loss']:>7.3f} (Box: {train_res['box_loss']:>5.3f}, Obj: {train_res['obj_loss']:>7.3f}, Cls: {train_res['cls_loss']:>5.3f}) | "
            f"Val Loss: {val_res['val_total_loss']:>7.3f} (IoU: {val_res['val_mean_iou']:>5.3f}, ClsAcc: {val_res['val_cls_acc']:>5.1f}%){is_best_str} | "
            f"LR: {current_lr:.6f} | {epoch_time:.1f}s"
        )

    total_training_time = time.time() - pipeline_start
    print("-" * 82)
    print(f"IRD Training Complete! Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} min).")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")
    print(f"Saved Checkpoints:        {out_path / 'ird_best.pt'}, {out_path / 'ird_last.pt'}")
    print(f"Saved History:            {out_path / 'ird_history.csv'}, {out_path / 'ird_history.json'}")
    print("=" * 82)

    return {
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs,
        "history": history,
        "artifacts_dir": str(out_path.resolve()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRD — IndianRoadDetection Production Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="/content/indian_road_yolo",
                        help="Path to YOLO dataset root (default: /content/indian_road_yolo)")
    parser.add_argument("--output-dir", type=str, default="experiments/custom_model/ird_v1",
                        help="Output directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image resolution (default: 640)")
    parser.add_argument("--num-classes", type=int, default=12,
                        help="Number of detection classes (default: 12)")
    parser.add_argument("--box-weight", type=float, default=5.0,
                        help="Weight multiplier for CIoU box loss (default: 5.0)")
    parser.add_argument("--obj-weight", type=float, default=1.0,
                        help="Weight multiplier for focal objectness loss (default: 1.0)")
    parser.add_argument("--cls-weight", type=float, default=1.0,
                        help="Weight multiplier for focal classification loss (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reproducibility random seed (default: 42)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision training when CUDA is available")
    parser.add_argument("--decoder-version", type=str, default="v2_smooth", choices=["v2_smooth", "v1_legacy"],
                        help="Box decoder parameterization version (default: v2_smooth)")
    parser.add_argument("--quality-obj", action="store_true", default=True,
                        help="Use quality-aware IoU objectness targets (default: True)")
    parser.add_argument("--no-quality-obj", dest="quality_obj", action="store_false",
                        help="Disable quality-aware objectness (use legacy binary targets)")
    parser.add_argument("--matcher-version", type=str, default="v1_spatial", choices=["v1_spatial", "topk_adaptive_v2"],
                        help="Target assignment matcher version (default: v1_spatial)")
    parser.add_argument("--class-balanced-loss", action="store_true", default=False,
                        help="Enable class-frequency balanced positive focal classification loss")
    parser.add_argument("--small-obj-floor", action="store_true", default=False,
                        help="Enable Guaranteed Small-Object Presence Supervision (GSO floor 0.80 for scale < 96px)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to IRD checkpoint (.pt) to resume training from")
    parser.add_argument("--smoke-test", action="store_true", default=False,
                        help="Run short 2-epoch smoke test with small subset")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_epochs = args.epochs if args.epochs != 100 else 2
        print(f"[Smoke Test Mode] Setting epochs={smoke_epochs}, batch_size=2, max_samples=4...")
        train_ird(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=smoke_epochs,
            batch_size=2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            img_size=args.img_size,
            num_classes=args.num_classes,
            decoder_version=args.decoder_version,
            quality_aware_obj=args.quality_obj,
            matcher_version=args.matcher_version,
            class_balanced_loss=args.class_balanced_loss,
            small_obj_floor=args.small_obj_floor,
            device_str=args.device,
            workers=0,
            seed=args.seed,
            use_amp=args.amp,
            resume_path=args.resume,
            max_train_samples=4,
            max_val_samples=2,
        )
    else:
        train_ird(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            img_size=args.img_size,
            num_classes=args.num_classes,
            box_weight=args.box_weight,
            obj_weight=args.obj_weight,
            cls_weight=args.cls_weight,
            decoder_version=args.decoder_version,
            quality_aware_obj=args.quality_obj,
            matcher_version=args.matcher_version,
            class_balanced_loss=args.class_balanced_loss,
            small_obj_floor=args.small_obj_floor,
            device_str=args.device,
            workers=args.workers,
            seed=args.seed,
            use_amp=args.amp,
            resume_path=args.resume,
        )
