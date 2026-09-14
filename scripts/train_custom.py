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

import os
os.environ["MIOPEN_FIND_MODE"] = "2"

import argparse
import csv
import json
import math
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
from tqdm import tqdm

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
) -> Dict[str, Any]:
    """
    Computes transparent validation matches on CPU:
      1. IoU values for matched predictions on positive locations.
      2. Correct class flags for matched predictions.
      3. Total ground truth object count.
    """
    with torch.no_grad():
        grid_shapes = [(p.shape[2], p.shape[3]) for p in predictions.box_preds]
        matches = loss_fn.matcher.match(targets, grid_shapes, img_size=img_size)

        all_ious = []
        correct_cls_flags = []
        total_gt = int(targets.shape[0])

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
                correct_cls_flags.extend((pred_classes == gt_classes).cpu().tolist())

        return {
            "ious": all_ious,
            "correct_cls": correct_cls_flags,
            "total_gt": total_gt,
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
    epoch: int = 1,
    epochs: int = 50,
) -> Dict[str, float]:
    """Train the IRD model for one full epoch with live tqdm progress."""
    model.train()
    total_loss_sum = 0.0
    box_loss_sum = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    qual_loss_sum = 0.0
    aux_loss_sum = 0.0
    total_positives = 0
    batches = 0

    current_lr = float(optimizer.param_groups[0]["lr"])
    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:>2}/{epochs}] [Train]",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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

        t_loss = loss_result.total_loss.item()
        b_loss = loss_result.box_loss.item()
        o_loss = loss_result.objectness_loss.item()
        c_loss = loss_result.classification_loss.item()
        q_loss = loss_result.quality_loss.item() if loss_result.quality_loss is not None else 0.0
        a_loss = loss_result.aux_loss.item() if loss_result.aux_loss is not None else 0.0

        total_loss_sum += t_loss
        box_loss_sum += b_loss
        obj_loss_sum += o_loss
        cls_loss_sum += c_loss
        qual_loss_sum += q_loss
        aux_loss_sum += a_loss
        total_positives += loss_result.number_of_positive_samples
        batches += 1

        vram_str = f"{torch.cuda.memory_allocated(device) / (1024**3):.2f}G" if device.type == "cuda" else "cpu"
        pbar.set_postfix({
            "loss": f"{t_loss:.3f}",
            "box": f"{b_loss:.3f}",
            "obj": f"{o_loss:.3f}",
            "cls": f"{c_loss:.3f}",
            "lr": f"{current_lr:.5f}",
            "vram": vram_str,
        })

    n_b = max(batches, 1)
    return {
        "total_loss": total_loss_sum / n_b,
        "box_loss": box_loss_sum / n_b,
        "obj_loss": obj_loss_sum / n_b,
        "cls_loss": cls_loss_sum / n_b,
        "qual_loss": qual_loss_sum / n_b,
        "aux_loss": aux_loss_sum / n_b,
        "positives": total_positives,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: IndianRoadLoss,
    device: torch.device,
    img_size: Tuple[int, int] = (640, 640),
    epoch: int = 1,
    epochs: int = 50,
) -> Dict[str, float]:
    """Validate the IRD model without gradient computation."""
    model.eval()
    total_loss_sum = 0.0
    box_loss_sum = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    total_positives = 0
    cum_ious: List[float] = []
    cum_correct_cls: List[bool] = []
    total_gt_all = 0
    batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:>2}/{epochs}] [Val]  ",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )

    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(images)
            loss_result: LossResult = loss_fn(predictions, targets, img_size=img_size)

            t_loss = loss_result.total_loss.item()
            total_loss_sum += t_loss
            box_loss_sum += loss_result.box_loss.item()
            obj_loss_sum += loss_result.objectness_loss.item()
            cls_loss_sum += loss_result.classification_loss.item()
            total_positives += loss_result.number_of_positive_samples
            batches += 1

            pbar.set_postfix({"val_loss": f"{t_loss:.3f}"})

            # Transparent CPU metric accumulation
            m = compute_lightweight_metrics(predictions, targets, loss_fn, img_size=img_size)
            cum_ious.extend(m["ious"])
            cum_correct_cls.extend(m["correct_cls"])
            total_gt_all += m["total_gt"]

    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_b = max(batches, 1)

    if cum_ious:
        ious_arr = np.array(cum_ious, dtype=np.float32)
        cls_arr = np.array(cum_correct_cls, dtype=bool)

        val_iou = float(np.mean(ious_arr))
        val_acc = float(np.mean(cls_arr) * 100.0)

        # Matched detections with IoU >= 0.50 and correct class
        hit_50 = (ious_arr >= 0.50) & cls_arr
        val_recall = float(hit_50.sum() / max(total_gt_all, 1))
        val_precision = float(hit_50.sum() / max(len(ious_arr), 1))
        val_map50 = val_precision * val_recall

        # Multi-threshold IoU (0.50 to 0.95 in steps of 0.05)
        ap_list = []
        for t in np.linspace(0.50, 0.95, 10):
            hit_t = (ious_arr >= t) & cls_arr
            r_t = hit_t.sum() / max(total_gt_all, 1)
            p_t = hit_t.sum() / max(len(ious_arr), 1)
            ap_list.append(p_t * r_t)
        val_map50_95 = float(np.mean(ap_list))
    else:
        val_iou = 0.0
        val_acc = 0.0
        val_recall = 0.0
        val_precision = 0.0
        val_map50 = 0.0
        val_map50_95 = 0.0

    return {
        "val_total_loss": total_loss_sum / n_b,
        "val_box_loss": box_loss_sum / n_b,
        "val_obj_loss": obj_loss_sum / n_b,
        "val_cls_loss": cls_loss_sum / n_b,
        "val_positives": total_positives,
        "val_mean_iou": round(val_iou, 4),
        "val_cls_acc": round(val_acc, 2),
        "val_recall": round(val_recall, 4),
        "val_precision": round(val_precision, 4),
        "val_map50": round(val_map50, 4),
        "val_map50_95": round(val_map50_95, 4),
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
    data_dir: str = "data/indian_road_yolo",
    output_dir: str = "experiments/custom_model/final_training_50ep",
    epochs: int = 50,
    batch_size: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    img_size: int = 640,
    num_classes: int = 12,
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    cls_weight: float = 1.0,
    decoder_version: str = "v2_smooth",
    quality_aware_obj: bool = True,
    matcher_version: str = "topk_adaptive_v2",
    class_balanced_loss: bool = True,
    small_obj_floor: bool = False,
    use_atd: bool = True,
    use_ssdp: bool = True,
    use_fgbr: bool = True,
    use_quality: bool = True,
    use_cdg: bool = True,
    use_aux_one2one: bool = True,
    version: str = "v1.5",
    loss_type: str = "indian_road",
    device_str: str = "auto",
    workers: int = 0,
    seed: int = 42,
    use_amp: bool = True,
    resume_path: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main training function for IRD V1.5 (IndianRoadDetection).
    
    Args:
        data_dir: Path to YOLO-format dataset directory.
        output_dir: Output directory for checkpoints, history CSV/JSON, config.
        epochs: Number of training epochs (default: 50).
        batch_size: Batch size (default: 12, verified safe on RX 7700 XT 12GB).
        lr: Initial learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        img_size: Image input resolution (640x640).
        num_classes: Number of detection classes (default: 12).
        box_weight: Weight for CIoU box loss.
        obj_weight: Weight for focal objectness loss.
        cls_weight: Weight for focal classification loss.
        device_str: 'auto', 'cuda', or 'cpu'.
        workers: DataLoader worker count (default: 0 for stable Windows execution).
        seed: Random seed.
        use_amp: Whether to use mixed precision when CUDA/ROCm is available.
        resume_path: Path to checkpoint to resume from.
        max_train_samples: Optional cap on training samples.
        max_val_samples: Optional cap on validation samples.
        
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

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    total_gpu_mem = (
        torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if device.type == "cuda"
        else 0.0
    )

    print("=" * 82)
    print(f"{'IRD V1.5 — IndianRoadDetection Production Training Pipeline':^82}")
    print("=" * 82)
    print(f"Official Model Name:      IRD V1.5 (IndianRoadDetection)")
    print(f"Internal Class Name:      IndianRoadDetector")
    print(f"Compute Device:           {device} [{gpu_name} ({total_gpu_mem:.2f} GB)]")
    print(f"AMP Mixed Precision:      {amp_active}")
    print(f"Dataset Path:             {data_dir}")
    print(f"Target Image Size:        {img_size} x {img_size}")
    print(f"Number of Classes:        {num_classes}")
    print(f"Total Epochs:             {epochs}")
    print(f"Batch Size:               {batch_size}")
    print(f"DataLoader Workers:       {workers}")
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

    num_workers = workers if sys.platform != "win32" else 0
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

    # 4. Initialize IRD V1.5 Model & Loss
    model = IndianRoadDetector(
        num_classes=num_classes,
        use_atd=use_atd,
        use_ssdp=use_ssdp,
        use_fgbr=use_fgbr,
        use_quality=use_quality,
        use_cdg=use_cdg,
    ).to(device)

    if version == "v2" or loss_type == "task_aligned":
        from src.models.losses.task_aligned_loss import TaskAlignedLoss
        loss_fn = TaskAlignedLoss(
            num_classes=num_classes,
            box_weight=box_weight,
            cls_weight=cls_weight,
            qual_weight=0.5,
            obj_weight=obj_weight,
            strides=tuple(model.strides),
            topk=10,
            tal_alpha=0.5,
            tal_beta=6.0,
            class_balanced=class_balanced_loss,
        ).to(device)
    else:
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
            use_aux_one2one=use_aux_one2one,
        ).to(device)

    param_counts = model.get_parameter_counts(only_trainable=True)
    print(f"IRD Model Parameter Count:")
    print(f"  Backbone:               {param_counts['backbone']:>10,} ({param_counts['backbone']/1e6:.2f}M)")
    print(f"  Neck:                   {param_counts['neck']:>10,} ({param_counts['neck']/1e6:.2f}M)")
    print(f"  Decoupled Head:         {param_counts['head']:>10,} ({param_counts['head']/1e6:.2f}M)")
    print(f"  Complete IRD V1.5:      {param_counts['total']:>10,} ({param_counts['total']/1e6:.2f}M)")
    print(f"Architecture Modules:")
    print(f"  Anisotropic Disentangler (ATD): {use_atd}")
    print(f"  Selective Spatial Detail (SSDP): {use_ssdp}")
    print(f"  Fine-Grained Box Refiner (FGBR): {use_fgbr}")
    print(f"  Localization Quality Branch:     {use_quality}")
    print(f"  Cross-Domain Gating (CDG):       {use_cdg}")
    print(f"  Auxiliary One-to-One Matcher:    {use_aux_one2one}")
    print(f"Loss Configuration:")
    print(f"  Matcher Version:                 {matcher_version}")
    print(f"  Class-Balanced Loss:             {class_balanced_loss}")
    print(f"  Small-Object Floor (GSO):        {small_obj_floor}")
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
        "model_name": "IRD V1.5 (IndianRoadDetection)",
        "version": "1.5",
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
        "use_atd": use_atd,
        "use_ssdp": use_ssdp,
        "use_fgbr": use_fgbr,
        "use_quality": use_quality,
        "use_cdg": use_cdg,
        "use_aux_one2one": use_aux_one2one,
        "device": str(device),
        "gpu_name": gpu_name,
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
    best_map50 = 0.0
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
    print(f"\n>>> STARTING IRD V1.5 TRAINING: Epoch {start_epoch} of {epochs} <<<\n", flush=True)
    pipeline_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Reset peak memory tracking per epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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
            epoch=epoch,
            epochs=epochs,
        )

        scheduler.step()

        # Validation phase
        val_res = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            img_size=(img_size, img_size),
            epoch=epoch,
            epochs=epochs,
        )

        epoch_time = time.time() - epoch_start

        # GPU memory & speed stats
        if device.type == "cuda":
            vram_alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
            vram_res_gb = torch.cuda.memory_reserved(device) / (1024**3)
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            headroom_gb = total_vram_gb - max(peak_vram_gb, vram_res_gb)
        else:
            vram_alloc_gb = vram_res_gb = peak_vram_gb = headroom_gb = 0.0

        img_per_sec = len(train_dataset) / max(epoch_time, 0.001)

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
            "val_recall": val_res["val_recall"],
            "val_precision": val_res["val_precision"],
            "val_map50": val_res["val_map50"],
            "val_map50_95": val_res["val_map50_95"],
            "lr": round(current_lr, 6),
            "images_per_sec": round(img_per_sec, 2),
            "epoch_time_sec": round(epoch_time, 2),
            "vram_allocated_gb": round(vram_alloc_gb, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
        }
        history.append(record)

        # Best checkpoint selection by validation loss
        is_best_str = ""
        if val_res["val_total_loss"] < best_val_loss:
            best_val_loss = val_res["val_total_loss"]
            best_map50 = val_res["val_map50"]
            is_best_str = " (BEST)"
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

        # Live terminal output block (matching Step 6 requirements)
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch}/{epochs}]")
        print("-" * 80)
        print(f"Train Loss:       {train_res['total_loss']:.4f} (Box: {train_res['box_loss']:.4f}, Obj: {train_res['obj_loss']:.4f}, Cls: {train_res['cls_loss']:.4f})")
        print(f"Val Loss:         {val_res['val_total_loss']:.4f} (Box: {val_res['val_box_loss']:.4f}, Obj: {val_res['val_obj_loss']:.4f}, Cls: {val_res['val_cls_loss']:.4f})")
        print(f"mAP50:            {val_res['val_map50']:.4f}")
        print(f"mAP50-95:         {val_res['val_map50_95']:.4f}")
        print(f"Recall:           {val_res['val_recall']:.4f}")
        print(f"Precision:        {val_res['val_precision']:.4f}")
        print(f"Mean IoU:         {val_res['val_mean_iou']:.4f}")
        print(f"Class Accuracy:   {val_res['val_cls_acc']:.1f}%")
        print(f"Images/sec:       {img_per_sec:.2f}")
        print(f"Epoch Time:       {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
        print(f"Learning Rate:    {current_lr:.6f}")
        print(f"VRAM Allocated:   {vram_alloc_gb:.2f} GB")
        print(f"VRAM Reserved:    {vram_res_gb:.2f} GB")
        print(f"Peak VRAM:        {peak_vram_gb:.2f} GB (Safety Headroom: {headroom_gb:.2f} GB)")
        print(f"GPU Utilization:  Active ({gpu_name})")
        print(f"Best Metric:      Val Loss = {best_val_loss:.4f}{is_best_str}")
        print("=" * 80 + "\n", flush=True)

    total_training_time = time.time() - pipeline_start

    # Final summary export
    final_metrics = {
        "final_epoch": epochs,
        "best_epoch": min(range(len(history)), key=lambda i: history[i]["val_total_loss"]) + 1 if history else epochs,
        "best_val_loss": best_val_loss,
        "best_map50": max((h.get("val_map50", 0.0) for h in history), default=0.0),
        "best_map50_95": max((h.get("val_map50_95", 0.0) for h in history), default=0.0),
        "best_recall": max((h.get("val_recall", 0.0) for h in history), default=0.0),
        "best_precision": max((h.get("val_precision", 0.0) for h in history), default=0.0),
        "final_train_loss": history[-1]["train_total_loss"] if history else None,
        "final_val_loss": history[-1]["val_total_loss"] if history else None,
        "total_training_time_sec": round(total_training_time, 2),
        "average_epoch_time_sec": round(total_training_time / max(epochs, 1), 2),
        "average_img_per_sec": round(len(train_dataset) / max(total_training_time / max(epochs, 1), 0.001), 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "checkpoints": {
            "best": str((out_path / "ird_best.pt").resolve()),
            "last": str((out_path / "ird_last.pt").resolve()),
        },
        "history_csv": str((out_path / "ird_history.csv").resolve()),
        "history_json": str((out_path / "ird_history.json").resolve()),
        "completed": True,
    }
    with open(out_path / "ird_final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("=" * 82)
    print(f"IRD V1.5 Training Completed Successfully ({epochs}/{epochs} Epochs)!")
    print(f"Total Training Time:      {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    print(f"Average Epoch Time:       {total_training_time/max(epochs,1):.2f}s")
    print(f"Best Validation Loss:     {best_val_loss:.4f} (Epoch {final_metrics['best_epoch']})")
    print(f"Best Proxy mAP50:         {final_metrics['best_map50']:.4f}")
    print(f"Best Proxy mAP50-95:      {final_metrics['best_map50_95']:.4f}")
    print(f"Peak VRAM:                {peak_vram_gb:.2f} GB (Safe Headroom: {headroom_gb:.2f} GB)")
    print(f"Checkpoints:              {out_path / 'ird_best.pt'}, {out_path / 'ird_last.pt'}")
    print(f"History Files:            {out_path / 'ird_history.csv'}, {out_path / 'ird_history.json'}")
    print("=" * 82, flush=True)

    return {
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs,
        "history": history,
        "final_metrics": final_metrics,
        "artifacts_dir": str(out_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IRD — IndianRoadDetection Production Training Pipeline (V1.5 & V2)")
    parser.add_argument("--data-dir", type=str, default="data/indian_road_yolo",
                        help="Path to YOLO dataset root (default: data/indian_road_yolo)")
    parser.add_argument("--output-dir", type=str, default="experiments/custom_model/final_training_50ep",
                        help="Output directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Training batch size (default: 12)")
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
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker processes (default: 0 for stable Windows execution)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reproducibility random seed (default: 42)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision training when CUDA/ROCm is available")
    parser.add_argument("--decoder-version", type=str, default="v2_smooth", choices=["v2_smooth", "v1_legacy"],
                        help="Box decoder parameterization version (default: v2_smooth)")
    parser.add_argument("--quality-obj", action="store_true", default=True,
                        help="Use quality-aware IoU objectness targets (default: True)")
    parser.add_argument("--no-quality-obj", dest="quality_obj", action="store_false",
                        help="Disable quality-aware objectness (use legacy binary targets)")
    parser.add_argument("--matcher-version", type=str, default="topk_adaptive_v2", choices=["v1_spatial", "topk_adaptive_v2"],
                        help="Target assignment matcher version (default: topk_adaptive_v2)")
    parser.add_argument("--class-balanced-loss", action="store_true", default=True,
                        help="Enable class-frequency balanced positive focal classification loss (default: True)")
    parser.add_argument("--small-obj-floor", action="store_true", default=False,
                        help="Enable Guaranteed Small-Object Presence Supervision (GSO floor 0.80 for scale < 96px)")
    parser.add_argument("--use-atd", action="store_true", default=True,
                        help="Enable Anisotropic Traffic Disentangler (ATD) in neck (default: True)")
    parser.add_argument("--use-ssdp", action="store_true", default=True,
                        help="Enable Selective Spatial Detail Pathway (SSDP) injecting P2 into N3 (default: True)")
    parser.add_argument("--use-fgbr", action="store_true", default=True,
                        help="Enable Fine-Grained Boundary Refiner (FGBR) on N3 box regression (default: True)")
    parser.add_argument("--use-quality", action="store_true", default=True,
                        help="Enable Localization Quality Branch (LQB) in head (default: True)")
    parser.add_argument("--use-cdg", action="store_true", default=True,
                        help="Enable Cross-Domain Gating (CDG) in neck (default: True)")
    parser.add_argument("--use-aux-one2one", action="store_true", default=True,
                        help="Enable Auxiliary One-to-One Matcher branch in loss (default: True)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to IRD checkpoint (.pt) to resume training from")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1.5", "v2"],
                        help="Model version: 'v1.5' (baseline) or 'v2' (Task-Aligned) (default: v1.5)")
    parser.add_argument("--loss-type", type=str, default="indian_road", choices=["indian_road", "task_aligned"],
                        help="Loss type: 'indian_road' (V1.5) or 'task_aligned' (V2 Varifocal + TAL) (default: indian_road)")
    parser.add_argument("--smoke-test", action="store_true", default=False,
                        help="Run short 2-epoch smoke test with small subset")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_epochs = args.epochs if args.epochs != 50 else 2
        print(f"[Smoke Test Mode] Setting epochs={smoke_epochs}, batch_size=4, max_train=8, max_val=4...")
        train_ird(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=smoke_epochs,
            batch_size=4,
            lr=args.lr,
            weight_decay=args.weight_decay,
            img_size=args.img_size,
            num_classes=args.num_classes,
            decoder_version=args.decoder_version,
            quality_aware_obj=args.quality_obj,
            matcher_version=args.matcher_version,
            class_balanced_loss=args.class_balanced_loss,
            small_obj_floor=args.small_obj_floor,
            use_atd=args.use_atd,
            use_ssdp=args.use_ssdp,
            use_fgbr=args.use_fgbr,
            use_quality=args.use_quality,
            use_cdg=args.use_cdg,
            use_aux_one2one=args.use_aux_one2one,
            version=args.version,
            loss_type=args.loss_type,
            device_str=args.device,
            workers=0,
            seed=args.seed,
            use_amp=args.amp,
            resume_path=args.resume,
            max_train_samples=8,
            max_val_samples=4,
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
            use_atd=args.use_atd,
            use_ssdp=args.use_ssdp,
            use_fgbr=args.use_fgbr,
            use_quality=args.use_quality,
            use_cdg=args.use_cdg,
            use_aux_one2one=args.use_aux_one2one,
            version=args.version,
            loss_type=args.loss_type,
            device_str=args.device,
            workers=args.workers,
            seed=args.seed,
            use_amp=args.amp,
            resume_path=args.resume,
        )


if __name__ == "__main__":
    main()
