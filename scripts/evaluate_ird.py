"""
Standalone Evaluation Pipeline for IRD V1 — IndianRoadDetection.

Official Model Name: IRD (IndianRoadDetection)
Model Class: IndianRoadDetector (pure PyTorch, exactly 4,241,529 parameters)

Key Capabilities:
1. Pure PyTorch & NumPy implementation (100% independent of Ultralytics).
2. Multi-scale head output decoding (P3/N3 stride 8, P4/N4 stride 16, P5/N5 stride 32).
3. Class-aware Non-Maximum Suppression (NMS).
4. Full COCO-standard metric computation:
   - Precision
   - Recall
   - mAP@0.50
   - mAP@0.50:0.95 (10 IoU thresholds: 0.50 to 0.95 in 0.05 increments)
   - Per-class AP@0.50 and AP@0.50:0.95
5. Inference latency and FPS benchmarking.
6. Machine-readable JSON export (experiments/custom_model/ird_evaluation.json).
7. Built-in synthetic unit tests for IoU, NMS, matching, and AP calculation.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector, build_detector

# Exact 12 benchmark classes (YOLOv8 & IRD V1)
BENCHMARK_CLASSES: List[str] = [
    "person",           # 0
    "rider",            # 1
    "car",              # 2
    "truck",            # 3
    "bus",              # 4
    "motorcycle",       # 5
    "bicycle",          # 6
    "autorickshaw",     # 7
    "animal",           # 8
    "vehicle fallback", # 9
    "traffic light",    # 10
    "traffic sign",     # 11
]

NUM_CLASSES = len(BENCHMARK_CLASSES)
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10).round(2)  # [0.50, 0.55, ..., 0.95]


# ==============================================================================
# 1. Box Geometry & NMS (Pure PyTorch)
# ==============================================================================

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] in (x1, y1, x2, y2).
        boxes2: Tensor of shape [M, 4] in (x1, y1, x2, y2).
        
    Returns:
        Tensor of shape [N, M] with IoU values in [0, 1].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)

    # Intersections [N, M]
    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    # Union [N, M]
    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union_area + 1e-16)


def pure_pytorch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Pure PyTorch implementation of Non-Maximum Suppression.
    Independent of Ultralytics and external C++ dependencies.
    
    Args:
        boxes: Tensor of shape [N, 4] in (x1, y1, x2, y2).
        scores: Tensor of shape [N] with confidence scores.
        iou_threshold: IoU threshold for suppression.
        
    Returns:
        Tensor of shape [K] containing indices of retained boxes.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = (ovr <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.65,
    max_coordinate: float = 10000.0,
) -> torch.Tensor:
    """
    Perform class-aware NMS by applying spatial offsets per class ID.
    
    Args:
        boxes: Tensor of shape [N, 4] in (x1, y1, x2, y2).
        scores: Tensor of shape [N].
        class_ids: Tensor of shape [N] with integer class labels.
        iou_threshold: NMS IoU threshold.
        max_coordinate: Coordinate shift offset to isolate classes spatially.
        
    Returns:
        Tensor of kept indices.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    offsets = class_ids.float().unsqueeze(1) * max_coordinate
    offset_boxes = boxes + offsets
    return pure_pytorch_nms(offset_boxes, scores, iou_threshold)


# ==============================================================================
# 2. Multi-Scale IRD Head Output Decoding
# ==============================================================================

def decode_ird_predictions(
    head_output: Any,
    img_size: int = 640,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode raw multi-scale IRD predictions across strides 8, 16, and 32.
    
    Formulation:
      cx = (gx + 2.0 * sigmoid(tx) - 0.5) * stride
      cy = (gy + 2.0 * sigmoid(ty) - 0.5) * stride
      w  = stride * exp(clamp(tw, -4.0, 4.0))
      h  = stride * exp(clamp(th, -4.0, 4.0))
      score = sigmoid(obj) * sigmoid(cls_c)
      
    Args:
        head_output: HeadOutput container from IndianRoadDetector.
        img_size: Resolution of input image (default: 640).
        conf_threshold: Candidate score filtering threshold.
        iou_threshold: NMS IoU threshold.
        device: Active torch device.
        
    Returns:
        Tuple of (retained_boxes [K, 4], retained_scores [K], retained_class_ids [K]).
    """
    all_boxes_list: List[torch.Tensor] = []
    all_scores_list: List[torch.Tensor] = []
    all_classes_list: List[torch.Tensor] = []

    box_preds = head_output.box_preds  # 3 tensors: [B, 4, H, W]
    obj_preds = head_output.obj_preds  # 3 tensors: [B, 1, H, W]
    cls_preds = head_output.cls_preds  # 3 tensors: [B, 12, H, W]
    strides = head_output.strides      # [8, 16, 32]

    # Process first item in batch [B=1]
    for scale_idx in range(len(strides)):
        stride = strides[scale_idx]
        b_pred = box_preds[scale_idx][0]  # [4, H, W]
        o_pred = obj_preds[scale_idx][0]  # [1, H, W]
        c_pred = cls_preds[scale_idx][0]  # [12, H, W]

        H, W = b_pred.shape[1], b_pred.shape[2]

        # Generate grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        tx = b_pred[0]
        ty = b_pred[1]
        tw = b_pred[2]
        th = b_pred[3]

        # Decode centers and dimensions
        cx = (grid_x + torch.sigmoid(tx) * 2.0 - 0.5) * stride
        cy = (grid_y + torch.sigmoid(ty) * 2.0 - 0.5) * stride
        w = stride * torch.exp(tw.clamp(min=-4.0, max=4.0))
        h = stride * torch.exp(th.clamp(min=-4.0, max=4.0))

        x1 = (cx - w / 2.0).clamp(min=0.0, max=float(img_size))
        y1 = (cy - h / 2.0).clamp(min=0.0, max=float(img_size))
        x2 = (cx + w / 2.0).clamp(min=0.0, max=float(img_size))
        y2 = (cy + h / 2.0).clamp(min=0.0, max=float(img_size))

        scale_boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)  # [H*W, 4]

        # Probabilities
        obj_prob = torch.sigmoid(o_pred[0]).reshape(-1, 1)  # [H*W, 1]
        cls_prob = torch.sigmoid(c_pred).permute(1, 2, 0).reshape(-1, NUM_CLASSES)  # [H*W, 12]

        # Combined detection confidence per class
        scale_scores = obj_prob * cls_prob  # [H*W, 12]

        # Filter by candidate confidence threshold
        max_scores, class_ids = scale_scores.max(dim=-1)
        valid_mask = max_scores >= conf_threshold

        if valid_mask.any():
            all_boxes_list.append(scale_boxes[valid_mask])
            all_scores_list.append(max_scores[valid_mask])
            all_classes_list.append(class_ids[valid_mask])

    if not all_boxes_list:
        empty = torch.empty((0,), device=device)
        return torch.empty((0, 4), device=device), empty, torch.empty((0,), dtype=torch.long, device=device)

    all_boxes = torch.cat(all_boxes_list, dim=0)
    all_scores = torch.cat(all_scores_list, dim=0)
    all_classes = torch.cat(all_classes_list, dim=0)

    # Class-aware NMS
    keep_indices = class_aware_nms(all_boxes, all_scores, all_classes, iou_threshold=iou_threshold)

    return all_boxes[keep_indices], all_scores[keep_indices], all_classes[keep_indices]


# ==============================================================================
# 3. Metric Computation Engine (Precision, Recall, mAP50, mAP50-95)
# ==============================================================================

def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute 101-point interpolated Average Precision (standard COCO evaluation).
    
    Args:
        recalls: 1D array of recall values.
        precisions: 1D array of precision values.
        
    Returns:
        Float AP score in [0, 1].
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Append boundary points
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute monotonic precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101 equidistant recall points
    recall_levels = np.linspace(0.0, 1.0, 101)
    interp_precisions = np.zeros_like(recall_levels)

    for idx, r in enumerate(recall_levels):
        match = np.where(mrec >= r)[0]
        if len(match) > 0:
            interp_precisions[idx] = mpre[match[0]]

    return float(np.mean(interp_precisions))


def evaluate_class_predictions(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    class_id: int,
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Match predictions and ground-truths for one class across all 10 IoU thresholds.
    
    Args:
        preds: List of prediction dicts {'img_id': int, 'score': float, 'box': [4]}.
        gts: List of ground-truth dicts {'img_id': int, 'box': [4]}.
        class_id: Target class integer ID.
        iou_thresholds: Array of IoU evaluation thresholds.
        
    Returns:
        Dict containing precision, recall, AP@0.50, AP@0.50:0.95, and per-threshold APs.
    """
    n_gt = len(gts)
    n_pred = len(preds)

    if n_gt == 0:
        return {
            "n_gt": 0,
            "n_pred": n_pred,
            "precision_50": 0.0,
            "recall_50": 0.0,
            "ap50": 0.0,
            "ap50_95": 0.0,
            "ap_per_iou": {f"iou_{t:.2f}": 0.0 for t in iou_thresholds},
        }

    if n_pred == 0:
        return {
            "n_gt": n_gt,
            "n_pred": 0,
            "precision_50": 0.0,
            "recall_50": 0.0,
            "ap50": 0.0,
            "ap50_95": 0.0,
            "ap_per_iou": {f"iou_{t:.2f}": 0.0 for t in iou_thresholds},
        }

    # Group ground truths by image ID
    gt_by_img: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for gt in gts:
        gt_by_img[gt["img_id"]].append(torch.tensor(gt["box"], dtype=torch.float32))

    # Sort predictions by confidence score descending
    sorted_preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    ap_per_iou: Dict[str, float] = {}
    p50 = 0.0
    r50 = 0.0

    for t_idx, iou_thresh in enumerate(iou_thresholds):
        # Track matched ground truths per image for this IoU threshold
        matched_gt: Dict[int, Set[int]] = defaultdict(set)
        tp = np.zeros(n_pred, dtype=np.float32)
        fp = np.zeros(n_pred, dtype=np.float32)

        for p_idx, pred in enumerate(sorted_preds):
            img_id = pred["img_id"]
            p_box = torch.tensor(pred["box"], dtype=torch.float32).unsqueeze(0)

            if img_id not in gt_by_img:
                fp[p_idx] = 1.0
                continue

            img_gts = torch.stack(gt_by_img[img_id], dim=0)
            ious = box_iou_xyxy(p_box, img_gts).squeeze(0)  # [M]

            best_iou, best_gt_idx = ious.max(dim=0)
            best_iou_val = best_iou.item()
            best_gt_idx_val = best_gt_idx.item()

            if best_iou_val >= iou_thresh:
                if best_gt_idx_val not in matched_gt[img_id]:
                    tp[p_idx] = 1.0
                    matched_gt[img_id].add(best_gt_idx_val)
                else:
                    fp[p_idx] = 1.0  # Duplicate detection
            else:
                fp[p_idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(n_gt)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-16)

        ap = compute_ap_101(recalls, precisions)
        ap_per_iou[f"iou_{iou_thresh:.2f}"] = ap

        if abs(iou_thresh - 0.50) < 1e-4:
            p50 = float(precisions[-1]) if len(precisions) > 0 else 0.0
            r50 = float(recalls[-1]) if len(recalls) > 0 else 0.0

    ap50 = ap_per_iou.get("iou_0.50", 0.0)
    ap50_95 = float(np.mean(list(ap_per_iou.values())))

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "precision_50": p50,
        "recall_50": r50,
        "ap50": ap50,
        "ap50_95": ap50_95,
        "ap_per_iou": ap_per_iou,
    }


# ==============================================================================
# 4. Dataset Loader (YOLO Format from data.yaml)
# ==============================================================================

def parse_data_yaml(yaml_path: Path) -> Tuple[Path, List[str]]:
    """
    Parse YOLO data.yaml configuration file.
    
    Returns:
        Tuple of (val_images_dir, class_names).
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    path_val: Optional[str] = None
    val_val: Optional[str] = None
    class_names: List[str] = []
    in_names = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("path:"):
            path_val = stripped.split("path:", 1)[1].strip().strip('"').strip("'")
        elif stripped.startswith("val:"):
            val_val = stripped.split("val:", 1)[1].strip().strip('"').strip("'")
        elif stripped.startswith("names:"):
            in_names = True
            # Check for inline list e.g. names: ['a', 'b']
            if "[" in stripped and "]" in stripped:
                raw_list = stripped.split("[", 1)[1].rsplit("]", 1)[0]
                class_names = [x.strip().strip("'").strip('"') for x in raw_list.split(",")]
                in_names = False
        elif in_names:
            if ":" in stripped:
                parts = stripped.split(":", 1)
                class_names.append(parts[1].strip().strip("'").strip('"'))
            elif stripped.startswith("-"):
                class_names.append(stripped.lstrip("-").strip().strip("'").strip('"'))
            else:
                in_names = False

    # Resolve val directory
    base_dir = Path(path_val) if path_val else yaml_path.parent
    val_dir_rel = Path(val_val) if val_val else Path("images/val")

    val_images_dir = base_dir / val_dir_rel
    if not val_images_dir.exists():
        # Fallback relative to data.yaml
        alt = yaml_path.parent / val_dir_rel
        if alt.exists():
            val_images_dir = alt

    # If class_names empty, use benchmark defaults
    if not class_names:
        class_names = BENCHMARK_CLASSES

    return val_images_dir, class_names


# ==============================================================================
# 5. Full Evaluation Pipeline
# ==============================================================================

def run_evaluation(
    weights_path: Optional[str] = None,
    data_yaml_path: str = "data/indian_road_yolo/data.yaml",
    img_size: int = 640,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    device_str: str = "auto",
    output_json_path: str = "experiments/custom_model/ird_evaluation.json",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute full IRD V1 benchmark evaluation.
    """
    # 1. Resolve Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 65)
    print("IRD V1 Standalone Benchmark Evaluation")
    print(f"Device:               {device}")
    print(f"Image Resolution:     {img_size}x{img_size}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"NMS IoU Threshold:    {iou_threshold}")
    print("=" * 65)

    # 2. Build IRD Model
    model = build_detector(num_classes=NUM_CLASSES)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded IndianRoadDetector: {n_params:,} trainable parameters")

    # 3. Load Checkpoint Weights
    weights_loaded = False
    if weights_path and Path(weights_path).exists():
        ckpt_p = Path(weights_path)
        print(f"Loading weights from: {ckpt_p}...")
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        weights_loaded = True
        print("Checkpoint weights successfully restored.")
    else:
        print(f"WARNING: Weights path '{weights_path}' not found. Evaluating with initialized weights.")

    # 4. Resolve Dataset Paths
    yaml_p = Path(data_yaml_path)
    if not yaml_p.exists():
        # Fallback candidates
        candidates = [
            Path("data/benchmark_test/data.yaml"),
            Path("data/indian_road_yolo/data.yaml"),
            Path("data/sample_16/data.yaml"),
        ]
        for cand in candidates:
            if cand.exists():
                yaml_p = cand
                print(f"Specified YAML not found. Using fallback: {yaml_p}")
                break

    if not yaml_p.exists():
        raise FileNotFoundError(f"Cannot find dataset configuration at {data_yaml_path}")

    val_img_dir, class_names = parse_data_yaml(yaml_p)
    base_dir = val_img_dir.parent.parent
    val_lbl_dir = base_dir / "labels" / val_img_dir.name

    print(f"Validation Images Dir: {val_img_dir}")
    print(f"Validation Labels Dir: {val_lbl_dir}")

    # Gather images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths: List[Path] = []
    for ext in extensions:
        image_paths.extend(sorted(list(val_img_dir.glob(ext))))

    if max_samples and max_samples > 0:
        image_paths = image_paths[:max_samples]

    n_images = len(image_paths)
    if n_images == 0:
        raise RuntimeError(f"No validation images found in {val_img_dir}")

    print(f"Evaluating {n_images} images across {NUM_CLASSES} classes...\n")

    # 5. Inference & Prediction Collection
    all_preds_per_class: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(NUM_CLASSES)}
    all_gts_per_class: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(NUM_CLASSES)}

    latencies: List[float] = []

    for img_idx, img_p in enumerate(image_paths):
        # Load and preprocess image
        with Image.open(img_p) as img_raw:
            orig_w, orig_h = img_raw.size
            img_rgb = img_raw.convert("RGB").resize((img_size, img_size))

        img_np = np.array(img_rgb, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Timed inference
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            head_out = model(img_tensor)
            boxes, scores, class_ids = decode_ird_predictions(
                head_out,
                img_size=img_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Scale predictions back to original image space
        scale_x = orig_w / float(img_size)
        scale_y = orig_h / float(img_size)

        for b, s, c in zip(boxes.cpu().numpy(), scores.cpu().numpy(), class_ids.cpu().numpy()):
            c_int = int(c)
            if 0 <= c_int < NUM_CLASSES:
                scaled_box = [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                all_preds_per_class[c_int].append({
                    "img_id": img_idx,
                    "score": float(s),
                    "box": scaled_box,
                })

        # Load Ground Truths for this image
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_cls = int(parts[0])
                        cx = float(parts[1]) * orig_w
                        cy = float(parts[2]) * orig_h
                        w = float(parts[3]) * orig_w
                        h = float(parts[4]) * orig_h
                        gt_box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
                        if 0 <= gt_cls < NUM_CLASSES:
                            all_gts_per_class[gt_cls].append({
                                "img_id": img_idx,
                                "box": gt_box,
                            })

        if (img_idx + 1) % 10 == 0 or (img_idx + 1) == n_images:
            avg_lat = np.mean(latencies)
            print(f"Processed [{img_idx + 1:>5}/{n_images}] images | Latency: {avg_lat:.2f} ms/img | FPS: {1000.0/avg_lat:.1f}", end="\r", flush=True)

    print()

    # 6. Metric Computation
    per_class_results: Dict[str, Any] = {}
    ap50_list: List[float] = []
    ap50_95_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []

    for c in range(NUM_CLASSES):
        cls_name = BENCHMARK_CLASSES[c]
        res = evaluate_class_predictions(
            preds=all_preds_per_class[c],
            gts=all_gts_per_class[c],
            class_id=c,
            iou_thresholds=IOU_THRESHOLDS,
        )
        per_class_results[cls_name] = res

        # Include classes in global mAP if they have ground truths
        if res["n_gt"] > 0:
            ap50_list.append(res["ap50"])
            ap50_95_list.append(res["ap50_95"])
            precision_list.append(res["precision_50"])
            recall_list.append(res["recall_50"])

    mean_precision = float(np.mean(precision_list)) if precision_list else 0.0
    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
    mAP50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    mAP50_95 = float(np.mean(ap50_95_list)) if ap50_95_list else 0.0

    avg_ms = float(np.mean(latencies))
    fps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0

    # 7. Print Required Benchmark Report
    print("\n" + "=" * 40)
    print("IRD V1 BENCHMARK RESULTS")
    print("=" * 24)
    print(f"Images evaluated: {n_images}")
    print(f"Parameters:       {n_params:,}")
    print(f"Image size:       {img_size}")
    print()
    print(f"Precision:        {mean_precision:.3f}")
    print(f"Recall:           {mean_recall:.3f}")
    print(f"mAP@0.50:         {mAP50:.3f}")
    print(f"mAP@0.50:0.95:    {mAP50_95:.3f}")
    print()
    print("Per-class AP@0.50:")
    for cls_name in BENCHMARK_CLASSES:
        ap = per_class_results[cls_name]["ap50"]
        print(f"{cls_name:<18} {ap:.3f}")
    print()
    print("Inference latency:")
    print(f"Average ms/image: {avg_ms:.2f}")
    print(f"FPS:              {fps:.2f}")
    print("=" * 40 + "\n")

    # 8. Save Machine-Readable JSON Export
    json_record = {
        "model_name": "IRD — IndianRoadDetection (V1)",
        "checkpoint_path": str(weights_path) if weights_path else None,
        "dataset_path": str(yaml_p.resolve()),
        "image_size": img_size,
        "number_of_images": n_images,
        "parameter_count": n_params,
        "precision": round(mean_precision, 4),
        "recall": round(mean_recall, 4),
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "per_class_ap50": {cls: round(per_class_results[cls]["ap50"], 4) for cls in BENCHMARK_CLASSES},
        "per_class_ap50_95": {cls: round(per_class_results[cls]["ap50_95"], 4) for cls in BENCHMARK_CLASSES},
        "average_latency_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
    }

    out_json_p = Path(output_json_path)
    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_p, "w", encoding="utf-8") as f:
        json.dump(json_record, f, indent=2)

    print(f"Machine-readable results saved to: {out_json_p.resolve()}")

    return json_record


# ==============================================================================
# 6. Built-In Synthetic Unit Tests
# ==============================================================================

def run_unit_tests() -> bool:
    """
    Run self-contained unit tests verifying:
    1. IoU calculation
    2. Non-Maximum Suppression (NMS)
    3. Prediction / GT matching
    4. 101-point Average Precision (AP) calculation
    """
    print("\n" + "=" * 50)
    print("Running IRD Evaluator Built-In Unit Tests")
    print("=" * 50)

    # 1. IoU Calculation Test
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]], dtype=torch.float32)
    ious = box_iou_xyxy(b1, b2).squeeze(0).numpy()

    # b1 and b2[0] are identical -> IoU = 1.0
    assert abs(ious[0] - 1.0) < 1e-5, f"Expected 1.0, got {ious[0]}"
    # b1 and b2[1] overlap 50x100 / 150x100 -> IoU = 50 / 150 = 0.3333
    assert abs(ious[1] - 1.0 / 3.0) < 1e-4, f"Expected 0.3333, got {ious[1]}"
    # b1 and b2[2] do not overlap -> IoU = 0.0
    assert abs(ious[2] - 0.0) < 1e-5, f"Expected 0.0, got {ious[2]}"
    print("PASS: IoU calculation verified.")

    # 2. Pure PyTorch NMS Test
    overlap_boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],  # Box A
        [1.0, 1.0, 10.0, 10.0],  # Box B (heavy overlap with A)
        [50.0, 50.0, 60.0, 60.0],  # Box C (isolated)
    ], dtype=torch.float32)
    overlap_scores = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32)

    keep = pure_pytorch_nms(overlap_boxes, overlap_scores, iou_threshold=0.5).numpy()
    assert 0 in keep and 2 in keep and 1 not in keep, f"NMS failed to suppress duplicate: keep={keep}"
    print("PASS: Non-Maximum Suppression (NMS) verified.")

    # 3. Prediction / Ground Truth Matching Test
    mock_preds = [
        {"img_id": 1, "score": 0.95, "box": [0.0, 0.0, 10.0, 10.0]},
        {"img_id": 1, "score": 0.85, "box": [0.0, 0.0, 10.0, 10.0]},  # duplicate
        {"img_id": 1, "score": 0.70, "box": [50.0, 50.0, 60.0, 60.0]},  # FP
    ]
    mock_gts = [
        {"img_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},  # 1 GT
    ]
    match_res = evaluate_class_predictions(mock_preds, mock_gts, class_id=0, iou_thresholds=np.array([0.50]))
    assert match_res["n_gt"] == 1
    assert match_res["n_pred"] == 3
    assert abs(match_res["recall_50"] - 1.0) < 1e-5  # matched the 1 GT
    print("PASS: Prediction / GT greedy matching verified.")

    # 4. AP Calculation Test
    # Perfect detector: Recall = 1.0, Precision = 1.0 -> AP = 1.0
    perfect_rec = np.array([0.2, 0.5, 1.0])
    perfect_prec = np.array([1.0, 1.0, 1.0])
    ap_perfect = compute_ap_101(perfect_rec, perfect_prec)
    assert abs(ap_perfect - 1.0) < 1e-4, f"Expected 1.0, got {ap_perfect}"

    # Half precision detector
    half_rec = np.array([0.5, 1.0])
    half_prec = np.array([0.5, 0.5])
    ap_half = compute_ap_101(half_rec, half_prec)
    assert abs(ap_half - 0.5) < 1e-4, f"Expected 0.5, got {ap_half}"
    print("PASS: 101-point COCO Average Precision (AP) calculation verified.")

    print("=" * 50)
    print("ALL EVALUATION UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 50 + "\n")
    return True


# ==============================================================================
# 7. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRD V1 Benchmark Evaluation Pipeline")
    parser.add_argument("--weights", type=str, default="experiments/custom_model/ird_v1/ird_best.pt",
                        help="Path to IRD checkpoint (.pt)")
    parser.add_argument("--data", type=str, default="data/indian_road_yolo/data.yaml",
                        help="Path to dataset data.yaml")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image resolution (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence score filtering threshold (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.65,
                        help="NMS IoU threshold (default: 0.65)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--output-json", type=str, default="experiments/custom_model/ird_evaluation.json",
                        help="Path to save evaluation JSON")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of validation images to evaluate (optional)")
    parser.add_argument("--run-unit-tests", action="store_true", default=False,
                        help="Run self-contained unit tests before evaluation")

    args = parser.parse_args()

    if args.run_unit_tests:
        run_unit_tests()

    # If unit tests flag was given alone and weights don't exist, exit cleanly
    if args.run_unit_tests and not Path(args.data).exists() and not Path("data/benchmark_test/data.yaml").exists():
        sys.exit(0)

    run_evaluation(
        weights_path=args.weights,
        data_yaml_path=args.data,
        img_size=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device_str=args.device,
        output_json_path=args.output_json,
        max_samples=args.max_samples,
    )
