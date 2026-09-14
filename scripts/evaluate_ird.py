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
from typing import Any, Dict, List, Optional, Tuple, Union

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


from src.models.box_coder import (
    NUM_CLASSES,
    class_aware_nms,
    decode_ird_predictions_authoritative,
    pure_pytorch_nms,
)


def decode_ird_predictions(
    head_output: Any,
    img_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    max_det: int = 300,
    obj_gate: Optional[float] = None,
    decoder_version: str = "v2_smooth",
    device: torch.device = torch.device("cpu"),
    return_diagnostics: bool = False,
    score_mode: str = "sqrt_quality",
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """
    Authoritative decoding pipeline for IRD multi-scale predictions.
    Delegates directly to src.models.box_coder.decode_ird_predictions_authoritative.
    """
    return decode_ird_predictions_authoritative(
        head_output=head_output,
        img_size=img_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        obj_gate=obj_gate,
        decoder_version=decoder_version,
        device=device,
        return_diagnostics=return_diagnostics,
        score_mode=score_mode,
    )


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

    # Group ground truths by image ID and pre-stack once
    gt_by_img_raw: Dict[int, List[List[float]]] = defaultdict(list)
    for gt in gts:
        gt_by_img_raw[gt["img_id"]].append(gt["box"])
    gt_tensors: Dict[int, torch.Tensor] = {
        img_id: torch.tensor(boxes, dtype=torch.float32) for img_id, boxes in gt_by_img_raw.items()
    }

    # Group predictions by image ID
    preds_by_img: Dict[int, List[Tuple[int, float, List[float]]]] = defaultdict(list)
    for p_idx, p in enumerate(preds):
        preds_by_img[p["img_id"]].append((p_idx, p["score"], p["box"]))

    # Precompute best IoU and GT index for each prediction via image-level vectorized matrix operations
    best_iou_arr = np.zeros(n_pred, dtype=np.float32)
    best_gt_arr = np.full(n_pred, -1, dtype=np.int32)
    img_id_arr = np.zeros(n_pred, dtype=np.int32)
    score_arr = np.zeros(n_pred, dtype=np.float32)

    for img_id, p_list in preds_by_img.items():
        p_indices = [x[0] for x in p_list]
        scores = [x[1] for x in p_list]
        boxes = [x[2] for x in p_list]

        img_id_arr[p_indices] = img_id
        score_arr[p_indices] = scores

        if img_id not in gt_tensors:
            continue

        b1 = torch.tensor(boxes, dtype=torch.float32)
        b2 = gt_tensors[img_id]

        area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0.0) * (b1[:, 3] - b1[:, 1]).clamp(min=0.0)
        area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0.0) * (b2[:, 3] - b2[:, 1]).clamp(min=0.0)
        inter_x1 = torch.maximum(b1[:, None, 0], b2[None, :, 0])
        inter_y1 = torch.maximum(b1[:, None, 1], b2[None, :, 1])
        inter_x2 = torch.minimum(b1[:, None, 2], b2[None, :, 2])
        inter_y2 = torch.minimum(b1[:, None, 3], b2[None, :, 3])
        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h
        union_area = area1[:, None] + area2[None, :] - inter_area
        ious = inter_area / (union_area + 1e-16)
        best_ious, best_indices = ious.max(dim=1)
        best_iou_arr[p_indices] = best_ious.cpu().numpy()
        best_gt_arr[p_indices] = best_indices.cpu().numpy()

    # Sort all predictions globally by score descending
    sort_order = np.argsort(-score_arr)
    sorted_best_iou = best_iou_arr[sort_order]
    sorted_best_gt = best_gt_arr[sort_order]
    sorted_img_ids = img_id_arr[sort_order]

    ap_per_iou: Dict[str, float] = {}
    p50 = 0.0
    r50 = 0.0

    for t_idx, iou_thresh in enumerate(iou_thresholds):
        matched_gt: set = set()
        tp = np.zeros(n_pred, dtype=np.float32)
        fp = np.zeros(n_pred, dtype=np.float32)

        for i in range(n_pred):
            gid = sorted_best_gt[i]
            if gid >= 0 and sorted_best_iou[i] >= iou_thresh:
                key = (sorted_img_ids[i], gid)
                if key not in matched_gt:
                    tp[i] = 1.0
                    matched_gt.add(key)
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

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
    iou_threshold: float = 0.50,
    max_det: int = 300,
    device_str: str = "auto",
    output_json_path: str = "experiments/custom_model/ird_evaluation.json",
    max_samples: Optional[int] = None,
    obj_gate: Optional[float] = None,
    decoder_version: str = "v2_smooth",
    use_atd: Optional[bool] = None,
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
    print(f"Max Detections / Img: {max_det}")
    print("=" * 65)

    # 2. Inspect Checkpoint & Auto-detect Architecture Configuration
    weights_loaded = False
    use_atd_detected = use_atd if use_atd is not None else False
    loaded_sd = None
    if weights_path and Path(weights_path).exists():
        ckpt_p = Path(weights_path)
        print(f"Inspecting weights from: {ckpt_p}...")
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            loaded_sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            loaded_sd = ckpt["model"]
        elif isinstance(ckpt, dict):
            loaded_sd = ckpt
        if loaded_sd is not None:
            clean_sd = {k.replace("_orig_mod.", ""): v for k, v in loaded_sd.items()}
            if any("atd" in k for k in clean_sd.keys()):
                use_atd_detected = True
                print("Detected AnisotropicTrafficDisentangler (ATD) weights in checkpoint.")
            elif isinstance(ckpt, dict) and "config" in ckpt and ckpt["config"].get("use_atd", False):
                use_atd_detected = True

            use_ssdp_detected = any("ssdp" in k for k in clean_sd.keys())
            if isinstance(ckpt, dict) and "config" in ckpt:
                use_ssdp_detected = use_ssdp_detected or ckpt["config"].get("use_ssdp", False)
            if use_ssdp_detected:
                print("Detected SelectiveSpatialDetailPathway (SSDP) weights in checkpoint.")

            use_fgbr_detected = any("fgbr" in k for k in clean_sd.keys())
            if isinstance(ckpt, dict) and "config" in ckpt:
                use_fgbr_detected = use_fgbr_detected or ckpt["config"].get("use_fgbr", False)
            if use_fgbr_detected:
                print("Detected FineGrainedBoundaryRefiner (FGBR) weights in checkpoint.")

            loaded_sd = clean_sd

    # 3. Build IRD Model with appropriate architecture
    model = build_detector(
        num_classes=NUM_CLASSES,
        use_atd=use_atd_detected,
        use_ssdp=use_ssdp_detected,
        use_fgbr=use_fgbr_detected,
    )
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded IndianRoadDetector: {n_params:,} trainable parameters (use_atd={use_atd_detected}, use_ssdp={use_ssdp_detected}, use_fgbr={use_fgbr_detected})")

    if loaded_sd is not None:
        model.load_state_dict(loaded_sd)
        weights_loaded = True
        print("Checkpoint weights successfully restored.")
    elif weights_path:
        print(f"WARNING: Weights path '{weights_path}' not found. Evaluating with initialized weights.")

    # 4. Resolve Dataset Paths
    yaml_p = Path(data_yaml_path)
    if not yaml_p.exists():
        # Fallback candidates
        candidates = [
            Path("data/benchmark_test/data.yaml"),
            Path("data/indian_road_yolo/data.yaml"),
        ]
        for c in candidates:
            if c.exists():
                yaml_p = c
                break

    if not yaml_p.exists():
        raise FileNotFoundError(f"Cannot find dataset configuration: {data_yaml_path}")

    val_images_dir, class_names = parse_data_yaml(yaml_p)
    val_lbl_dir = val_images_dir.parent.parent / "labels" / val_images_dir.name
    if not val_lbl_dir.exists():
        val_lbl_dir = val_images_dir.parent / "labels"

    # Find validation images
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    val_images: List[Path] = []
    if val_images_dir.exists():
        for ext in image_extensions:
            val_images.extend(val_images_dir.glob(f"*{ext}"))
            val_images.extend(val_images_dir.glob(f"*{ext.upper()}"))
    val_images = sorted(list(set(val_images)))

    if max_samples and max_samples > 0:
        val_images = val_images[:max_samples]

    n_images = len(val_images)
    print(f"Found {n_images} validation images in: {val_images_dir.resolve()}")
    if n_images == 0:
        raise FileNotFoundError(f"No validation images found in: {val_images_dir}")

    # 5. Run Evaluation Loop
    print("\nRunning inference & prediction gathering...")
    latencies: List[float] = []

    all_preds_per_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_gts_per_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    # Warmup
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        _ = model(dummy)

    for img_idx, img_p in enumerate(val_images):
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
                max_det=max_det,
                obj_gate=obj_gate,
                decoder_version=decoder_version,
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

    for c_id, c_name in enumerate(BENCHMARK_CLASSES):
        preds = all_preds_per_class[c_id]
        gts = all_gts_per_class[c_id]
        res = evaluate_class_predictions(preds, gts, class_id=c_id, iou_thresholds=IOU_THRESHOLDS)
        per_class_results[c_name] = res

        if res["n_gt"] > 0:
            ap50_list.append(res["ap50"])
            ap50_95_list.append(res["ap50_95"])
            precision_list.append(res["precision_50"])
            recall_list.append(res["recall_50"])

    mAP50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    mAP50_95 = float(np.mean(ap50_95_list)) if ap50_95_list else 0.0
    mean_precision = float(np.mean(precision_list)) if precision_list else 0.0
    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    fps = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0

    total_preds = sum(len(preds) for preds in all_preds_per_class.values())
    total_gts = sum(len(gts) for gts in all_gts_per_class.values())
    per_class_preds_count = {cls_name: len(all_preds_per_class[c_id]) for c_id, cls_name in enumerate(BENCHMARK_CLASSES)}
    per_class_gts_count = {cls_name: len(all_gts_per_class[c_id]) for c_id, cls_name in enumerate(BENCHMARK_CLASSES)}

    # Size-based distributions (COCO standard: small < 32^2, medium 32^2 to 96^2, large > 96^2)
    pred_size_counts = {"small": 0, "medium": 0, "large": 0}
    for preds in all_preds_per_class.values():
        for p in preds:
            b = p["box"]
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            if area < 32 * 32:
                pred_size_counts["small"] += 1
            elif area <= 96 * 96:
                pred_size_counts["medium"] += 1
            else:
                pred_size_counts["large"] += 1

    gt_size_counts = {"small": 0, "medium": 0, "large": 0}
    for gts in all_gts_per_class.values():
        for g in gts:
            b = g["box"]
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            if area < 32 * 32:
                gt_size_counts["small"] += 1
            elif area <= 96 * 96:
                gt_size_counts["medium"] += 1
            else:
                gt_size_counts["large"] += 1

    # 7. Print Console Report
    print("\n" + "=" * 40)
    print("IRD V1 BENCHMARK EVALUATION RESULTS")
    print("=" * 40)
    print(f"Model:            IRD (IndianRoadDetection)")
    print(f"Parameters:       {n_params:,}")
    print(f"Images evaluated: {n_images}")
    print(f"Total Preds:      {total_preds}")
    print(f"Total GTs:        {total_gts}")
    print(f"Precision:        {mean_precision:.3f}")
    print(f"Recall:           {mean_recall:.3f}")
    print(f"mAP@0.50:         {mAP50:.3f}")
    print(f"mAP@0.50:0.95:    {mAP50_95:.3f}")
    print()
    print(f"GT Sizes:   Small: {gt_size_counts['small']}, Med: {gt_size_counts['medium']}, Large: {gt_size_counts['large']}")
    print(f"Pred Sizes: Small: {pred_size_counts['small']}, Med: {pred_size_counts['medium']}, Large: {pred_size_counts['large']}")
    print()
    print("Per-class AP@0.50 & Predictions:")
    for cls_name in BENCHMARK_CLASSES:
        ap = per_class_results[cls_name]["ap50"]
        n_p = per_class_preds_count[cls_name]
        n_g = per_class_gts_count[cls_name]
        print(f"{cls_name:<18} AP50: {ap:.3f} | Preds: {n_p:>5} | GT: {n_g:>5}")
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
        "total_predictions": total_preds,
        "total_ground_truths": total_gts,
        "pred_size_counts": pred_size_counts,
        "gt_size_counts": gt_size_counts,
        "precision": round(mean_precision, 4),
        "recall": round(mean_recall, 4),
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "per_class_predictions": per_class_preds_count,
        "per_class_ground_truths": per_class_gts_count,
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
    5. Class-aware NMS & multi-condition candidate filtering
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

    # 5. Class-Aware NMS & Synthetic Multi-Condition Test
    synth_boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],  # Box 0: class 0, score 0.90
        [12.0, 12.0, 52.0, 52.0],  # Box 1: class 0, score 0.80 (same class, high overlap with Box 0)
        [11.0, 11.0, 51.0, 51.0],  # Box 2: class 1, score 0.85 (different class, same location -> must keep!)
        [10.0, 10.0, 50.0, 50.0],  # Box 3: class 0, score 0.70 (exact duplicate of Box 0 -> must suppress!)
        [100.0, 100.0, 150.0, 150.0], # Box 4: class 2, score 0.05 (low-confidence box -> filtered out)
    ], dtype=torch.float32)
    synth_scores = torch.tensor([0.90, 0.80, 0.85, 0.70, 0.05], dtype=torch.float32)
    synth_classes = torch.tensor([0, 0, 1, 0, 2], dtype=torch.long)

    # Apply confidence filtering at 0.25
    conf_filter = synth_scores >= 0.25
    f_boxes = synth_boxes[conf_filter]
    f_scores = synth_scores[conf_filter]
    f_classes = synth_classes[conf_filter]

    # Verify low-confidence box 4 was excluded
    assert len(f_boxes) == 4, f"Expected 4 boxes after conf>=0.25, got {len(f_boxes)}"

    # Run class-aware NMS
    keep_synth = class_aware_nms(f_boxes, f_scores, f_classes, iou_threshold=0.50, max_det=300).tolist()
    assert 0 in keep_synth, "Box 0 (class 0, 0.90) must be kept"
    assert 2 in keep_synth, "Box 2 (class 1, 0.85) must be kept (different class)"
    assert 1 not in keep_synth, "Box 1 (class 0, 0.80) must be suppressed by Box 0"
    assert 3 not in keep_synth, "Box 3 (class 0, 0.70 duplicate) must be suppressed"
    assert len(keep_synth) == 2, f"Expected exactly 2 kept detections, got {len(keep_synth)}"

    # Verify max_det capping
    many_boxes = torch.stack([
        torch.arange(0, 1000, 10, dtype=torch.float32),
        torch.arange(0, 1000, 10, dtype=torch.float32),
        torch.arange(5, 1005, 10, dtype=torch.float32),
        torch.arange(5, 1005, 10, dtype=torch.float32),
    ], dim=-1)
    many_scores = torch.linspace(0.99, 0.26, 100)
    many_classes = torch.zeros(100, dtype=torch.long)
    keep_max = class_aware_nms(many_boxes, many_scores, many_classes, iou_threshold=0.50, max_det=10)
    assert len(keep_max) == 10, f"Expected 10 detections under max_det=10, got {len(keep_max)}"
    print("PASS: Class-aware NMS & synthetic multi-condition test verified successfully.")

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
    parser.add_argument("--iou", type=float, default=0.50,
                        help="NMS IoU threshold (default: 0.50)")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image (default: 300)")
    parser.add_argument("--obj-gate", type=float, default=None,
                        help="Early objectness gate threshold in [0, 1] (default: None, auto-matches conf)")
    parser.add_argument("--decoder-version", type=str, default="v2_smooth", choices=["v2_smooth", "v1_legacy"],
                        help="Box decoder parameterization version (default: v2_smooth)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--output-json", type=str, default="experiments/custom_model/ird_evaluation.json",
                        help="Path to save evaluation JSON")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of validation images to evaluate (optional)")
    parser.add_argument("--use-atd", action="store_true", default=None,
                        help="Explicitly enable ATD neck module (defaults to auto-detection from checkpoint)")
    parser.add_argument("--run-unit-tests", action="store_true", default=False,
                        help="Run self-contained unit tests before evaluation")

    args = parser.parse_args()

    if args.run_unit_tests:
        run_unit_tests()

    # If unit tests flag was given standalone or weights don't exist, exit cleanly
    if args.run_unit_tests and (len(sys.argv) == 2 or not Path(args.weights).exists()):
        sys.exit(0)

    run_evaluation(
        weights_path=args.weights,
        data_yaml_path=args.data,
        img_size=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
        device_str=args.device,
        output_json_path=args.output_json,
        max_samples=args.max_samples,
        obj_gate=args.obj_gate,
        decoder_version=args.decoder_version,
        use_atd=args.use_atd,
    )
