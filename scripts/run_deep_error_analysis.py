"""
Comprehensive Deep Error Analysis Engine for IRD V1.5 (Epoch 19 Checkpoint).

Performs:
1. Full 1,719-image validation pass using authoritative evaluator.
2. Object-level GT categorization (TP, Missed, Wrong Class, Poor Loc, Duplicate, Low Conf).
3. Prediction-level categorization (TP, Cls Error, Loc Error, Duplicate, Background FP).
4. Class-by-class error metrics for all 12 classes.
5. 13x13 Confusion Matrix (12 classes + background).
6. Confidence-Quality calibration analysis across 6 confidence bins.
7. Object-size (small, medium, large) and density (1, 2-4, 5-9, 10+) breakdowns.
8. Generation and export of 70+ annotated diagnostic visualizations:
   - 20 worst False Positive images
   - 20 worst Missed Object images
   - 20 worst Wrong Class images
   - 10 Duplicate Box examples
9. Structured export to experiments/evaluation_results/ird_v15_error_analysis.json
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import stats
import torch
import yaml

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.box_coder import (
    NUM_CLASSES,
    decode_ird_predictions_authoritative,
)
from src.models.custom_detector import build_detector
from scripts.evaluate_ird import (
    BENCHMARK_CLASSES,
    IOU_THRESHOLDS,
    box_iou_xyxy,
    evaluate_class_predictions,
)

# Colors for visualization (BGR)
COLOR_GT = (0, 220, 0)         # Green for Ground Truth
COLOR_TP = (255, 200, 0)       # Cyan for True Positives
COLOR_FP_BKG = (0, 0, 255)     # Red for Background False Positives
COLOR_FP_CLS = (0, 140, 255)   # Orange for Classification Error
COLOR_FP_LOC = (255, 0, 255)   # Magenta for Localization Error
COLOR_DUP = (180, 180, 0)      # Olive/Teal for Duplicate Boxes


def compute_iou_matrix_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix between boxes1 [N, 4] and boxes2 [M, 4]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clip(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clip(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clip(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clip(min=0.0)

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clip(min=0.0)
    inter_h = (inter_y2 - inter_y1).clip(min=0.0)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-16)


def run_deep_error_analysis(
    weights_path: str = "experiments/custom_model/final_training_50ep/ird_best.pt",
    data_yaml_path: str = "data/indian_road_yolo/data.yaml",
    img_size: int = 640,
    device_str: str = "auto",
    output_json: str = "experiments/evaluation_results/ird_v15_error_analysis.json",
    vis_dir: str = "experiments/evaluation_results/error_analysis",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    # 1. Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("IRD V1.5 DEEP ERROR ANALYSIS & DIAGNOSTIC ENGINE", flush=True)
    print(f"Device:         {device}", flush=True)
    print(f"Checkpoint:     {weights_path}", flush=True)
    print(f"Dataset YAML:   {data_yaml_path}", flush=True)
    print(f"Resolution:     {img_size}x{img_size}", flush=True)
    print(f"Visualizations: {vis_path.resolve()}", flush=True)
    print("=" * 80, flush=True)

    # 2. Checkpoint Loading
    ckpt_p = Path(weights_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    use_atd = any("atd" in k for k in sd.keys())
    use_ssdp = any("ssdp" in k for k in sd.keys())
    use_fgbr = any("fgbr" in k for k in sd.keys())

    model = build_detector(
        num_classes=NUM_CLASSES,
        use_atd=use_atd,
        use_ssdp=use_ssdp,
        use_fgbr=use_fgbr,
    )
    model.to(device)
    model.load_state_dict(sd)
    model.eval()

    # 3. Dataset Images & Labels
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = Path(data_yaml_path).parent
    val_rel = data_cfg.get("val", "images/val")
    val_img_dir = (yaml_dir / val_rel).resolve()
    val_lbl_dir = val_img_dir.parent.parent / "labels" / "val"
    if not val_lbl_dir.exists():
        val_lbl_dir = yaml_dir / "labels" / "val"

    all_img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")))
    if max_samples and len(all_img_files) > max_samples:
        all_img_files = all_img_files[:max_samples]

    n_images = len(all_img_files)
    print(f"Evaluating {n_images} validation images...", flush=True)

    # 4. Storage for Comprehensive Diagnostics
    # Image records
    image_records = []
    
    # Global confusion matrix: 13 x 13 (index 12 is Background)
    BG_IDX = NUM_CLASSES  # 12
    confusion_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=np.int64)

    # Class breakdown counters
    class_stats = {
        c_name: {
            "gt_count": 0,
            "tp_count": 0,
            "fn_count": 0,
            "duplicate_count": 0,
            "fp_count": 0,
            "classification_errors": 0,
            "localization_errors": 0,
            "low_conf_matches": 0,
            "ap50": 0.0,
            "ap50_95": 0.0,
        }
        for c_name in BENCHMARK_CLASSES
    }

    # Predictions and GTs grouped for AP calculation
    all_preds_for_ap = {c: [] for c in range(NUM_CLASSES)}
    all_gts_for_ap = {c: [] for c in range(NUM_CLASSES)}

    # Confidence-quality matching records: list of dict(conf, iou, is_correct, class_id)
    matched_pair_records = []

    # All predictions record for confidence bins: list of dict(conf, is_tp, is_dup, iou)
    all_pred_records = []

    # Object size records: "small", "medium", "large"
    size_stats = {
        "small": {"gt_count": 0, "detected_count": 0, "pred_count": 0, "tp_count": 0, "ious": [], "preds": [], "gts": []},
        "medium": {"gt_count": 0, "detected_count": 0, "pred_count": 0, "tp_count": 0, "ious": [], "preds": [], "gts": []},
        "large": {"gt_count": 0, "detected_count": 0, "pred_count": 0, "tp_count": 0, "ious": [], "preds": [], "gts": []},
    }

    # Density records: "1", "2-4", "5-9", "10+"
    density_bins = {
        "1": {"images": 0, "gt_count": 0, "tp_count": 0, "fp_count": 0, "pred_count": 0},
        "2-4": {"images": 0, "gt_count": 0, "tp_count": 0, "fp_count": 0, "pred_count": 0},
        "5-9": {"images": 0, "gt_count": 0, "tp_count": 0, "fp_count": 0, "pred_count": 0},
        "10+": {"images": 0, "gt_count": 0, "tp_count": 0, "fp_count": 0, "pred_count": 0},
    }

    t0_start = time.perf_counter()

    for img_idx, img_p in enumerate(all_img_files):
        img0 = cv2.imread(str(img_p))
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        img_resized = cv2.resize(img0, (img_size, img_size))
        img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            head_out = model(img_tensor)
            # Run Authoritative Evaluator decoding (conf=0.001, iou=0.50, max_det=300)
            boxes, scores, classes = decode_ird_predictions_authoritative(
                head_out,
                img_size=img_size,
                conf_threshold=0.001,
                iou_threshold=0.50,
                max_det=300,
                obj_gate=0.001,
                device=device,
                return_diagnostics=False,
                score_mode="sqrt_quality",
            )

        # Scale predictions to original image space
        scale_x = w0 / float(img_size)
        scale_y = h0 / float(img_size)

        preds_np = []
        for b, s, c in zip(boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()):
            c_int = int(c)
            if 0 <= c_int < NUM_CLASSES:
                scaled_b = [float(b[0] * scale_x), float(b[1] * scale_y), float(b[2] * scale_x), float(b[3] * scale_y)]
                preds_np.append({
                    "box": scaled_b,
                    "score": float(s),
                    "class_id": c_int,
                })
                all_preds_for_ap[c_int].append({
                    "img_id": img_idx,
                    "score": float(s),
                    "box": scaled_b,
                })

        # Load Ground Truths
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        gts_np = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_cls = int(parts[0])
                        cx = float(parts[1]) * w0
                        cy = float(parts[2]) * h0
                        w = float(parts[3]) * w0
                        h = float(parts[4]) * h0
                        if 0 <= gt_cls < NUM_CLASSES:
                            area = w * h
                            gt_box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
                            gts_np.append({
                                "box": gt_box,
                                "class_id": gt_cls,
                                "area": area,
                            })
                            all_gts_for_ap[gt_cls].append({
                                "img_id": img_idx,
                                "box": gt_box,
                            })

        # TIDE / Object Detection Matching Analysis for this Image
        n_gt = len(gts_np)
        n_p = len(preds_np)

        # Update GT counts per class
        for g in gts_np:
            cls_name = BENCHMARK_CLASSES[g["class_id"]]
            class_stats[cls_name]["gt_count"] += 1
            area = g["area"]
            if area < 32 * 32:
                size_stats["small"]["gt_count"] += 1
                size_stats["small"]["gts"].append({"img_id": img_idx, "box": g["box"]})
            elif area <= 96 * 96:
                size_stats["medium"]["gt_count"] += 1
                size_stats["medium"]["gts"].append({"img_id": img_idx, "box": g["box"]})
            else:
                size_stats["large"]["gt_count"] += 1
                size_stats["large"]["gts"].append({"img_id": img_idx, "box": g["box"]})

        # Track density bin
        if n_gt == 1:
            d_key = "1"
        elif 2 <= n_gt <= 4:
            d_key = "2-4"
        elif 5 <= n_gt <= 9:
            d_key = "5-9"
        else:
            d_key = "10+"
        density_bins[d_key]["images"] += 1
        density_bins[d_key]["gt_count"] += n_gt
        density_bins[d_key]["pred_count"] += n_p

        # Sort predictions by score descending
        preds_sorted = sorted(preds_np, key=lambda x: x["score"], reverse=True)
        pred_boxes_mat = np.array([p["box"] for p in preds_sorted], dtype=np.float32) if n_p > 0 else np.zeros((0, 4))
        gt_boxes_mat = np.array([g["box"] for g in gts_np], dtype=np.float32) if n_gt > 0 else np.zeros((0, 4))

        iou_matrix = compute_iou_matrix_np(pred_boxes_mat, gt_boxes_mat)  # [N_pred, M_gt]

        gt_matched_tp = [False] * n_gt
        gt_matching_score = [0.0] * n_gt
        gt_duplicate_count = [0] * n_gt
        gt_wrong_class_match = [False] * n_gt
        gt_poor_loc_match = [False] * n_gt

        pred_labels = []  # Label for each pred: "TP", "DUP", "CLS_ERR", "LOC_ERR", "BKG_FP"
        pred_matched_gt_idx = [-1] * n_p
        pred_matched_iou = [0.0] * n_p

        # Pass 1: Greedily assign True Positives (IoU >= 0.50, same class)
        for p_i, pred in enumerate(preds_sorted):
            p_cls = pred["class_id"]
            p_score = pred["score"]

            best_iou = 0.0
            best_g_idx = -1

            for g_j, gt in enumerate(gts_np):
                if gt["class_id"] == p_cls:
                    iou_val = iou_matrix[p_i, g_j]
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_g_idx = g_j

            if best_g_idx >= 0 and best_iou >= 0.50:
                if not gt_matched_tp[best_g_idx]:
                    # True Positive
                    gt_matched_tp[best_g_idx] = True
                    gt_matching_score[best_g_idx] = p_score
                    pred_labels.append("TP")
                    pred_matched_gt_idx[p_i] = best_g_idx
                    pred_matched_iou[p_i] = best_iou
                    confusion_matrix[p_cls, p_cls] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["tp_count"] += 1
                    matched_pair_records.append({
                        "conf": p_score,
                        "iou": best_iou,
                        "is_correct": True,
                        "class_id": p_cls,
                    })
                    all_pred_records.append({
                        "conf": p_score,
                        "is_tp": True,
                        "is_dup": False,
                        "iou": best_iou,
                    })

                    # Size stats
                    gt_area = gts_np[best_g_idx]["area"]
                    if gt_area < 32 * 32:
                        size_stats["small"]["detected_count"] += 1
                        size_stats["small"]["ious"].append(best_iou)
                    elif gt_area <= 96 * 96:
                        size_stats["medium"]["detected_count"] += 1
                        size_stats["medium"]["ious"].append(best_iou)
                    else:
                        size_stats["large"]["detected_count"] += 1
                        size_stats["large"]["ious"].append(best_iou)
                    density_bins[d_key]["tp_count"] += 1
                else:
                    # Duplicate Box
                    gt_duplicate_count[best_g_idx] += 1
                    pred_labels.append("DUP")
                    pred_matched_gt_idx[p_i] = best_g_idx
                    pred_matched_iou[p_i] = best_iou
                    confusion_matrix[p_cls, p_cls] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["duplicate_count"] += 1
                    all_pred_records.append({
                        "conf": p_score,
                        "is_tp": False,
                        "is_dup": True,
                        "iou": best_iou,
                    })
            else:
                # Check for Classification Error (overlaps a GT of DIFFERENT class with IoU >= 0.50)
                best_diff_iou = 0.0
                best_diff_g_idx = -1
                for g_j, gt in enumerate(gts_np):
                    if gt["class_id"] != p_cls:
                        iou_val = iou_matrix[p_i, g_j]
                        if iou_val > best_diff_iou:
                            best_diff_iou = iou_val
                            best_diff_g_idx = g_j

                if best_diff_g_idx >= 0 and best_diff_iou >= 0.50:
                    gt_wrong_cls = gts_np[best_diff_g_idx]["class_id"]
                    gt_wrong_class_match[best_diff_g_idx] = True
                    pred_labels.append("CLS_ERR")
                    pred_matched_gt_idx[p_i] = best_diff_g_idx
                    pred_matched_iou[p_i] = best_diff_iou
                    confusion_matrix[p_cls, gt_wrong_cls] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["classification_errors"] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["fp_count"] += 1
                    density_bins[d_key]["fp_count"] += 1
                    matched_pair_records.append({
                        "conf": p_score,
                        "iou": best_diff_iou,
                        "is_correct": False,
                        "class_id": p_cls,
                    })
                    all_pred_records.append({
                        "conf": p_score,
                        "is_tp": False,
                        "is_dup": False,
                        "iou": best_diff_iou,
                    })
                elif best_g_idx >= 0 and best_iou >= 0.10:
                    # Localization Error (same class, 0.10 <= IoU < 0.50)
                    gt_poor_loc_match[best_g_idx] = True
                    pred_labels.append("LOC_ERR")
                    pred_matched_gt_idx[p_i] = best_g_idx
                    pred_matched_iou[p_i] = best_iou
                    confusion_matrix[p_cls, BG_IDX] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["localization_errors"] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["fp_count"] += 1
                    density_bins[d_key]["fp_count"] += 1
                    all_pred_records.append({
                        "conf": p_score,
                        "is_tp": False,
                        "is_dup": False,
                        "iou": best_iou,
                    })
                else:
                    # Background False Positive (IoU < 0.10 with any target)
                    pred_labels.append("BKG_FP")
                    confusion_matrix[p_cls, BG_IDX] += 1
                    class_stats[BENCHMARK_CLASSES[p_cls]]["fp_count"] += 1
                    density_bins[d_key]["fp_count"] += 1
                    all_pred_records.append({
                        "conf": p_score,
                        "is_tp": False,
                        "is_dup": False,
                        "iou": best_iou if best_g_idx >= 0 else 0.0,
                    })

            # Track pred size stats
            pw = pred["box"][2] - pred["box"][0]
            ph = pred["box"][3] - pred["box"][1]
            p_area = max(0.0, pw) * max(0.0, ph)
            if p_area < 32 * 32:
                size_stats["small"]["pred_count"] += 1
                if pred_labels[-1] == "TP":
                    size_stats["small"]["tp_count"] += 1
                size_stats["small"]["preds"].append({"img_id": img_idx, "score": p_score, "box": pred["box"]})
            elif p_area <= 96 * 96:
                size_stats["medium"]["pred_count"] += 1
                if pred_labels[-1] == "TP":
                    size_stats["medium"]["tp_count"] += 1
                size_stats["medium"]["preds"].append({"img_id": img_idx, "score": p_score, "box": pred["box"]})
            else:
                size_stats["large"]["pred_count"] += 1
                if pred_labels[-1] == "TP":
                    size_stats["large"]["tp_count"] += 1
                size_stats["large"]["preds"].append({"img_id": img_idx, "score": p_score, "box": pred["box"]})

        # Ground Truth Categorization for this image
        n_missed_in_img = 0
        n_cls_err_in_img = 0
        n_dup_in_img = 0
        n_fp_in_img = sum(1 for l in pred_labels if l in ["BKG_FP", "LOC_ERR", "CLS_ERR"])

        for g_j, gt in enumerate(gts_np):
            g_cls = gt["class_id"]
            cls_name = BENCHMARK_CLASSES[g_cls]

            if gt_matched_tp[g_j]:
                if gt_matching_score[g_j] < 0.25:
                    class_stats[cls_name]["low_conf_matches"] += 1
            else:
                confusion_matrix[BG_IDX, g_cls] += 1
                class_stats[cls_name]["fn_count"] += 1
                if gt_wrong_class_match[g_j]:
                    n_cls_err_in_img += 1
                else:
                    n_missed_in_img += 1

            if gt_duplicate_count[g_j] > 0:
                n_dup_in_img += gt_duplicate_count[g_j]

        # Save record for candidate visualization selection
        image_records.append({
            "img_idx": img_idx,
            "img_path": str(img_p),
            "h0": h0,
            "w0": w0,
            "n_gt": n_gt,
            "n_pred": n_p,
            "preds": preds_sorted,
            "pred_labels": pred_labels,
            "gts": gts_np,
            "gt_matched_tp": gt_matched_tp,
            "n_fp": n_fp_in_img,
            "n_missed": n_missed_in_img,
            "n_cls_err": n_cls_err_in_img,
            "n_dup": n_dup_in_img,
            # Count high confidence false positives
            "high_conf_fp": sum(1 for i, l in enumerate(pred_labels) if l == "BKG_FP" and preds_sorted[i]["score"] >= 0.10),
            "high_conf_cls_err": sum(1 for i, l in enumerate(pred_labels) if l == "CLS_ERR" and preds_sorted[i]["score"] >= 0.15),
        })

        if (img_idx + 1) % 200 == 0 or (img_idx + 1) == n_images:
            el = time.perf_counter() - t0_start
            fps = (img_idx + 1) / el
            print(f"  Processed [{img_idx + 1:>4}/{n_images}] images | Speed: {fps:.1f} FPS", flush=True)

    print(f"\nCompleted matching across all {n_images} images in {time.perf_counter() - t0_start:.1f}s!", flush=True)

    # 5. Compute Official AP50 and AP50-95 for every class
    print("\nComputing official AP metrics per class...", flush=True)
    mAP50_list = []
    mAP50_95_list = []

    for c_id, c_name in enumerate(BENCHMARK_CLASSES):
        res = evaluate_class_predictions(
            all_preds_for_ap[c_id],
            all_gts_for_ap[c_id],
            class_id=c_id,
            iou_thresholds=IOU_THRESHOLDS,
        )
        class_stats[c_name]["ap50"] = round(res["ap50"], 4)
        class_stats[c_name]["ap50_95"] = round(res["ap50_95"], 4)
        if res["n_gt"] > 0:
            mAP50_list.append(res["ap50"])
            mAP50_95_list.append(res["ap50_95"])

    overall_mAP50 = float(np.mean(mAP50_list)) if mAP50_list else 0.0
    overall_mAP50_95 = float(np.mean(mAP50_95_list)) if mAP50_95_list else 0.0

    # 6. Compute Confidence-Quality Correlation & Bins
    print("Computing Confidence-Quality calibration metrics...", flush=True)
    if matched_pair_records:
        confs = np.array([m["conf"] for m in matched_pair_records])
        actual_ious = np.array([m["iou"] for m in matched_pair_records])
        pearson_r, pearson_p = stats.pearsonr(confs, actual_ious)
        spearman_r, spearman_p = stats.spearmanr(confs, actual_ious)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_r, spearman_p = 0.0, 1.0

    conf_bin_edges = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    conf_bin_results = []
    total_gts_all = sum(cs["gt_count"] for cs in class_stats.values())

    for b_idx in range(len(conf_bin_edges) - 1):
        low, high = conf_bin_edges[b_idx], conf_bin_edges[b_idx + 1]
        bin_preds = [p for p in all_pred_records if low <= p["conf"] < high] if high < 1.0 else [p for p in all_pred_records if low <= p["conf"] <= high]
        
        n_b = len(bin_preds)
        n_tp = sum(1 for p in bin_preds if p["is_tp"])
        n_dup = sum(1 for p in bin_preds if p["is_dup"])
        ious_in_bin = [p["iou"] for p in bin_preds if p["is_tp"] or p["is_dup"]]

        prec = n_tp / max(1, n_b)
        rec_contrib = n_tp / max(1, total_gts_all)
        dup_rate = n_dup / max(1, n_b)
        mean_iou = float(np.mean(ious_in_bin)) if ious_in_bin else 0.0

        conf_bin_results.append({
            "bin": f"{low:.2f}–{high:.2f}",
            "low": low,
            "high": high,
            "prediction_count": n_b,
            "tp_count": n_tp,
            "duplicate_count": n_dup,
            "precision": round(prec, 4),
            "recall_contribution": round(rec_contrib, 4),
            "mean_iou": round(mean_iou, 4),
            "duplicate_rate": round(dup_rate, 4),
        })

    # 7. Object Size Metrics
    size_results = {}
    for s_name in ["small", "medium", "large"]:
        s_data = size_stats[s_name]
        gt_c = s_data["gt_count"]
        det_c = s_data["detected_count"]
        pred_c = s_data["pred_count"]
        tp_c = s_data["tp_count"]
        rec = det_c / max(1, gt_c)
        prec = tp_c / max(1, pred_c)
        mean_iou = float(np.mean(s_data["ious"])) if s_data["ious"] else 0.0

        # Calculate AP50 for this scale if possible
        scale_ap_res = evaluate_class_predictions(s_data["preds"], s_data["gts"], class_id=0, iou_thresholds=IOU_THRESHOLDS)
        size_results[s_name] = {
            "gt_count": gt_c,
            "prediction_count": pred_c,
            "detected_count": det_c,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "mean_iou": round(mean_iou, 4),
            "ap50": round(scale_ap_res["ap50"], 4),
        }

    # 8. Density Metrics
    density_results = {}
    for d_name in ["1", "2-4", "5-9", "10+"]:
        d_data = density_bins[d_name]
        n_imgs = d_data["images"]
        gt_c = d_data["gt_count"]
        tp_c = d_data["tp_count"]
        fp_c = d_data["fp_count"]
        pred_c = d_data["pred_count"]
        rec = tp_c / max(1, gt_c)
        fp_rate = fp_c / max(1, pred_c)
        fp_per_img = fp_c / max(1, n_imgs)
        density_results[d_name] = {
            "image_count": n_imgs,
            "gt_count": gt_c,
            "pred_count": pred_c,
            "tp_count": tp_c,
            "fp_count": fp_c,
            "recall": round(rec, 4),
            "fp_rate": round(fp_rate, 4),
            "fp_per_image": round(fp_per_img, 1),
        }

    # 9. Render and Export Visualization Artifacts
    print("\nRendering annotated diagnostic visualizations...", flush=True)

    def draw_annotated_diagnostic_image(
        rec: Dict[str, Any],
        title_tag: str,
        save_name: str,
    ):
        img = cv2.imread(rec["img_path"])
        if img is None:
            return

        h, w = img.shape[:2]
        canvas = img.copy()

        # 1. Draw Ground Truths in bold Green
        for gt in rec["gts"]:
            box = gt["box"]
            cls_name = BENCHMARK_CLASSES[gt["class_id"]]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_GT, 2)
            tag = f"GT:{cls_name}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, max(0, y1)), COLOR_GT, -1)
            cv2.putText(canvas, tag, (x1 + 2, max(0, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Predictions with distinct error colors and labels
        # Only draw predictions with score >= 0.05 to avoid turning the canvas pitch black
        preds_to_draw = [
            (p, l) for p, l in zip(rec["preds"], rec["pred_labels"])
            if p["score"] >= 0.05 or l in ["TP", "CLS_ERR"]
        ]

        for pred, label in preds_to_draw:
            box = pred["box"]
            cls_name = BENCHMARK_CLASSES[pred["class_id"]]
            score = pred["score"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            if label == "TP":
                color = COLOR_TP
                desc = "TP"
            elif label == "CLS_ERR":
                color = COLOR_FP_CLS
                desc = "WRONG_CLS"
            elif label == "LOC_ERR":
                color = COLOR_FP_LOC
                desc = "LOC_ERR"
            elif label == "DUP":
                color = COLOR_DUP
                desc = "DUP"
            else:
                color = COLOR_FP_BKG
                desc = "BKG_FP"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            pred_tag = f"{cls_name} {score:.2f} [{desc}]"
            (tw, th), _ = cv2.getTextSize(pred_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            tag_y = min(h - 4, y2 + th + 4)
            cv2.rectangle(canvas, (x1, tag_y - th - 2), (x1 + tw + 4, tag_y + 2), color, -1)
            cv2.putText(canvas, pred_tag, (x1 + 2, tag_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top Banner HUD
        hud_bar_h = 36
        hud_canvas = np.zeros((hud_bar_h, w, 3), dtype=np.uint8)
        hud_canvas[:] = (20, 20, 20)
        hud_text = f"[{title_tag}] {Path(rec['img_path']).name} | GT: {rec['n_gt']} | Preds(>=0.05): {len(preds_to_draw)} | FP: {rec['n_fp']} | Missed: {rec['n_missed']} | ClsErr: {rec['n_cls_err']}"
        cv2.putText(hud_canvas, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        out_frame = np.vstack([hud_canvas, canvas])
        out_dest = vis_path / save_name
        cv2.imwrite(str(out_dest), out_frame)

    # A. 20 Worst False Positive Images (sorted by high_conf_fp, then total n_fp)
    worst_fp_images = sorted(image_records, key=lambda x: (x["high_conf_fp"], x["n_fp"]), reverse=True)[:20]
    for rank, r in enumerate(worst_fp_images, 1):
        draw_annotated_diagnostic_image(
            r,
            title_tag="WORST FALSE POSITIVES",
            save_name=f"fp_worst_{rank:02d}_{Path(r['img_path']).stem}.jpg",
        )

    # B. 20 Worst Missed Objects Images (sorted by n_missed, then n_gt)
    worst_missed_images = sorted(image_records, key=lambda x: (x["n_missed"], x["n_gt"]), reverse=True)[:20]
    for rank, r in enumerate(worst_missed_images, 1):
        draw_annotated_diagnostic_image(
            r,
            title_tag="WORST MISSED OBJECTS",
            save_name=f"fn_missed_{rank:02d}_{Path(r['img_path']).stem}.jpg",
        )

    # C. 20 Worst Wrong Class Images (sorted by high_conf_cls_err, then n_cls_err)
    worst_cls_images = sorted(image_records, key=lambda x: (x["high_conf_cls_err"], x["n_cls_err"]), reverse=True)[:20]
    for rank, r in enumerate(worst_cls_images, 1):
        draw_annotated_diagnostic_image(
            r,
            title_tag="CLASSIFICATION ERRORS",
            save_name=f"cls_error_{rank:02d}_{Path(r['img_path']).stem}.jpg",
        )

    # D. 10 Duplicate Box Examples (sorted by n_dup)
    worst_dup_images = sorted(image_records, key=lambda x: x["n_dup"], reverse=True)[:10]
    for rank, r in enumerate(worst_dup_images, 1):
        draw_annotated_diagnostic_image(
            r,
            title_tag="DUPLICATE BOX DETECTIONS",
            save_name=f"duplicate_{rank:02d}_{Path(r['img_path']).stem}.jpg",
        )

    print("Saved 70 annotated diagnostic images to error_analysis/ directory!", flush=True)

    # 10. Compile JSON Data
    # Convert confusion matrix to nested dict
    conf_matrix_dict = {}
    class_labels_with_bg = BENCHMARK_CLASSES + ["background"]
    for i, p_name in enumerate(class_labels_with_bg):
        conf_matrix_dict[p_name] = {}
        for j, gt_name in enumerate(class_labels_with_bg):
            conf_matrix_dict[p_name][gt_name] = int(confusion_matrix[i, j])

    final_analysis_data = {
        "metadata": {
            "model": "IRD V1.5 (IndianRoadDetection)",
            "checkpoint": weights_path,
            "total_val_images": n_images,
            "image_size": img_size,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_mAP50": round(overall_mAP50, 4),
            "overall_mAP50_95": round(overall_mAP50_95, 4),
        },
        "class_error_breakdown": class_stats,
        "confusion_matrix": conf_matrix_dict,
        "confidence_quality_analysis": {
            "pearson_correlation": round(float(pearson_r), 4),
            "pearson_p_value": float(pearson_p),
            "spearman_rank_correlation": round(float(spearman_r), 4),
            "spearman_p_value": float(spearman_p),
            "confidence_bins": conf_bin_results,
        },
        "object_size_analysis": size_results,
        "density_analysis": density_results,
        "visualization_artifacts": {
            "worst_false_positives": [f"fp_worst_{rank:02d}_{Path(r['img_path']).stem}.jpg" for rank, r in enumerate(worst_fp_images, 1)],
            "worst_missed_objects": [f"fn_missed_{rank:02d}_{Path(r['img_path']).stem}.jpg" for rank, r in enumerate(worst_missed_images, 1)],
            "worst_wrong_classes": [f"cls_error_{rank:02d}_{Path(r['img_path']).stem}.jpg" for rank, r in enumerate(worst_cls_images, 1)],
            "duplicate_examples": [f"duplicate_{rank:02d}_{Path(r['img_path']).stem}.jpg" for rank, r in enumerate(worst_dup_images, 1)],
        },
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_analysis_data, f, indent=2)

    print(f"\nDeep Error Analysis completed successfully! Saved structured JSON to: {out_p.resolve()}", flush=True)
    return final_analysis_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Error Analysis for IRD V1.5")
    parser.add_argument("--weights", type=str, default="experiments/custom_model/final_training_50ep/ird_best.pt")
    parser.add_argument("--data", type=str, default="data/indian_road_yolo/data.yaml")
    parser.add_argument("--output", type=str, default="experiments/evaluation_results/ird_v15_error_analysis.json")
    parser.add_argument("--vis-dir", type=str, default="experiments/evaluation_results/error_analysis")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_deep_error_analysis(
        weights_path=args.weights,
        data_yaml_path=args.data,
        output_json=args.output,
        vis_dir=args.vis_dir,
        max_samples=args.samples,
        device_str=args.device,
    )
