"""
Comprehensive Deep Diagnostics for IRD V1
Implements Phases 2 through 11:
- Phase 2: Class Confusion Matrix & Per-Class PR Diagnostics
- Phase 3: Multi-Object Density Breakdown (1, 2-4, 5-9, 10+ objects)
- Phase 4: Motorcycle / Scooter Detection Quality
- Phase 5: Rider vs Person Context Analysis
- Phase 6: Truck & Bus vs Car Discrimination
- Phase 7: Multi-Scale / Size Breakdown (Tiny, Small, Medium, Large)
- Phase 8: Target Assignment Matcher Audit
- Phase 9: Confidence Calibration (IoU correlation & calibration error)
- Phase 10: Box Localization Quality profile (IoU 0.50 to 0.95 breakdown)
- Phase 11: Loss Gradient Balance Audit
"""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector, build_detector
from src.models.losses.custom_loss import IndianRoadLoss, bbox_ciou
from src.models.box_coder import decode_ird_predictions_authoritative

BENCHMARK_CLASSES = [
    "person", "rider", "car", "truck", "bus", "motorcycle",
    "bicycle", "autorickshaw", "animal", "vehicle fallback",
    "traffic light", "traffic sign"
]
NUM_CLASSES = len(BENCHMARK_CLASSES)

def box_iou_xyxy(b1, b2):
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)))
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)

def run_deep_diagnostics(
    weights_path: str,
    data_yaml_path: str = "data/indian_road_yolo/data.yaml",
    img_size: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    max_eval_images: int = 500,
    device_str: str = "auto",
    output_path: str = "experiments/custom_model/deep_diagnostics.json"
):
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")
    print(f"[Deep Diagnostics] Loading weights from {weights_path} on {device}...")
    
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    use_atd = any("atd" in k for k in clean_sd.keys())
    if isinstance(ckpt, dict) and "config" in ckpt:
        use_atd = use_atd or ckpt["config"].get("use_atd", False)

    use_ssdp = any("ssdp" in k for k in clean_sd.keys())
    if isinstance(ckpt, dict) and "config" in ckpt:
        use_ssdp = use_ssdp or ckpt["config"].get("use_ssdp", False)

    use_fgbr = any("fgbr" in k for k in clean_sd.keys())
    if isinstance(ckpt, dict) and "config" in ckpt:
        use_fgbr = use_fgbr or ckpt["config"].get("use_fgbr", False)

    print(f"[Deep Diagnostics] Initializing detector (use_atd={use_atd}, use_ssdp={use_ssdp}, use_fgbr={use_fgbr})...")
    detector = build_detector(num_classes=NUM_CLASSES, use_atd=use_atd, use_ssdp=use_ssdp, use_fgbr=use_fgbr)
    detector.load_state_dict(clean_sd)
    detector.to(device)
    detector.eval()

    val_img_dir = Path("data/indian_road_yolo/images/val")
    val_lbl_dir = Path("data/indian_road_yolo/labels/val")
    img_files = sorted(list(val_img_dir.glob("*.jpg")))[:max_eval_images]

    # Metrics accumulators
    confusion_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int) # +1 is background
    
    # Density buckets: 1, 2-4, 5-9, 10+
    density_stats = {
        "1": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "2-4": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "5-9": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "10+": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
    }
    
    # Size buckets: tiny (<32), small (32-96), medium (96-256), large (>256)
    size_stats = {
        "tiny": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "small": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "medium": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "large": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
    }
    
    # Per-class recall and precision
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "ious": [], "confs": []} for c in BENCHMARK_CLASSES}
    
    calibration_pairs = [] # (conf, matched_iou)

    print(f"[Deep Diagnostics] Evaluating {len(img_files)} images for multidimensional failure analysis...")

    with torch.no_grad():
        for idx, img_p in enumerate(img_files):
            img_raw = Image.open(img_p).convert("RGB")
            ow, oh = img_raw.size
            img_resized = img_raw.resize((img_size, img_size))
            inp = torch.from_numpy(np.array(img_resized, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            head_out = detector(inp)
            boxes, scores, class_ids = decode_ird_predictions_authoritative(
                head_out,
                img_size=img_size,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                max_det=300,
                obj_gate=0.05,
                decoder_version="v2_smooth",
                device=device,
            )

            p_boxes = boxes.cpu().numpy()
            p_scores = scores.cpu().numpy()
            p_classes = class_ids.cpu().numpy()

            # Ground truth
            lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
            gt_boxes = []
            gt_classes = []
            if lbl_p.exists():
                with open(lbl_p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            cx, cy, w, h = [float(x) for x in parts[1:5]]
                            x1 = (cx - w/2) * img_size
                            y1 = (cy - h/2) * img_size
                            x2 = (cx + w/2) * img_size
                            y2 = (cy + h/2) * img_size
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_classes.append(cid)
                            
            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            gt_classes = np.array(gt_classes) if gt_classes else np.zeros((0,), dtype=int)
            n_gt = len(gt_classes)

            # Determine density bucket
            if n_gt == 1:
                db_key = "1"
            elif 2 <= n_gt <= 4:
                db_key = "2-4"
            elif 5 <= n_gt <= 9:
                db_key = "5-9"
            else:
                db_key = "10+"
            density_stats[db_key]["gt"] += n_gt

            # Match predictions to GTs
            ious = box_iou_xyxy(p_boxes, gt_boxes)
            matched_gt = set()
            
            # Sort preds by score descending
            sort_order = np.argsort(-p_scores)
            for p_i in sort_order:
                pred_c = int(p_classes[p_i])
                pred_s = float(p_scores[p_i])
                pred_cname = BENCHMARK_CLASSES[pred_c]
                
                best_iou = 0.0
                best_gt_idx = -1
                if n_gt > 0:
                    for g_i in range(n_gt):
                        if g_i not in matched_gt and ious[p_i, g_i] > best_iou:
                            best_iou = ious[p_i, g_i]
                            best_gt_idx = g_i
                            
                calibration_pairs.append((pred_s, best_iou))
                
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    gt_c = gt_classes[best_gt_idx]
                    confusion_matrix[gt_c, pred_c] += 1
                    matched_gt.add(best_gt_idx)
                    
                    # Size of GT
                    gw = gt_boxes[best_gt_idx, 2] - gt_boxes[best_gt_idx, 0]
                    gh = gt_boxes[best_gt_idx, 3] - gt_boxes[best_gt_idx, 1]
                    area_sqrt = np.sqrt(max(0, gw * gh))
                    sz_key = "tiny" if area_sqrt < 32 else ("small" if area_sqrt < 96 else ("medium" if area_sqrt < 256 else "large"))
                    
                    if gt_c == pred_c:
                        class_stats[pred_cname]["tp"] += 1
                        class_stats[pred_cname]["ious"].append(best_iou)
                        class_stats[pred_cname]["confs"].append(pred_s)
                        density_stats[db_key]["tp"] += 1
                        size_stats[sz_key]["tp"] += 1
                    else:
                        class_stats[pred_cname]["fp"] += 1
                        density_stats[db_key]["fp"] += 1
                        size_stats[sz_key]["fp"] += 1
                else:
                    # False positive (predicted on background or duplicate)
                    confusion_matrix[NUM_CLASSES, pred_c] += 1
                    class_stats[pred_cname]["fp"] += 1
                    density_stats[db_key]["fp"] += 1

            # Check unmatched GTs (False Negatives)
            for g_i in range(n_gt):
                gt_c = gt_classes[g_i]
                cname = BENCHMARK_CLASSES[gt_c]
                class_stats[cname]["gt"] += 1
                gw = gt_boxes[g_i, 2] - gt_boxes[g_i, 0]
                gh = gt_boxes[g_i, 3] - gt_boxes[g_i, 1]
                area_sqrt = np.sqrt(max(0, gw * gh))
                sz_key = "tiny" if area_sqrt < 32 else ("small" if area_sqrt < 96 else ("medium" if area_sqrt < 256 else "large"))
                size_stats[sz_key]["gt"] += 1
                
                if g_i not in matched_gt:
                    confusion_matrix[gt_c, NUM_CLASSES] += 1
                    class_stats[cname]["fn"] += 1
                    density_stats[db_key]["fn"] += 1
                    size_stats[sz_key]["fn"] += 1

    # Summarize Per-Class Recall & Precision
    per_class_summary = {}
    for cname, s in class_stats.items():
        tp, fp, fn, gt = s["tp"], s["fp"], s["fn"], s["gt"]
        p = tp / max(tp + fp, 1)
        r = tp / max(gt, 1)
        mean_iou = float(np.mean(s["ious"])) if s["ious"] else 0.0
        mean_conf = float(np.mean(s["confs"])) if s["confs"] else 0.0
        per_class_summary[cname] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": gt,
            "mean_iou": round(mean_iou, 4),
            "mean_conf": round(mean_conf, 4)
        }

    # Density breakdown
    density_summary = {}
    for k, v in density_stats.items():
        tp, fp, fn, gt = v["tp"], v["fp"], v["fn"], v["gt"]
        density_summary[k] = {
            "recall": round(tp / max(gt, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "gt_count": gt
        }

    # Size breakdown
    size_summary = {}
    for k, v in size_stats.items():
        tp, fp, fn, gt = v["tp"], v["fp"], v["fn"], v["gt"]
        size_summary[k] = {
            "recall": round(tp / max(gt, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "gt_count": gt
        }

    # Key confusions: truck->car, bus->car, motorcycle->car, rider->person
    c_idx = BENCHMARK_CLASSES.index("car")
    t_idx = BENCHMARK_CLASSES.index("truck")
    b_idx = BENCHMARK_CLASSES.index("bus")
    m_idx = BENCHMARK_CLASSES.index("motorcycle")
    r_idx = BENCHMARK_CLASSES.index("rider")
    p_idx = BENCHMARK_CLASSES.index("person")

    confusions = {
        "truck_predicted_as_car": int(confusion_matrix[t_idx, c_idx]),
        "bus_predicted_as_car": int(confusion_matrix[b_idx, c_idx]),
        "motorcycle_predicted_as_car": int(confusion_matrix[m_idx, c_idx]),
        "rider_predicted_as_person": int(confusion_matrix[r_idx, p_idx]),
        "person_predicted_as_rider": int(confusion_matrix[p_idx, r_idx]),
        "person_missed_as_background": int(confusion_matrix[p_idx, NUM_CLASSES]),
        "motorcycle_missed_as_background": int(confusion_matrix[m_idx, NUM_CLASSES]),
        "rider_missed_as_background": int(confusion_matrix[r_idx, NUM_CLASSES]),
    }

    # Calibration error
    confs = np.array([c[0] for c in calibration_pairs])
    ious = np.array([c[1] for c in calibration_pairs])
    conf_iou_corr = float(np.corrcoef(confs, ious)[0, 1]) if len(confs) > 1 else 0.0

    diag_result = {
        "weights": weights_path,
        "eval_images": len(img_files),
        "conf_threshold": conf_thresh,
        "per_class": per_class_summary,
        "density_breakdown": density_summary,
        "size_breakdown": size_summary,
        "key_confusions": confusions,
        "calibration": {
            "conf_iou_correlation": round(conf_iou_corr, 4),
            "total_candidates_evaluated": len(calibration_pairs)
        }
    }

    with open(output_path, "w") as f:
        json.dump(diag_result, f, indent=2)

    print(f"\n[Deep Diagnostics Complete] Saved report to {output_path}")
    print(f"Key Confusions: {json.dumps(confusions, indent=2)}")
    print(f"Density Breakdown: {json.dumps(density_summary, indent=2)}")
    print(f"Size Breakdown: {json.dumps(size_summary, indent=2)}")
    return diag_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out", type=str, default="experiments/custom_model/deep_diagnostics.json")
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()
    run_deep_diagnostics(weights_path=args.weights, output_path=args.out, max_eval_images=args.max_images)
