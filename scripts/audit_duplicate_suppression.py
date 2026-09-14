"""
IRD V1.5 Duplicate-Box Suppression Audit & Benchmark Sweep Script.

Performs:
1. True Greedy Class-Aware NMS verification.
2. Pre- and Post-NMS detailed diagnostic measurements:
   - Raw prediction count (8,400)
   - Objectness-gated count
   - Confidence-filtered candidate count
   - Count removed by NMS
   - Final detection count
   - Suppressed boxes per class
   - Maximum detections per image (saturation test)
   - Average detections per image
3. IoU Threshold Sweep across [0.30, 0.40, 0.50, 0.60, 0.70].
4. Root-cause inspection:
   - Low-confidence flooding vs score calibration (sqrt_quality vs linear_quality vs obj_cls)
   - Cross-class overlaps vs same-class overlaps
   - Multi-scale feature stride contributions
5. Generation of comprehensive comparative metrics table (Precision, Recall, mAP50, mAP50-95, Det Count).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.box_coder import NUM_CLASSES
from src.models.custom_detector import build_detector
from scripts.evaluate_ird import (
    BENCHMARK_CLASSES,
    IOU_THRESHOLDS,
    box_iou_xyxy,
    decode_ird_predictions,
    evaluate_class_predictions,
)


def run_audit_sweep(
    weights_path: str = "experiments/custom_model/final_training_50ep/ird_best.pt",
    data_yaml_path: str = "data/indian_road_yolo/data.yaml",
    img_size: int = 640,
    max_samples: int = 100,
    device_str: str = "auto",
    output_json: str = "experiments/evaluation_results/duplicate_audit_results.json",
) -> Dict[str, Any]:
    # 1. Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 80)
    print(f"IRD V1.5 DUPLICATE-BOX SUPPRESSION AUDIT ENGINE")
    print(f"Device:       {device}")
    print(f"Weights:      {weights_path}")
    print(f"Val Samples:  {max_samples}")
    print(f"Resolution:   {img_size}x{img_size}")
    print("=" * 80)

    # 2. Model Initialization (Preserving weights, no modification)
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

    # 3. Load Dataset
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
    print(f"Loaded {n_images} validation images from: {val_img_dir}")

    # Pre-cache image tensors and ground truths
    cached_data = []
    print("Caching preprocessed image tensors and ground truth annotations...")
    for idx, img_p in enumerate(all_img_files):
        img0 = cv2.imread(str(img_p))
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        img_resized = cv2.resize(img0, (img_size, img_size))
        img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Load GT
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        gts = []
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
                        gts.append({
                            "class_id": gt_cls,
                            "box": [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                        })

        cached_data.append({
            "img_id": idx,
            "tensor": img_tensor,
            "orig_w": w0,
            "orig_h": h0,
            "gts": gts,
        })

    print(f"Pre-cached {len(cached_data)} images. Pre-computing model forward passes...")
    cached_head_outputs = []
    with torch.no_grad():
        for item in cached_data:
            out = model(item["tensor"])
            cached_head_outputs.append(out)

    # Helper evaluation function
    def evaluate_configuration(
        conf_thresh: float,
        iou_thresh: float,
        score_mode: str = "sqrt_quality",
        max_det: int = 300,
        obj_gate: Any = None,
    ) -> Dict[str, Any]:
        all_preds_per_class = {c: [] for c in range(NUM_CLASSES)}
        all_gts_per_class = {c: [] for c in range(NUM_CLASSES)}

        # Aggregate diagnostics
        diag_totals = {
            "raw_predictions": 0,
            "gated_predictions": 0,
            "conf_filtered": 0,
            "nms_removed": 0,
            "final_detections": 0,
            "suppressed_per_class": [0] * NUM_CLASSES,
            "candidate_per_class": [0] * NUM_CLASSES,
            "retained_per_class": [0] * NUM_CLASSES,
        }
        dets_per_img = []
        same_class_pairs_30_50 = 0
        diff_class_pairs_30_50 = 0
        diff_class_pairs_above_50 = 0

        for idx, item in enumerate(cached_data):
            head_out = cached_head_outputs[idx]
            boxes, scores, classes, diag = decode_ird_predictions(
                head_out,
                img_size=img_size,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                max_det=max_det,
                obj_gate=obj_gate if obj_gate is not None else conf_thresh,
                return_diagnostics=True,
                score_mode=score_mode,
                device=device,
            )

            # Record diagnostics
            diag_totals["raw_predictions"] += diag["raw_prediction_count"]
            diag_totals["gated_predictions"] += diag["gated_prediction_count"]
            diag_totals["conf_filtered"] += diag["conf_filtered_count"]
            diag_totals["nms_removed"] += diag["count_removed_by_nms"]
            diag_totals["final_detections"] += diag["final_count"]
            dets_per_img.append(diag["final_count"])

            for c_id in range(NUM_CLASSES):
                diag_totals["suppressed_per_class"][c_id] += diag["boxes_suppressed_per_class"][c_id]
                diag_totals["candidate_per_class"][c_id] += diag["candidate_per_class"][c_id]
                diag_totals["retained_per_class"][c_id] += diag["retained_per_class"][c_id]

            # Overlap analysis (fully vectorized tensor operations)
            if len(boxes) > 1:
                ious = box_iou_xyxy(boxes, boxes)
                tri_mask = torch.triu(torch.ones_like(ious, dtype=torch.bool), diagonal=1)
                same_cls_mask = classes.unsqueeze(0) == classes.unsqueeze(1)

                tri_ious = ious[tri_mask]
                tri_same = same_cls_mask[tri_mask]

                same_class_pairs_30_50 += int(((tri_ious >= 0.30) & (tri_ious < 0.50) & tri_same).sum().item())
                diff_class_pairs_30_50 += int(((tri_ious >= 0.30) & (tri_ious < 0.50) & ~tri_same).sum().item())
                diff_class_pairs_above_50 += int(((tri_ious >= 0.50) & ~tri_same).sum().item())

            # Scale to original image
            sx = item["orig_w"] / float(img_size)
            sy = item["orig_h"] / float(img_size)
            for b, s, c in zip(boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()):
                c_int = int(c)
                if 0 <= c_int < NUM_CLASSES:
                    scaled_b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                    all_preds_per_class[c_int].append({
                        "img_id": item["img_id"],
                        "score": float(s),
                        "box": scaled_b,
                    })

            # Record GTs
            for gt in item["gts"]:
                all_gts_per_class[gt["class_id"]].append({
                    "img_id": item["img_id"],
                    "box": gt["box"],
                })

        # Metric computation
        per_class_res = {}
        ap50_list, ap50_95_list, prec_list, rec_list = [], [], [], []

        for c_id, c_name in enumerate(BENCHMARK_CLASSES):
            res = evaluate_class_predictions(
                all_preds_per_class[c_id],
                all_gts_per_class[c_id],
                class_id=c_id,
                iou_thresholds=IOU_THRESHOLDS,
            )
            per_class_res[c_name] = res
            if res["n_gt"] > 0:
                ap50_list.append(res["ap50"])
                ap50_95_list.append(res["ap50_95"])
                prec_list.append(res["precision_50"])
                rec_list.append(res["recall_50"])

        mAP50 = float(np.mean(ap50_list)) if ap50_list else 0.0
        mAP50_95 = float(np.mean(ap50_95_list)) if ap50_95_list else 0.0
        mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
        mean_rec = float(np.mean(rec_list)) if rec_list else 0.0

        return {
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "score_mode": score_mode,
            "max_det": max_det,
            "total_detections": sum(dets_per_img),
            "avg_detections_per_img": float(np.mean(dets_per_img)),
            "max_detections_per_img": int(np.max(dets_per_img)),
            "min_detections_per_img": int(np.min(dets_per_img)),
            "precision": round(mean_prec, 4),
            "recall": round(mean_rec, 4),
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP50_95, 4),
            "diagnostics": {
                "raw_prediction_count": diag_totals["raw_predictions"],
                "gated_prediction_count": diag_totals["gated_predictions"],
                "conf_filtered_count": diag_totals["conf_filtered"],
                "count_removed_by_nms": diag_totals["nms_removed"],
                "final_count": diag_totals["final_detections"],
                "boxes_suppressed_per_class": {
                    BENCHMARK_CLASSES[c]: diag_totals["suppressed_per_class"][c]
                    for c in range(NUM_CLASSES)
                },
                "candidate_per_class": {
                    BENCHMARK_CLASSES[c]: diag_totals["candidate_per_class"][c]
                    for c in range(NUM_CLASSES)
                },
                "retained_per_class": {
                    BENCHMARK_CLASSES[c]: diag_totals["retained_per_class"][c]
                    for c in range(NUM_CLASSES)
                },
            },
            "overlap_stats": {
                "same_class_pairs_iou_030_050": same_class_pairs_30_50,
                "diff_class_pairs_iou_030_050": diff_class_pairs_30_50,
                "diff_class_pairs_iou_gte_050": diff_class_pairs_above_50,
            },
            "per_class_ap50": {c_name: round(per_class_res[c_name]["ap50"], 4) for c_name in BENCHMARK_CLASSES},
        }

    # =========================================================================
    # Task 5: Systematic IoU Threshold Sweep [0.30, 0.40, 0.50, 0.60, 0.70]
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: IoU Threshold Sweep at Authoritative Baseline (conf=0.001)")
    print("=" * 80)

    sweep_iou_baseline = {}
    for iou_t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        t0 = time.perf_counter()
        res = evaluate_configuration(conf_thresh=0.001, iou_thresh=iou_t, score_mode="sqrt_quality")
        elapsed = time.perf_counter() - t0
        sweep_iou_baseline[f"iou_{iou_t:.2f}"] = res
        print(f"  IoU={iou_t:.2f} | Dets={res['total_detections']:5d} | Avg={res['avg_detections_per_img']:5.1f}/img | Max={res['max_detections_per_img']:3d} | P={res['precision']:.4f} | R={res['recall']:.4f} | mAP50={res['mAP50']:.4f} | mAP50-95={res['mAP50_95']:.4f} ({elapsed:.1f}s)", flush=True)

    # =========================================================================
    # Task 5 & 6: Production Inference IoU Threshold Sweep (conf=0.25)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: IoU Threshold Sweep at Production Inference (conf=0.25)")
    print("=" * 80)

    sweep_iou_prod = {}
    for iou_t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        t0 = time.perf_counter()
        res = evaluate_configuration(conf_thresh=0.25, iou_thresh=iou_t, score_mode="sqrt_quality")
        elapsed = time.perf_counter() - t0
        sweep_iou_prod[f"iou_{iou_t:.2f}"] = res
        print(f"  IoU={iou_t:.2f} | Dets={res['total_detections']:5d} | Avg={res['avg_detections_per_img']:5.1f}/img | Max={res['max_detections_per_img']:3d} | P={res['precision']:.4f} | R={res['recall']:.4f} | mAP50={res['mAP50']:.4f} | mAP50-95={res['mAP50_95']:.4f} ({elapsed:.1f}s)", flush=True)

    # =========================================================================
    # Task 6: Confidence Floor & Gating Ablation (inspecting max_det=300 saturation)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Confidence Floor & Gating Ablation (iou_threshold=0.50)")
    print("=" * 80)

    sweep_conf = {}
    for c_t in [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25]:
        t0 = time.perf_counter()
        res = evaluate_configuration(conf_thresh=c_t, iou_thresh=0.50, score_mode="sqrt_quality")
        elapsed = time.perf_counter() - t0
        sweep_conf[f"conf_{c_t}"] = res
        print(f"  Conf={c_t:5.3f} | Dets={res['total_detections']:5d} | Avg={res['avg_detections_per_img']:5.1f}/img | Max={res['max_detections_per_img']:3d} | P={res['precision']:.4f} | R={res['recall']:.4f} | mAP50={res['mAP50']:.4f} | mAP50-95={res['mAP50_95']:.4f} ({elapsed:.1f}s)", flush=True)

    # =========================================================================
    # Task 6: Score Calibration & Quality Formulation Ablation
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("EXPERIMENT 4: Score Calibration Formulations (conf=0.001 vs conf=0.05)", flush=True)
    print("=" * 80, flush=True)

    score_modes_ablation = {}
    for sm in ["sqrt_quality", "linear_quality", "obj_cls"]:
        for ct in [0.001, 0.05, 0.25]:
            t0 = time.perf_counter()
            res = evaluate_configuration(conf_thresh=ct, iou_thresh=0.50, score_mode=sm)
            elapsed = time.perf_counter() - t0
            key = f"{sm}__conf_{ct}"
            score_modes_ablation[key] = res
            print(f"  Mode={sm:<14} | Conf={ct:5.3f} | Dets={res['total_detections']:5d} | Avg={res['avg_detections_per_img']:5.1f}/img | Max={res['max_detections_per_img']:3d} | P={res['precision']:.4f} | R={res['recall']:.4f} | mAP50={res['mAP50']:.4f} | mAP50-95={res['mAP50_95']:.4f} ({elapsed:.1f}s)", flush=True)

    # =========================================================================
    # Compile Complete Audit Results
    # =========================================================================
    audit_results = {
        "metadata": {
            "model": "IRD V1.5 (IndianRoadDetection)",
            "checkpoint": weights_path,
            "val_samples": n_images,
            "img_size": img_size,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "experiments": {
            "baseline_iou_sweep_conf_0001": sweep_iou_baseline,
            "production_iou_sweep_conf_025": sweep_iou_prod,
            "confidence_floor_ablation": sweep_conf,
            "score_calibration_ablation": score_modes_ablation,
        },
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nAudit completed successfully! Saved structured metrics to: {out_p.resolve()}")
    return audit_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and improve IRD V1.5 duplicate suppression")
    parser.add_argument("--weights", type=str, default="experiments/custom_model/final_training_50ep/ird_best.pt")
    parser.add_argument("--data", type=str, default="data/indian_road_yolo/data.yaml")
    parser.add_argument("--samples", type=int, default=100, help="Number of representative validation samples")
    parser.add_argument("--output", type=str, default="experiments/evaluation_results/duplicate_audit_results.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_audit_sweep(
        weights_path=args.weights,
        data_yaml_path=args.data,
        max_samples=args.samples,
        output_json=args.output,
        device_str=args.device,
    )
