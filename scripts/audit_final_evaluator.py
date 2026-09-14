"""
Final IRD V1.5 Post-Processing Validation & Evaluator Audit.

Evaluates the full 1,719-image validation dataset:
1. max_det sweep: [100, 300, 500, 1000, 2000, 5000] at baseline evaluator conf=0.001
2. Full confidence score distribution (percentiles and counts/image above 0.001 to 0.50)
3. Production configuration: IoU=0.40, conf=0.25
4. Saves structured JSON to experiments/evaluation_results/final_postprocessing_audit.json
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.box_coder import (
    NUM_CLASSES,
    class_aware_nms,
    decode_ird_predictions_authoritative,
)
from src.models.custom_detector import build_detector
from scripts.evaluate_ird import (
    BENCHMARK_CLASSES,
    IOU_THRESHOLDS,
    evaluate_class_predictions,
)


def run_final_evaluator_audit(
    weights_path: str = "experiments/custom_model/final_training_50ep/ird_best.pt",
    data_yaml_path: str = "data/indian_road_yolo/data.yaml",
    img_size: int = 640,
    device_str: str = "auto",
    output_json: str = "experiments/evaluation_results/final_postprocessing_audit.json",
    max_samples: int = None,
) -> Dict[str, Any]:
    # 1. Device Setup
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 80, flush=True)
    print("FINAL IRD V1.5 POST-PROCESSING & EVALUATOR AUDIT", flush=True)
    print(f"Device:       {device}", flush=True)
    print(f"Checkpoint:   {weights_path}", flush=True)
    print(f"Dataset YAML: {data_yaml_path}", flush=True)
    print(f"Resolution:   {img_size}x{img_size}", flush=True)
    print("=" * 80, flush=True)

    # 2. Model Initialization (Zero weights modification)
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

    # 3. Dataset Resolution
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
    print(f"Loaded {n_images} validation images from {val_img_dir}", flush=True)

    # 4. Storage for full dataset evaluation
    # Per-image detections up to max_det=5000 at conf=0.001
    full_eval_dets_per_image = []  # list of list of {class_id, score, box}
    prod_dets_per_image = []       # list of list of {class_id, score, box} for prod (conf=0.25, iou=0.40)
    all_gts_per_class = {c: [] for c in range(NUM_CLASSES)}

    # Score distribution tracking
    all_candidate_scores: List[float] = []
    scores_per_image_counts = defaultdict(list)
    conf_thresholds = [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.250, 0.500]

    t_start = time.perf_counter()
    print("\nExecuting forward passes and greedy candidate extraction...", flush=True)

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

            # A. Extract full candidate detections up to max_det=5000 at conf=0.001, iou=0.50
            boxes_5k, scores_5k, classes_5k, diag = decode_ird_predictions_authoritative(
                head_out,
                img_size=img_size,
                conf_threshold=0.001,
                iou_threshold=0.50,
                max_det=5000,
                obj_gate=0.001,
                device=device,
                return_diagnostics=True,
                score_mode="sqrt_quality",
            )

            # B. Extract production detections at conf=0.25, iou=0.40, max_det=300
            boxes_prod, scores_prod, classes_prod = decode_ird_predictions_authoritative(
                head_out,
                img_size=img_size,
                conf_threshold=0.25,
                iou_threshold=0.40,
                max_det=300,
                obj_gate=0.25,
                device=device,
                return_diagnostics=False,
                score_mode="sqrt_quality",
            )

        # Record candidate score distribution stats
        s_np = scores_5k.cpu().numpy()
        # Sample for global percentiles (to prevent memory overload)
        if len(s_np) > 0:
            all_candidate_scores.extend(s_np[::max(1, len(s_np) // 50)].tolist())

        # Count per image above thresholds
        for ct in conf_thresholds:
            cnt = int((s_np >= ct).sum())
            scores_per_image_counts[ct].append(cnt)

        # Scale predictions to original image coordinates
        sx = w0 / float(img_size)
        sy = h0 / float(img_size)

        # Store baseline 5k detections
        img_5k_dets = []
        for b, s, c in zip(boxes_5k.cpu().numpy(), scores_5k.cpu().numpy(), classes_5k.cpu().numpy()):
            img_5k_dets.append({
                "img_id": img_idx,
                "score": float(s),
                "box": [float(b[0] * sx), float(b[1] * sy), float(b[2] * sx), float(b[3] * sy)],
                "class_id": int(c),
            })
        full_eval_dets_per_image.append(img_5k_dets)

        # Store production detections
        img_prod_dets = []
        for b, s, c in zip(boxes_prod.cpu().numpy(), scores_prod.cpu().numpy(), classes_prod.cpu().numpy()):
            img_prod_dets.append({
                "img_id": img_idx,
                "score": float(s),
                "box": [float(b[0] * sx), float(b[1] * sy), float(b[2] * sx), float(b[3] * sy)],
                "class_id": int(c),
            })
        prod_dets_per_image.append(img_prod_dets)

        # Load Ground Truth
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
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
                            all_gts_per_class[gt_cls].append({
                                "img_id": img_idx,
                                "box": [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                            })

        if (img_idx + 1) % 100 == 0 or (img_idx + 1) == n_images:
            elapsed = time.perf_counter() - t_start
            fps = (img_idx + 1) / elapsed
            print(f"  Processed [{img_idx + 1:>4}/{n_images}] images | FPS: {fps:.1f} ({1000.0/fps:.1f} ms/img)", flush=True)

    print(f"\nInference complete across {n_images} images in {time.perf_counter() - t_start:.1f}s!", flush=True)

    # =========================================================================
    # Task 3: Confidence Score Distribution Computation
    # =========================================================================
    print("\nComputing Confidence Score Distributions...", flush=True)
    candidate_scores_np = np.array(all_candidate_scores, dtype=np.float32)
    percentiles_to_calc = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    percentile_dict = {
        f"P{p}": float(np.percentile(candidate_scores_np, p))
        for p in percentiles_to_calc
    }

    counts_above_thresh_dict = {}
    for ct in conf_thresholds:
        counts = scores_per_image_counts[ct]
        counts_above_thresh_dict[f"above_{ct}"] = {
            "threshold": ct,
            "avg_per_image": float(np.mean(counts)),
            "median_per_image": float(np.median(counts)),
            "max_per_image": int(np.max(counts)),
            "min_per_image": int(np.min(counts)),
            "total_count": int(np.sum(counts)),
        }

    # =========================================================================
    # Task 2: max_det Sweep Evaluation [100, 300, 500, 1000, 2000, 5000]
    # =========================================================================
    max_det_values = [100, 300, 500, 1000, 2000, 5000]
    max_det_results: Dict[str, Any] = {}

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION SWEEP: Varying max_det at Authoritative Baseline (conf=0.001, iou=0.50)", flush=True)
    print("=" * 80, flush=True)

    for md in max_det_values:
        t_eval0 = time.perf_counter()

        # Gather sliced predictions per class
        all_preds_sliced = {c: [] for c in range(NUM_CLASSES)}
        det_counts_per_img = []

        for img_dets in full_eval_dets_per_image:
            sliced = img_dets[:md]
            det_counts_per_img.append(len(sliced))
            for d in sliced:
                all_preds_sliced[d["class_id"]].append({
                    "img_id": d["img_id"],
                    "score": d["score"],
                    "box": d["box"],
                })

        # Calculate metrics using 62x vectorized evaluation engine
        per_class_res = {}
        ap50_list, ap50_95_list, prec_list, rec_list = [], [], [], []

        for c_id, c_name in enumerate(BENCHMARK_CLASSES):
            res = evaluate_class_predictions(
                all_preds_sliced[c_id],
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
        eval_time = time.perf_counter() - t_eval0

        max_det_results[f"max_det_{md}"] = {
            "max_det": md,
            "total_predictions": sum(det_counts_per_img),
            "avg_predictions_per_img": float(np.mean(det_counts_per_img)),
            "max_predictions_per_img": int(np.max(det_counts_per_img)),
            "min_predictions_per_img": int(np.min(det_counts_per_img)),
            "precision": round(mean_prec, 4),
            "recall": round(mean_rec, 4),
            "mAP50": round(mAP50, 4),
            "mAP50_95": round(mAP50_95, 4),
            "per_class_ap50": {c_name: round(per_class_res[c_name]["ap50"], 4) for c_name in BENCHMARK_CLASSES},
            "eval_time_sec": round(eval_time, 2),
        }

        print(f"  max_det={md:>5} | Dets={sum(det_counts_per_img):>7} | Avg={np.mean(det_counts_per_img):>6.1f}/img | Max={np.max(det_counts_per_img):>4} | P={mean_prec:.4f} | R={mean_rec:.4f} | mAP50={mAP50:.4f} | mAP50-95={mAP50_95:.4f} ({eval_time:.1f}s)", flush=True)

    # =========================================================================
    # Task 1 & 5: Production Configuration Evaluation (conf=0.25, iou=0.40)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("EVALUATION: Production Configuration (conf=0.25, iou=0.40, max_det=300)", flush=True)
    print("=" * 80, flush=True)

    all_prod_sliced = {c: [] for c in range(NUM_CLASSES)}
    prod_counts_per_img = []

    for img_dets in prod_dets_per_image:
        prod_counts_per_img.append(len(img_dets))
        for d in img_dets:
            all_prod_sliced[d["class_id"]].append({
                "img_id": d["img_id"],
                "score": d["score"],
                "box": d["box"],
            })

    prod_per_class_res = {}
    p_ap50, p_ap50_95, p_prec, p_rec = [], [], [], []

    for c_id, c_name in enumerate(BENCHMARK_CLASSES):
        res = evaluate_class_predictions(
            all_prod_sliced[c_id],
            all_gts_per_class[c_id],
            class_id=c_id,
            iou_thresholds=IOU_THRESHOLDS,
        )
        prod_per_class_res[c_name] = res
        if res["n_gt"] > 0:
            p_ap50.append(res["ap50"])
            p_ap50_95.append(res["ap50_95"])
            p_prec.append(res["precision_50"])
            p_rec.append(res["recall_50"])

    prod_results = {
        "conf_threshold": 0.25,
        "iou_threshold": 0.40,
        "max_det": 300,
        "total_predictions": sum(prod_counts_per_img),
        "avg_predictions_per_img": float(np.mean(prod_counts_per_img)),
        "max_predictions_per_img": int(np.max(prod_counts_per_img)),
        "min_predictions_per_img": int(np.min(prod_counts_per_img)),
        "precision": round(float(np.mean(p_prec)), 4) if p_prec else 0.0,
        "recall": round(float(np.mean(p_rec)), 4) if p_rec else 0.0,
        "mAP50": round(float(np.mean(p_ap50)), 4) if p_ap50 else 0.0,
        "mAP50_95": round(float(np.mean(p_ap50_95)), 4) if p_ap50_95 else 0.0,
        "per_class_ap50": {c_name: round(prod_per_class_res[c_name]["ap50"], 4) for c_name in BENCHMARK_CLASSES},
    }

    print(f"  Production | Dets={prod_results['total_predictions']:>7} | Avg={prod_results['avg_predictions_per_img']:>6.1f}/img | Max={prod_results['max_predictions_per_img']:>4} | P={prod_results['precision']:.4f} | R={prod_results['recall']:.4f} | mAP50={prod_results['mAP50']:.4f} | mAP50-95={prod_results['mAP50_95']:.4f}", flush=True)

    # =========================================================================
    # Task 8: Save Structured JSON
    # =========================================================================
    final_audit_data = {
        "metadata": {
            "model": "IRD V1.5 (IndianRoadDetection)",
            "checkpoint": weights_path,
            "total_val_images": n_images,
            "image_size": img_size,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "score_distribution": {
            "percentiles": percentile_dict,
            "counts_per_image_above_threshold": counts_above_thresh_dict,
        },
        "max_det_sweep": max_det_results,
        "production_configuration": prod_results,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_audit_data, f, indent=2)

    print(f"\nAudit complete! Structured results saved to: {out_p.resolve()}", flush=True)
    return final_audit_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final IRD V1.5 Post-Processing Validation & Evaluator Audit")
    parser.add_argument("--weights", type=str, default="experiments/custom_model/final_training_50ep/ird_best.pt")
    parser.add_argument("--data", type=str, default="data/indian_road_yolo/data.yaml")
    parser.add_argument("--output", type=str, default="experiments/evaluation_results/final_postprocessing_audit.json")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_final_evaluator_audit(
        weights_path=args.weights,
        data_yaml_path=args.data,
        output_json=args.output,
        max_samples=args.samples,
        device_str=args.device,
    )
