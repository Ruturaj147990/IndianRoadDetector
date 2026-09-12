"""
Generate Targeted Final Visual Validation Suite for IRD V1 (IndianRoadDetection).

Selects validation images across all primary real-world failure cases and diagnostic subsets:
A. Dense Cars
B. Dense Motorcycles
C. Car + Motorcycle Mix
D. Motorcycle + Rider Pairs
E. Dense Traffic
F. Small / Distant Objects
G. Side Views
H. Diagonal Views
I. Night Scenes
J. Difficult Illumination
K. Pedestrians & Complex Intersections

Renders high-contrast bounding boxes with class labels, confidence scores, and HUD metrics.
"""

import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.infer_ird import (
    load_model,
    resolve_device,
    infer_single_image,
    draw_detections_cv2,
)


def run_visual_validation(
    weights_path: str = "experiments/custom_model/exp_loop2_adaptive_topk/ird_best.pt",
    subsets_json: str = "experiments/custom_model/val_diagnostic_subsets.json",
    val_images_dir: str = "data/indian_road_yolo/images/val",
    output_dir: str = "experiments/custom_model/final_visual_validation",
    conf_thresh: float = 0.20,
    obj_gate: float = 0.05,
    decoder_version: str = "v2_smooth",
    device_str: str = "cuda",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device, device_label = resolve_device(device_str)
    model, ckpt = load_model(weights_path, device)

    val_img_path = Path(val_images_dir)
    subsets_path = Path(subsets_json)

    selected_cases = []
    if subsets_path.exists():
        with open(subsets_path, "r") as f:
            subsets = json.load(f)
        for cat_name, img_list in subsets.items():
            for img_name in img_list:
                img_file = val_img_path / img_name
                if img_file.exists():
                    clean_cat = cat_name.replace("_", " ").title()
                    selected_cases.append((clean_cat, img_file))
                    break

    # Fallback / supplement with sorted stride images if needed
    if len(selected_cases) < 10:
        val_imgs = sorted(list(val_img_path.glob("*.jpg")))
        step = max(1, len(val_imgs) // 10)
        for i in range(10):
            selected_cases.append((f"Val Stride Sample {i+1}", val_imgs[i * step]))

    print("=" * 75)
    print("IRD V1 TARGETED VISUAL VALIDATION SUITE (10+ SCENARIOS)")
    print("=" * 75)
    print(f"Model Checkpoint:  {weights_path}")
    print(f"Inference Device:  {device_label}")
    print(f"Confidence Thresh: {conf_thresh} (only verified accepted detections)")
    print(f"Output Directory:  {out_path.resolve()}")
    print("-" * 75)

    summary = []
    for idx, (cat_label, img_p) in enumerate(selected_cases):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        boxes, scores, cids, lat_ms, timing = infer_single_image(
            model=model,
            image_bgr=img_bgr,
            img_size=640,
            conf_thresh=conf_thresh,
            iou_thresh=0.50,
            device=device,
            max_det=300,
            obj_gate=obj_gate,
            decoder_version=decoder_version,
        )

        hud = f"IRD V1 | {cat_label} | {len(boxes)} Dets | {lat_ms:.1f}ms | Conf >= {conf_thresh}"
        annotated = draw_detections_cv2(img_bgr, boxes, scores, cids, conf_thresh, hud_text=hud)

        slug = cat_label.lower().replace(" ", "_")[:24]
        save_file = out_path / f"val_{idx+1:02d}_{slug}_{img_p.stem}.jpg"
        cv2.imwrite(str(save_file), annotated)

        print(f"[{idx+1:02d}/{len(selected_cases)}] {cat_label:<25} | {img_p.name} | {len(boxes):>2} detections ({lat_ms:.1f} ms)")
        summary.append({
            "category": cat_label,
            "image": img_p.name,
            "output": save_file.name,
            "detections": len(boxes),
            "classes": [int(c) for c in cids],
            "scores": [round(float(s), 3) for s in scores],
        })

    summary_file = out_path / "visual_validation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 75)
    print(f"Successfully generated {len(summary)} visual validation outputs in: {out_path.resolve()}")
    print("=" * 75)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="experiments/custom_model/exp_loop2_adaptive_topk/ird_best.pt")
    parser.add_argument("--subsets", type=str, default="experiments/custom_model/val_diagnostic_subsets.json")
    parser.add_argument("--val-dir", type=str, default="data/indian_road_yolo/images/val")
    parser.add_argument("--out", type=str, default="experiments/custom_model/final_visual_validation")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--obj-gate", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_visual_validation(
        weights_path=args.weights,
        subsets_json=args.subsets,
        val_images_dir=args.val_dir,
        output_dir=args.out,
        conf_thresh=args.conf,
        obj_gate=args.obj_gate,
        device_str=args.device,
    )
