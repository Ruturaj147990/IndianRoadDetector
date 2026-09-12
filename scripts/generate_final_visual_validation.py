"""
Generate Final Visual Validation Suite on Representative Validation Images.

Selects images across different clips containing diverse vehicle types:
cars, motorcycles, autorickshaws, trucks, buses, riders, pedestrians.
Saves rendered detections to experiments/custom_model/final_visual_validation/.
"""

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
    weights_path: str = "experiments/custom_model/exp_a_full_10ep/ird_best.pt",
    val_images_dir: str = "data/indian_road_yolo/images/val",
    val_labels_dir: str = "data/indian_road_yolo/labels/val",
    output_dir: str = "experiments/custom_model/final_visual_validation",
    num_images: int = 10,
    conf_thresh: float = 0.25,
    obj_gate: float = 0.05,
    decoder_version: str = "v2_smooth",
    device_str: str = "cpu",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device, device_label = resolve_device(device_str)
    model, ckpt = load_model(weights_path, device)

    val_imgs = sorted(list(Path(val_images_dir).glob("*.jpg")))
    # Sample 10 images across different clips (stride through the sorted list)
    step = max(1, len(val_imgs) // num_images)
    selected_imgs = [val_imgs[i * step] for i in range(min(num_images, len(val_imgs)))]

    print("=" * 70)
    print("FINAL VISUAL VALIDATION SUITE")
    print(f"Weights:     {weights_path}")
    print(f"Device:      {device_label}")
    print(f"Confidence:  {conf_thresh} (zero low-conf display)")
    print(f"Output Dir:  {out_path.resolve()}")
    print("=" * 70)

    summary = []
    for idx, img_p in enumerate(selected_imgs):
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

        hud = f"IRD V1 | Val Image {idx+1}/{len(selected_imgs)} | {len(boxes)} Detections | {lat_ms:.1f}ms"
        annotated = draw_detections_cv2(img_bgr, boxes, scores, cids, conf_thresh, hud_text=hud)

        save_file = out_path / f"val_{idx+1:02d}_{img_p.stem}.jpg"
        cv2.imwrite(str(save_file), annotated)

        det_info = [f"Class {c}: {s:.2f}" for c, s in zip(cids, scores)]
        print(f"[{idx+1:02d}/{len(selected_imgs)}] {img_p.name}: {len(boxes)} detections ({lat_ms:.1f} ms)")
        summary.append({
            "image": img_p.name,
            "output": save_file.name,
            "detections": len(boxes),
            "classes": [int(c) for c in cids],
            "scores": [round(float(s), 3) for s in scores],
        })

    print("-" * 70)
    print(f"Successfully generated {len(summary)} visual validation images in {out_path}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_visual_validation()
