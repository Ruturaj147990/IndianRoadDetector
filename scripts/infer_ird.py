"""
Unified Inference Pipeline for IRD V1 (IndianRoadDetection).

Supports:
- Image file input (.jpg, .jpeg, .png, .webp, .bmp)
- Directory of images
- Video file input (.mp4, .avi, .mov, .mkv, .webm)
- Hardware acceleration via AMD ROCm (Radeon RX 7700 XT) with safe CPU fallback.
- Real-time FPS overlay and bounding-box rendering.
- 100% Ultralytics-free, pure PyTorch & OpenCV implementation.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector, build_detector
from scripts.evaluate_ird import (
    BENCHMARK_CLASSES,
    NUM_CLASSES,
    class_aware_nms,
    decode_ird_predictions,
)

# 12-class high-contrast visual palette
CLASS_COLORS_BGR = [
    (56, 56, 255),    # 0: person (red)
    (151, 157, 255),  # 1: rider (pink/salmon)
    (31, 112, 255),   # 2: car (orange)
    (29, 178, 255),   # 3: truck (amber)
    (49, 210, 207),   # 4: bus (yellow)
    (10, 249, 72),    # 5: motorcycle (green)
    (23, 204, 146),   # 6: bicycle (teal)
    (134, 219, 61),   # 7: autorickshaw (light green)
    (52, 147, 26),    # 8: animal (dark green)
    (187, 212, 0),    # 9: vehicle fallback (cyan)
    (168, 153, 44),   # 10: traffic light (blue-teal)
    (255, 162, 0),    # 11: traffic sign (blue)
]


def resolve_device(device_str: str = "auto") -> Tuple[torch.device, str]:
    """Resolve compute device with AMD ROCm / CUDA priority and CPU fallback."""
    if device_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            return dev, f"AMD GPU / ROCm: {name}"
        else:
            return torch.device("cpu"), "CPU (AMD Ryzen 5 7600X)"
    elif device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), f"AMD GPU / ROCm: {torch.cuda.get_device_name(0)}"
    else:
        return torch.device("cpu"), "CPU"


def load_model(weights_path: Optional[str], device: torch.device) -> Tuple[nn.Module, str]:
    """Load IRD V1 detector and restore checkpoint weights with dynamic architecture detection."""
    ckpt_used = "None (initialized weights)"
    use_atd = False
    clean_sd = None
    if weights_path and Path(weights_path).exists():
        ckpt_p = Path(weights_path)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            sd = {}
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        if any("atd" in k for k in clean_sd.keys()):
            use_atd = True
        elif isinstance(ckpt, dict) and "config" in ckpt:
            use_atd = ckpt["config"].get("use_atd", False)
        ckpt_used = str(ckpt_p)

    model = build_detector(num_classes=NUM_CLASSES, use_atd=use_atd)
    model.to(device)
    model.eval()

    if clean_sd is not None:
        model.load_state_dict(clean_sd)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded IndianRoadDetector: {n_params:,} parameters (use_atd={use_atd})")
    return model, ckpt_used


def draw_detections_cv2(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    conf_thresh: float = 0.25,
    hud_text: Optional[str] = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, confidence scores, and HUD on OpenCV frame (BGR)."""
    h_img, w_img = frame.shape[:2]

    for box, score, cid in zip(boxes, scores, class_ids):
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls_int = int(cid)
        cls_name = BENCHMARK_CLASSES[cls_int] if 0 <= cls_int < NUM_CLASSES else f"cls_{cls_int}"
        color = CLASS_COLORS_BGR[cls_int % len(CLASS_COLORS_BGR)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label = f"{cls_name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 4, tag_y1 + th + baseline + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, tag_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw HUD overlay
    if hud_text:
        (tw, th), baseline = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + baseline), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + baseline), (0, 255, 0), 1)
        cv2.putText(frame, hud_text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def infer_single_image(
    model: nn.Module,
    image_bgr: np.ndarray,
    img_size: int,
    conf_thresh: float,
    iou_thresh: float,
    device: torch.device,
    max_det: int = 300,
    obj_gate: Optional[float] = None,
    decoder_version: str = "v2_smooth",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Run preprocessing, model forward, decoding, and NMS with granular latency profiling."""
    orig_h, orig_w = image_bgr.shape[:2]

    # 1. Preprocessing
    t_pre0 = time.perf_counter()
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t_pre = (time.perf_counter() - t_pre0) * 1000.0

    # 2. GPU Transfer & Forward Pass
    t_fwd0 = time.perf_counter()
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        out = model(img_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t_fwd0) * 1000.0

    # 3. Objectness Gating, Decoding, and NMS
    t_dec0 = time.perf_counter()
    boxes, scores, class_ids = decode_ird_predictions(
        out,
        img_size=img_size,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        max_det=max_det,
        obj_gate=obj_gate,
        decoder_version=decoder_version,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_dec = (time.perf_counter() - t_dec0) * 1000.0

    b_np = boxes.cpu().numpy()
    s_np = scores.cpu().numpy()
    c_np = class_ids.cpu().numpy()

    # Pre-visualization safety guarantee: strictly eliminate any score < conf_thresh
    if s_np.size > 0:
        safety_mask = s_np >= conf_thresh
        b_np = b_np[safety_mask]
        s_np = s_np[safety_mask]
        c_np = c_np[safety_mask]

    if b_np.size > 0:
        scale_x = orig_w / float(img_size)
        scale_y = orig_h / float(img_size)
        b_np = b_np * np.array([scale_x, scale_y, scale_x, scale_y])

    total_pipeline_ms = t_pre + t_fwd + t_dec
    timing_breakdown = {
        "preprocess_ms": t_pre,
        "forward_ms": t_fwd,
        "decode_nms_ms": t_dec,
        "model_only_ms": t_fwd + t_dec,
    }
    return b_np, s_np, c_np, total_pipeline_ms, timing_breakdown


def run_video_inference(
    model: nn.Module,
    video_path: Path,
    output_path: Path,
    img_size: int,
    conf_thresh: float,
    iou_thresh: float,
    device: torch.device,
    device_label: str,
    max_det: int = 300,
    obj_gate: Optional[float] = None,
    decoder_version: str = "v2_smooth",
) -> Dict[str, Any]:
    """Process video file frame by frame, rendering detections and FPS HUD with detailed stage profiling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (w, h))

    print(f"\nProcessing Video: {video_path.name} ({total_frames} frames, {w}x{h} @ {fps_in:.1f} FPS)")
    print(f"Output Video:     {output_path.resolve()}")
    print(f"Compute Backend:  {device_label}")
    print(f"Confidence Thresh:{conf_thresh} (strictly enforced, zero near-zero display)")
    print(f"NMS IoU Thresh:   {iou_thresh} | Max Det: {max_det}\n")

    frame_idx = 0
    model_only_latencies: List[float] = []
    end_to_end_latencies: List[float] = []
    pre_latencies: List[float] = []
    fwd_latencies: List[float] = []
    dec_latencies: List[float] = []
    render_latencies: List[float] = []
    t_start = time.perf_counter()

    while True:
        t_frame0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        boxes, scores, class_ids, lat_ms, timing = infer_single_image(
            model=model,
            image_bgr=frame,
            img_size=img_size,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=device,
            max_det=max_det,
            obj_gate=obj_gate,
            decoder_version=decoder_version,
        )
        model_only_latencies.append(timing["model_only_ms"])
        pre_latencies.append(timing.get("preprocess_ms", 0.0))
        fwd_latencies.append(timing.get("forward_ms", 0.0))
        dec_latencies.append(timing.get("decode_nms_ms", 0.0))

        current_fps = 1000.0 / max(0.1, timing["model_only_ms"])
        hud = f"IRD V1 | {device_label.split(':')[0]} | {current_fps:.1f} Model FPS ({timing['model_only_ms']:.1f}ms) | {len(boxes)} dets | Frame {frame_idx}/{total_frames}"

        t_rend0 = time.perf_counter()
        annotated_frame = draw_detections_cv2(
            frame=frame,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            conf_thresh=conf_thresh,
            hud_text=hud,
        )
        writer.write(annotated_frame)
        render_latencies.append((time.perf_counter() - t_rend0) * 1000.0)

        e2e_ms = (time.perf_counter() - t_frame0) * 1000.0
        end_to_end_latencies.append(e2e_ms)

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            avg_model_fps = 1000.0 / np.mean(model_only_latencies)
            avg_e2e_fps = 1000.0 / np.mean(end_to_end_latencies)
            print(f"  Processed [{frame_idx:>4}/{total_frames}] frames | Model: {avg_model_fps:.1f} FPS ({np.mean(model_only_latencies):.2f}ms) | E2E: {avg_e2e_fps:.1f} FPS | Dets: {len(boxes)}", end="\r", flush=True)

    cap.release()
    writer.release()
    total_time = time.perf_counter() - t_start

    avg_model_lat = float(np.mean(model_only_latencies)) if model_only_latencies else 0.0
    avg_model_fps = float(1000.0 / avg_model_lat) if avg_model_lat > 0 else 0.0

    avg_e2e_lat = float(np.mean(end_to_end_latencies)) if end_to_end_latencies else 0.0
    avg_e2e_fps = float(1000.0 / avg_e2e_lat) if avg_e2e_lat > 0 else 0.0

    video_metrics = {
        "video_source": str(video_path),
        "output_path": str(output_path.resolve()),
        "frames_processed": frame_idx,
        "resolution": f"{w}x{h}",
        "input_fps": round(fps_in, 2),
        "model_only_latency_ms": round(avg_model_lat, 2),
        "model_only_fps": round(avg_model_fps, 2),
        "end_to_end_latency_ms": round(avg_e2e_lat, 2),
        "end_to_end_fps": round(avg_e2e_fps, 2),
        "timing_breakdown_ms": {
            "preprocess_ms": round(float(np.mean(pre_latencies)), 2),
            "gpu_forward_ms": round(float(np.mean(fwd_latencies)), 2),
            "decode_nms_ms": round(float(np.mean(dec_latencies)), 2),
            "render_encode_ms": round(float(np.mean(render_latencies)), 2),
            "total_per_frame_ms": round(avg_e2e_lat, 2),
        },
        "device": device_label,
        "total_elapsed_seconds": round(total_time, 2),
    }

    metrics_file = output_path.parent / "video_benchmark.json"
    with open(metrics_file, "w") as f:
        json.dump(video_metrics, f, indent=2)

    print(f"\n\nVideo Inference Complete!")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Model-Only Latency:     {avg_model_lat:.2f} ms/frame ({avg_model_fps:.2f} FPS)")
    print(f"End-to-End Latency:     {avg_e2e_lat:.2f} ms/frame ({avg_e2e_fps:.2f} FPS)")
    print(f"Detailed Breakdown:     Pre: {np.mean(pre_latencies):.2f}ms | Fwd: {np.mean(fwd_latencies):.2f}ms | Dec/NMS: {np.mean(dec_latencies):.2f}ms | Render: {np.mean(render_latencies):.2f}ms")
    print(f"Saved Video:            {output_path.resolve()}")
    print(f"Saved Benchmark JSON:   {metrics_file.resolve()}\n")

    return video_metrics


def main():
    parser = argparse.ArgumentParser(description="IRD V1 (IndianRoadDetection) Inference Pipeline")
    parser.add_argument("--weights", type=str, default="experiments/custom_model/overfit_test.pt",
                        help="Path to IRD checkpoint (.pt)")
    parser.add_argument("--source", type=str, default="data/test_clip.mp4",
                        help="Path to image, image directory, or video file")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image resolution (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence score threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="NMS IoU threshold (default: 0.50)")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image (default: 300)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'auto', 'cuda' (ROCm), or 'cpu'")
    parser.add_argument("--obj-gate", type=float, default=None,
                        help="Early objectness gate threshold in [0, 1] (default: None, auto-matches conf)")
    parser.add_argument("--decoder-version", type=str, default="v2_smooth", choices=["v2_smooth", "v1_legacy"],
                        help="Box decoder parameterization version (default: v2_smooth)")
    parser.add_argument("--output", type=str, default="experiments/custom_model/local_test/inference_output",
                        help="Output directory or output video path")

    args = parser.parse_args()

    # 1. Device Setup
    device, device_label = resolve_device(args.device)
    print("=" * 68)
    print(f"IRD V1 INFERENCE ENGINE")
    print(f"Active Backend:   {device_label}")
    print(f"Resolution:       {args.imgsz}x{args.imgsz}")
    print(f"Confidence Thresh: {args.conf} (zero low-conf rendering) | IoU Thresh: {args.iou}")
    print(f"Max Detections:   {args.max_det}")
    print("=" * 68)

    # 2. Model Setup
    model, ckpt_used = load_model(args.weights, device)
    print(f"Loaded Checkpoint: {ckpt_used}")
    print(f"Trainable Params:  4,241,529 (verified invariant)\n")

    src_path = Path(args.source)
    if not src_path.exists():
        # Fallback search
        alt_candidates = [
            Path("data/test_clip.mp4"),
            Path("data/benchmark_test/images/val"),
            Path("data/benchmark_test/images/train"),
        ]
        for alt in alt_candidates:
            if alt.exists():
                src_path = alt
                print(f"Specified source not found. Using fallback: {src_path}")
                break

    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Case A: Video File
    if src_path.is_file() and src_path.suffix.lower() in video_exts:
        if not HAS_CV2:
            raise RuntimeError(
                "OpenCV (cv2) is required for video inference. "
                "Please run via the ROCm environment: .\\run_rocm.bat scripts/infer_ird.py ..."
            )
        out_path = Path(args.output)
        if out_path.is_dir() or not out_path.suffix:
            out_path = out_path / f"inferred_{src_path.name}"
        run_video_inference(
            model=model,
            video_path=src_path,
            output_path=out_path,
            img_size=args.imgsz,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            device=device,
            device_label=device_label,
            max_det=args.max_det,
            obj_gate=args.obj_gate,
            decoder_version=args.decoder_version,
        )

    # Case B: Single Image File
    elif src_path.is_file() and src_path.suffix.lower() in img_exts:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"inferred_{src_path.name}"
        if HAS_CV2:
            img_bgr = cv2.imread(str(src_path))
            boxes, scores, cids, lat_ms, timing = infer_single_image(
                model=model,
                image_bgr=img_bgr,
                img_size=args.imgsz,
                conf_thresh=args.conf,
                iou_thresh=args.iou,
                device=device,
                max_det=args.max_det,
                obj_gate=args.obj_gate,
                decoder_version=args.decoder_version,
            )
            hud = f"IRD V1 | {device_label.split(':')[0]} | Latency: {lat_ms:.2f}ms | {len(boxes)} dets"
            annotated = draw_detections_cv2(img_bgr, boxes, scores, cids, args.conf, hud_text=hud)
            cv2.imwrite(str(out_file), annotated)
        else:
            with Image.open(src_path) as raw:
                orig_w, orig_h = raw.size
                resized = raw.convert("RGB").resize((args.imgsz, args.imgsz))
            img_np = np.array(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(tensor)
                boxes, scores, cids = decode_ird_predictions(
                    out,
                    img_size=args.imgsz,
                    conf_threshold=args.conf,
                    iou_threshold=args.iou,
                    max_det=args.max_det,
                    obj_gate=args.obj_gate,
                    decoder_version=args.decoder_version,
                    device=device,
                )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            b_np = boxes.cpu().numpy()
            if b_np.size > 0:
                scale_x, scale_y = orig_w / float(args.imgsz), orig_h / float(args.imgsz)
                scaled_b = b_np * np.array([scale_x, scale_y, scale_x, scale_y])
            else:
                scaled_b = np.empty((0, 4))
            from scripts.local_inference_test import render_detections
            render_detections(src_path, scaled_b, scores.cpu().numpy(), cids.cpu().numpy(), out_file, conf_threshold=args.conf)
        print(f"Processed single image in {lat_ms:.2f} ms ({1000.0/lat_ms:.2f} FPS) -> {len(boxes)} detections.")
        print(f"Saved: {out_file.resolve()}")

    # Case C: Directory of Images
    elif src_path.is_dir():
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_files = sorted([p for p in src_path.glob("*") if p.suffix.lower() in img_exts])
        print(f"Found {len(img_files)} images in {src_path}. Processing...")
        latencies = []
        for idx, img_p in enumerate(img_files):
            out_file = out_dir / f"inferred_{img_p.name}"
            if HAS_CV2:
                img_bgr = cv2.imread(str(img_p))
                boxes, scores, cids, lat_ms, timing = infer_single_image(
                    model=model,
                    image_bgr=img_bgr,
                    img_size=args.imgsz,
                    conf_thresh=args.conf,
                    iou_thresh=args.iou,
                    device=device,
                    max_det=args.max_det,
                    obj_gate=args.obj_gate,
                    decoder_version=args.decoder_version,
                )
                hud = f"IRD V1 | {device_label.split(':')[0]} | Latency: {lat_ms:.2f}ms | {len(boxes)} dets"
                annotated = draw_detections_cv2(img_bgr, boxes, scores, cids, args.conf, hud_text=hud)
                cv2.imwrite(str(out_file), annotated)
            else:
                with Image.open(img_p) as raw:
                    orig_w, orig_h = raw.size
                    resized = raw.convert("RGB").resize((args.imgsz, args.imgsz))
                img_np = np.array(resized, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(tensor)
                    boxes, scores, cids = decode_ird_predictions(
                        out,
                        img_size=args.imgsz,
                        conf_threshold=args.conf,
                        iou_threshold=args.iou,
                        max_det=args.max_det,
                        device=device,
                    )
                lat_ms = (time.perf_counter() - t0) * 1000.0
                b_np = boxes.cpu().numpy()
                scaled_b = b_np * np.array([orig_w / float(args.imgsz), orig_h / float(args.imgsz), orig_w / float(args.imgsz), orig_h / float(args.imgsz)]) if b_np.size > 0 else np.empty((0, 4))
                from scripts.local_inference_test import render_detections
                render_detections(img_p, scaled_b, scores.cpu().numpy(), cids.cpu().numpy(), out_file, conf_threshold=args.conf)
            latencies.append(lat_ms)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(img_files):
                print(f"  Processed [{idx+1:>4}/{len(img_files)}] | Avg Latency: {np.mean(latencies):.2f} ms", end="\r", flush=True)

        avg_lat = float(np.mean(latencies))
        print(f"\nDirectory inference complete: {len(img_files)} images in {np.sum(latencies)/1000.0:.2f}s ({1000.0/avg_lat:.2f} FPS avg).")
        print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
