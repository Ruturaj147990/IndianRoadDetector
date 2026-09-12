"""
Official Baseline Training Script: Ultralytics YOLOv8s on Indian Road Dataset.

Used exclusively for fair, apples-to-apples benchmarking against IRD V1 on the
exact same clip-disjoint dataset split (data/indian_road_yolo/data.yaml).

Reports:
- Trainable parameters
- Precision, Recall
- mAP50, mAP50:95
- Per-class AP
- Model inference latency & FPS
"""

import argparse
import json
import time
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics is required for baseline comparison. Run: pip install ultralytics --no-deps")


def run_yolov8s_baseline(
    data_yaml: str = "data/indian_road_yolo/data.yaml",
    epochs: int = 10,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    output_dir: str = "experiments/yolov8s_baseline",
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YOLOv8s Official Fair Baseline Training Run")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Image Size:   {imgsz}")
    print(f"Device:       {device}")
    print("=" * 70)

    # Initialize YOLOv8s from standard pretrained weights
    model = YOLO("yolov8s.pt")

    # Train
    start_time = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(out_path),
        name="yolov8s_run",
        exist_ok=True,
        plots=False,
        save=True,
        verbose=True,
        workers=0,
    )
    total_train_time = time.time() - start_time

    # Validate
    val_results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        workers=0,
    )

    # Parameter count
    param_count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    # Extract per-class AP if available
    per_class_ap50 = {}
    per_class_ap50_95 = {}
    try:
        class_names = [model.names[i] for i in range(len(model.names))]
        if hasattr(val_results.box, "ap50") and val_results.box.ap50 is not None:
            for i, name in enumerate(class_names):
                if i < len(val_results.box.ap50):
                    per_class_ap50[name] = float(val_results.box.ap50[i])
        if hasattr(val_results.box, "maps") and val_results.box.maps is not None:
            for i, name in enumerate(class_names):
                if i < len(val_results.box.maps):
                    per_class_ap50_95[name] = float(val_results.box.maps[i])
    except Exception as e:
        print(f"Warning extracting per-class metrics: {e}")

    # Peak VRAM
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    # Benchmark Latency & FPS
    dummy = torch.randn(1, 3, imgsz, imgsz).to(next(model.model.parameters()).device)
    for _ in range(10):
        _ = model(dummy)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = ((time.time() - t0) / iterations) * 1000.0
    fps = 1000.0 / latency_ms

    summary = {
        "model": "YOLOv8s",
        "parameters": param_count,
        "epochs": epochs,
        "imgsz": imgsz,
        "train_time_sec": round(total_train_time, 2),
        "precision": float(val_results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap50_95,
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 2),
        "vram_mb": vram_mb,
    }

    with open(out_path / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"YOLOv8s Training Completed in {total_train_time/60:.2f} min")
    print(f"Parameters: {param_count:,}")
    print(f"mAP50:      {summary['mAP50']:.4f}")
    print(f"mAP50:95:   {summary['mAP50_95']:.4f}")
    print(f"Latency:    {summary['latency_ms']} ms ({summary['fps']} FPS)")
    print(f"VRAM:       {summary['vram_mb']} MB")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8s Baseline for IRD Benchmark Comparison")
    parser.add_argument("--data-yaml", type=str, default="data/indian_road_yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="experiments/yolov8s_baseline")
    args = parser.parse_args()

    run_yolov8s_baseline(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
    )
