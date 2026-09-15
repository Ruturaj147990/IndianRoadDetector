"""
Automated Batch Size Calibration & Throughput Profiler for AMD Radeon RX 7700 XT 12GB.

Benchmarks IRD V2 forward, backward, and optimization throughput across candidate
batch sizes (4, 8, 12, 16) to select the configuration that maximizes sustained
images/second while leaving safe VRAM headroom (>= 2.0 GB).
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector
from src.models.losses.task_aligned_loss import TaskAlignedLoss


def benchmark_batch_size(
    model: nn.Module,
    loss_fn: nn.Module,
    batch_size: int,
    device: torch.device,
    img_size: int = 640,
    warmup_steps: int = 3,
    eval_steps: int = 15,
) -> Dict[str, float]:
    """Measures latency, throughput, and peak VRAM for a specific batch size under AMP."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Synthetic image [B, 3, 640, 640]
    # Targets: [B, 6] -> [batch_idx, class_id, cx, cy, w, h]
    dummy_img = torch.randn(batch_size, 3, img_size, img_size, device=device)
    dummy_targets = torch.tensor(
        [[i, 0, 0.5, 0.5, 0.2, 0.2] for i in range(batch_size)],
        device=device,
        dtype=torch.float32,
    )

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            preds = model(dummy_img)
            loss_res = loss_fn(preds, dummy_targets)
        scaler.scale(loss_res.total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    # Timed benchmark loop
    fwd_times: List[float] = []
    bwd_times: List[float] = []
    iter_times: List[float] = []

    for _ in range(eval_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda"):
            preds = model(dummy_img)
            loss_res = loss_fn(preds, dummy_targets)
        torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()

        t_bwd_start = time.perf_counter()
        scaler.scale(loss_res.total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()

        iter_times.append(time.perf_counter() - t0)
        fwd_times.append(t_fwd_end - t_fwd_start)
        bwd_times.append(t_bwd_end - t_bwd_start)

    avg_iter_s = float(sum(iter_times) / len(iter_times))
    avg_fwd_ms = float((sum(fwd_times) / len(fwd_times)) * 1000.0)
    avg_bwd_ms = float((sum(bwd_times) / len(bwd_times)) * 1000.0)
    img_per_sec = float(batch_size / avg_iter_s)
    batches_per_sec = float(1.0 / avg_iter_s)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    peak_vram_gb = float(peak_bytes / (1024**3))
    total_vram_gb = float(total_bytes / (1024**3))
    headroom_gb = float((total_bytes - peak_bytes) / (1024**3))

    return {
        "batch_size": batch_size,
        "img_per_sec": round(img_per_sec, 1),
        "batches_per_sec": round(batches_per_sec, 2),
        "avg_iter_ms": round(avg_iter_s * 1000.0, 1),
        "avg_fwd_ms": round(avg_fwd_ms, 1),
        "avg_bwd_ms": round(avg_bwd_ms, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "total_vram_gb": round(total_vram_gb, 2),
        "headroom_gb": round(headroom_gb, 2),
        "is_safe": bool(headroom_gb >= 2.0),
    }


def run_calibration(
    candidates: List[int] = [4, 8, 12, 16],
    img_size: int = 640,
    output_path: Optional[str] = None,
) -> int:
    """Sweeps candidate batch sizes and returns the optimal batch size."""
    if not torch.cuda.is_available():
        print("[Calibration] No CUDA/ROCm device available. Falling back to batch size 4.")
        return 4

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 72)
    print(f"IRD V2 BATCH SIZE CALIBRATION & PROFILING ON: {gpu_name}")
    print("=" * 72)

    model = IndianRoadDetector().to(device)
    loss_fn = TaskAlignedLoss().to(device)

    results: List[Dict[str, float]] = []

    print(f"{'Batch':<8} | {'Throughput':<12} | {'Latency':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM':<10} | {'Headroom':<10} | {'Status'}")
    print("-" * 88)

    for bs in candidates:
        try:
            res = benchmark_batch_size(model, loss_fn, bs, device, img_size=img_size)
            results.append(res)
            status = "SAFE" if res["is_safe"] else "LOW HEADROOM"
            print(
                f"{res['batch_size']:<8} | "
                f"{res['img_per_sec']:>6.1f} img/s | "
                f"{res['avg_iter_ms']:>6.1f} ms | "
                f"{res['avg_fwd_ms']:>6.1f} ms | "
                f"{res['avg_bwd_ms']:>6.1f} ms | "
                f"{res['peak_vram_gb']:>5.2f} GB | "
                f"{res['headroom_gb']:>5.2f} GB | "
                f"{status}"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:<8} | {'OOM':<12} | {'-':<10} | {'-':<10} | {'-':<10} | {'> 12 GB':<10} | {'0.00 GB':<10} | OOM")
            torch.cuda.empty_cache()
            break

    print("=" * 72)

    # Select best safe batch size based on img_per_sec
    safe_results = [r for r in results if r["is_safe"]]
    if not safe_results:
        # Fallback to smallest tested if none left 2GB
        best = min(results, key=lambda r: r["batch_size"])
    else:
        best = max(safe_results, key=lambda r: r["img_per_sec"])

    selected_bs = int(best["batch_size"])
    print(f"--> RECOMMENDED OPTIMAL BATCH SIZE: {selected_bs} ({best['img_per_sec']} img/s, Peak VRAM: {best['peak_vram_gb']} GB, Headroom: {best['headroom_gb']} GB)")
    print("=" * 72)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump({"gpu": gpu_name, "best_batch_size": selected_bs, "benchmark": results}, f, indent=2)

    return selected_bs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate IRD V2 batch size on GPU.")
    parser.add_argument("--candidates", nargs="+", type=int, default=[4, 8, 12, 16], help="Batch size candidates")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    parser.add_argument("--output", type=str, default="experiments/custom_model/v2_full_646k/calibration_results.json")
    args = parser.parse_args()

    run_calibration(candidates=args.candidates, img_size=args.img_size, output_path=args.output)
