import os
os.environ["MIOPEN_FIND_MODE"] = "2"

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector
from src.models.losses.custom_loss import IndianRoadLoss
from scripts.train_custom import YOLODetectionDataset, ird_collate_fn, resolve_dataset_dirs


def test_batch_size(batch_size: int, data_dir: str = "data/indian_road_yolo", num_iters: int = 10):
    device = torch.device("cuda:0")
    print(f"\n" + "=" * 70, flush=True)
    print(f" Testing Batch Size = {batch_size} on {torch.cuda.get_device_name(0)} ", flush=True)
    print("=" * 70, flush=True)

    # Empty cache before test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    train_img_dir, train_lbl_dir, _, _ = resolve_dataset_dirs(Path(data_dir))
    dataset = YOLODetectionDataset(
        img_dir=train_img_dir,
        lbl_dir=train_lbl_dir,
        img_size=(640, 640),
        max_samples=batch_size * (num_iters + 2),
        augment=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ird_collate_fn,
        pin_memory=True,
    )

    # Instantiate IRD V1.5 model & loss
    model = IndianRoadDetector(
        num_classes=12,
        use_atd=True,
        use_ssdp=True,
        use_fgbr=True,
        use_quality=True,
        use_cdg=True,
    ).to(device)
    model.train()

    loss_fn = IndianRoadLoss(
        num_classes=12,
        box_weight=5.0,
        obj_weight=1.0,
        cls_weight=1.0,
        matcher_version="topk_adaptive_v2",
        class_balanced_loss=True,
        use_aux_one2one=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    data_iter = iter(dataloader)

    # Warmup 2 iterations
    for _ in range(2):
        images, targets = next(data_iter)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss_res = loss_fn(preds, targets, img_size=(640, 640))
        scaler.scale(loss_res.total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Timed iterations
    t0 = time.perf_counter()
    for i in range(num_iters):
        images, targets = next(data_iter)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss_res = loss_fn(preds, targets, img_size=(640, 640))
        scaler.scale(loss_res.total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)
    total_time = time.perf_counter() - t0

    allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    total_gpu_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    headroom_mb = total_gpu_mb - max(peak_mb, reserved_mb)

    images_processed = batch_size * num_iters
    img_per_sec = images_processed / total_time
    ms_per_batch = (total_time / num_iters) * 1000.0

    print(f"  Allocated VRAM:     {allocated_mb:.1f} MB ({allocated_mb/1024:.2f} GB)", flush=True)
    print(f"  Reserved VRAM:      {reserved_mb:.1f} MB ({reserved_mb/1024:.2f} GB)", flush=True)
    print(f"  Peak VRAM:          {peak_mb:.1f} MB ({peak_mb/1024:.2f} GB)", flush=True)
    print(f"  Headroom to 12GB:   {headroom_mb:.1f} MB ({headroom_mb/1024:.2f} GB)", flush=True)
    print(f"  Throughput:         {img_per_sec:.2f} img/s ({ms_per_batch:.1f} ms/batch)", flush=True)

    del model, optimizer, scaler, dataloader, dataset
    torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "peak_mb": peak_mb,
        "reserved_mb": reserved_mb,
        "headroom_mb": headroom_mb,
        "img_per_sec": img_per_sec,
        "ms_per_batch": ms_per_batch,
        "success": True,
    }


if __name__ == "__main__":
    candidates = [4, 8, 12, 16, 20, 24]
    results = []
    for bs in candidates:
        try:
            res = test_batch_size(bs, num_iters=8)
            results.append(res)
            # If headroom drops below 1200 MB (~1.2 GB), stop increasing
            if res["headroom_mb"] < 1200:
                print(f"\nStopping calibration: Headroom at batch {bs} is {res['headroom_mb']:.1f} MB (< 1.2 GB).", flush=True)
                break
        except torch.cuda.OutOfMemoryError:
            print(f"\nCUDA OutOfMemoryError encountered at batch size {bs}!", flush=True)
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"\nError at batch size {bs}: {e}", flush=True)
            break

    print("\n" + "=" * 70, flush=True)
    print(f"{'BATCH SIZE CALIBRATION SUMMARY':^70}", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Batch Size':<12}{'Peak VRAM (GB)':<16}{'Headroom (GB)':<16}{'Speed (img/s)':<16}{'Status':<10}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(f"{r['batch_size']:<12}{r['peak_mb']/1024:<16.2f}{r['headroom_mb']/1024:<16.2f}{r['img_per_sec']:<16.2f}{'SAFE':<10}", flush=True)
    print("=" * 70, flush=True)
