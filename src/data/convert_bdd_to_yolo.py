"""
Dataset Conversion Pipeline: BDD100K to YOLO Format for Indian Road Dataset.

Official Dataset Source: thirdeyelabs/indian-road-dataset (Hugging Face)
Target Benchmark: IRD V1 vs YOLOv8s (~10,000 images, approx 8,000 train / 2,000 val)

Key Guarantees:
1. Clip-Level Disjoint Splitting:
   - Split is performed strictly at the video clip level using deterministic SHA-256 hashing.
   - All frames of a clip belong entirely to either train or validation.
   - Zero clip leakage between train and validation splits.
2. Exact YOLOv8 Benchmark Class Order:
   0: person, 1: rider, 2: car, 3: truck, 4: bus, 5: motorcycle,
   6: bicycle, 7: autorickshaw, 8: animal, 9: vehicle fallback,
   10: traffic light, 11: traffic sign.
3. RAM-Safe Streaming:
   - Streams directly from Hugging Face via IterableDataset.
   - Writes images and labels directly to disk without loading the full dataset into RAM.
4. Clip Integrity over Exact Counts:
   - Completes active clips at boundaries before stopping.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Exact YOLOv8 benchmark class order
CLASS_NAMES: List[str] = [
    "person",           # 0
    "rider",            # 1
    "car",              # 2
    "truck",            # 3
    "bus",              # 4
    "motorcycle",       # 5
    "bicycle",          # 6
    "autorickshaw",     # 7
    "animal",           # 8
    "vehicle fallback", # 9
    "traffic light",    # 10
    "traffic sign",     # 11
]

CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def print_class_mapping() -> None:
    """Print the official benchmark class mapping."""
    print("=" * 60)
    print("Official Benchmark Class Mapping (YOLOv8 & IRD V1):")
    print("=" * 60)
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx:>2}: {name}")
    print("=" * 60)


def assign_clip_to_split(clip_id: str, split_ratio: float = 0.8, seed: int = 42) -> str:
    """
    Deterministically assign a clip to 'train' or 'val' using SHA-256.
    
    Guarantees:
    - Pure function of (seed, clip_id).
    - Platform and Python run-independent.
    - Zero clip leakage: all frames with this clip_id will map to the exact same split.
    """
    key = f"{seed}_{clip_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    val = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "train" if val < split_ratio else "val"


def convert_bdd_box_to_yolo(
    box2d: Dict[str, float],
    img_width: int,
    img_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert BDD100K box2d {x1, y1, x2, y2} to normalized YOLO {cx, cy, w, h}.
    
    Returns None if box coordinates are degenerate or outside image.
    """
    try:
        x1 = float(box2d["x1"])
        y1 = float(box2d["y1"])
        x2 = float(box2d["x2"])
        y2 = float(box2d["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    # Order coordinates
    x_min = max(0.0, min(float(img_width), min(x1, x2)))
    x_max = max(0.0, min(float(img_width), max(x1, x2)))
    y_min = max(0.0, min(float(img_height), min(y1, y2)))
    y_max = max(0.0, min(float(img_height), max(y1, y2)))

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Reject degenerate or sub-pixel boxes
    if box_w <= 1.0 or box_h <= 1.0:
        return None

    cx = (x_min + x_max) / (2.0 * float(img_width))
    cy = (y_min + y_max) / (2.0 * float(img_height))
    w = box_w / float(img_width)
    h = box_h / float(img_height)

    # Strictly clamp to [0.0, 1.0]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def write_data_yaml(output_dir: Path) -> Path:
    """Generate YOLO data.yaml configuration file."""
    yaml_path = output_dir / "data.yaml"
    abs_path = output_dir.resolve().as_posix()

    names_block = "\n".join([f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES)])
    yaml_content = f"""# Indian Road Dataset (IRD V1 & YOLOv8 Benchmark)
# Deterministic clip-disjoint split
path: {abs_path}
train: images/train
val: images/val

names:
{names_block}

nc: {len(CLASS_NAMES)}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return yaml_path


def run_conversion(
    dataset_name: str = "thirdeyelabs/indian-road-dataset",
    output_dir: str = "data/indian_road_yolo",
    target_total: int = 10000,
    split_ratio: float = 0.8,
    seed: int = 42,
    max_images: Optional[int] = None,
) -> Dict[str, any]:
    """
    Stream and convert BDD100K-style Hugging Face dataset to YOLO format.
    
    Ensures clip-level integrity: streaming stops only at clip boundaries
    once the target count is satisfied.
    """
    from datasets import load_dataset
    from PIL import Image

    out_path = Path(output_dir)
    img_train_dir = out_path / "images" / "train"
    img_val_dir = out_path / "images" / "val"
    lbl_train_dir = out_path / "labels" / "train"
    lbl_val_dir = out_path / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print_class_mapping()

    limit = max_images if max_images is not None else target_total
    print(f"\nStarting streaming conversion from Hugging Face '{dataset_name}'...")
    print(f"Target count: ~{limit} images (approx {split_ratio*100:.0f}% train / {(1-split_ratio)*100:.0f}% val)")
    print(f"Deterministic seed: {seed}")
    print(f"Output directory: {out_path.resolve()}\n")

    # Load streaming dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)

    clip_assignments: Dict[str, str] = {}
    clip_frame_counts: Dict[str, int] = {}
    split_counts = {"train": 0, "val": 0}
    total_written = 0
    total_boxes = 0
    current_clip_id: Optional[str] = None
    start_time = time.time()

    for sample_idx, sample in enumerate(ds):
        json_data = sample.get("json", {})
        name = json_data.get("name", "")
        name = name.replace("\\", "/")

        parts = name.split("/")
        if len(parts) >= 2:
            clip_id = parts[0]
            frame_filename = parts[-1]
        else:
            clip_id = "default_clip"
            frame_filename = name if name else f"frame_{sample_idx:06d}.jpg"

        # Check for clip transition
        if clip_id != current_clip_id:
            # Stop if target reached and at a clean clip boundary
            if total_written >= limit:
                print(f"\nTarget count reached ({total_written} >= {limit}). Stopping at clip boundary.")
                break

            current_clip_id = clip_id
            if clip_id not in clip_assignments:
                clip_assignments[clip_id] = assign_clip_to_split(clip_id, split_ratio, seed)
                clip_frame_counts[clip_id] = 0

        split = clip_assignments[clip_id]
        img_dest_dir = img_train_dir if split == "train" else img_val_dir
        lbl_dest_dir = lbl_train_dir if split == "train" else lbl_val_dir

        # Image handling
        img: Image.Image = sample.get("jpg")
        if img is None:
            continue

        img_w, img_h = img.size
        frame_stem = Path(frame_filename).stem
        unique_stem = f"{clip_id}__{frame_stem}"

        img_file_path = img_dest_dir / f"{unique_stem}.jpg"
        lbl_file_path = lbl_dest_dir / f"{unique_stem}.txt"

        # Save image (RAM safe: immediate write)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_file_path, quality=95)

        # Convert bounding boxes
        labels = json_data.get("labels", [])
        yolo_lines: List[str] = []

        for lbl in labels:
            cat_name = lbl.get("category")
            if cat_name not in CLASS_TO_ID:
                continue

            class_id = CLASS_TO_ID[cat_name]
            box2d = lbl.get("box2d")
            if not box2d:
                continue

            box_norm = convert_bdd_box_to_yolo(box2d, img_w, img_h)
            if box_norm is None:
                continue

            cx, cy, w, h = box_norm
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        # Save YOLO labels (.txt)
        with open(lbl_file_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        total_written += 1
        split_counts[split] += 1
        clip_frame_counts[clip_id] += 1

        if total_written % 20 == 0 or total_written == limit:
            elapsed = time.time() - start_time
            rate = total_written / max(0.1, elapsed)
            print(
                f"Progress: {total_written:>5} images written | "
                f"Train: {split_counts['train']:>5} | Val: {split_counts['val']:>5} | "
                f"Clips: {len(clip_assignments)} | {rate:.1f} img/s",
                end="\r",
                flush=True,
            )

    yaml_path = write_data_yaml(out_path)
    elapsed = time.time() - start_time

    train_clips = sum(1 for c, s in clip_assignments.items() if s == "train")
    val_clips = sum(1 for c, s in clip_assignments.items() if s == "val")

    print("\n" + "=" * 60)
    print("Dataset Conversion Summary:")
    print("=" * 60)
    print(f"Total Images Written:   {total_written}")
    print(f"Train Images:           {split_counts['train']} ({split_counts['train']/max(1, total_written)*100:.1f}%)")
    print(f"Val Images:             {split_counts['val']} ({split_counts['val']/max(1, total_written)*100:.1f}%)")
    print(f"Total Unique Clips:     {len(clip_assignments)}")
    print(f"Train Clips:            {train_clips}")
    print(f"Val Clips:              {val_clips}")
    print(f"Total Bounding Boxes:   {total_boxes}")
    print(f"YAML Configuration:     {yaml_path}")
    print(f"Elapsed Time:           {elapsed:.1f}s")
    print("=" * 60)

    return {
        "total_images": total_written,
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "total_clips": len(clip_assignments),
        "train_clips": train_clips,
        "val_clips": val_clips,
        "total_boxes": total_boxes,
        "data_yaml": str(yaml_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face Indian Road Dataset to YOLO format")
    parser.add_argument("--dataset-name", type=str, default="thirdeyelabs/indian-road-dataset",
                        help="Hugging Face dataset identifier")
    parser.add_argument("--output-dir", type=str, default="data/indian_road_yolo",
                        help="Destination directory for YOLO dataset")
    parser.add_argument("--target-total", type=int, default=10000,
                        help="Target total images across train + val (default: 10000)")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Ratio of clips to allocate to train split (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic random seed for clip assignment (default: 42)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional override to stop after N images (for testing/subsets)")

    args = parser.parse_args()

    run_conversion(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        target_total=args.target_total,
        split_ratio=args.split_ratio,
        seed=args.seed,
        max_images=args.max_images,
    )
