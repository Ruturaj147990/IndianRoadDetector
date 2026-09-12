"""
Verification and Audit Script for Indian Road YOLO Dataset.

Proves:
1. No clip occurs in both train and validation splits (Zero Clip Leakage).
2. Image and label counts match exactly (no orphans or missing pairs).
3. Every label uses one of the 12 official benchmark classes in exact order.
4. Bounding boxes are valid: cx, cy, w, h in [0.0, 1.0], w > 0, h > 0.
5. Prints explicit class mapping and final train/val/total sample and clip counts.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

# Exact YOLOv8 benchmark class order
BENCHMARK_CLASSES = [
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

NUM_CLASSES = len(BENCHMARK_CLASSES)


def print_class_mapping() -> None:
    """Print official benchmark class mapping."""
    print("=" * 65)
    print("Class mapping:")
    for idx, name in enumerate(BENCHMARK_CLASSES):
        print(f"  {idx:>2} {name}")
    print("=" * 65)


def extract_clip_id(filename: str) -> str:
    """Extract video clip ID from filename formatted as {clip_id}__{frame} or {clip_id}_{frame}."""
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__")[0]
    return stem.split("_")[0]


def verify_dataset(data_dir: Path) -> bool:
    """
    Rigorously audit the dataset and return True if all checks pass.
    """
    print_class_mapping()

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    data_yaml = data_dir / "data.yaml"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Missing 'images' or 'labels' in {data_dir}")
        return False

    splits = ["train", "val"]
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    split_images: Dict[str, List[Path]] = {}
    split_labels: Dict[str, List[Path]] = {}
    split_clips: Dict[str, Set[str]] = {}
    split_boxes_per_class: Dict[str, Counter] = {s: Counter() for s in splits}

    total_errors = 0

    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split

        if not img_split_dir.exists():
            print(f"ERROR: Missing directory {img_split_dir}")
            total_errors += 1
            continue

        imgs = sorted([p for p in img_split_dir.iterdir() if p.suffix.lower() in img_extensions])
        lbls = sorted(list(lbl_split_dir.glob("*.txt"))) if lbl_split_dir.exists() else []

        split_images[split] = imgs
        split_labels[split] = lbls
        split_clips[split] = set()

        img_stems = {p.stem: p for p in imgs}
        lbl_stems = {p.stem: p for p in lbls}

        # 1. Image and Label Parity Check
        missing_labels = img_stems.keys() - lbl_stems.keys()
        orphan_labels = lbl_stems.keys() - img_stems.keys()

        if missing_labels:
            print(f"ERROR [{split}]: {len(missing_labels)} images missing corresponding label file!")
            total_errors += len(missing_labels)
        if orphan_labels:
            print(f"ERROR [{split}]: {len(orphan_labels)} label files have no corresponding image!")
            total_errors += len(orphan_labels)

        # 2. Extract clip IDs
        for img_p in imgs:
            clip_id = extract_clip_id(img_p.name)
            split_clips[split].add(clip_id)

        # 3. Inspect label files for class IDs and box validity
        for lbl_p in lbls:
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"ERROR [{split}]: Malformed label line {line_idx+1} in {lbl_p.name}: '{line}'")
                    total_errors += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    print(f"ERROR [{split}]: Non-numeric values in {lbl_p.name}: '{line}'")
                    total_errors += 1
                    continue

                # Check class ID
                if not (0 <= cls_id < NUM_CLASSES):
                    print(f"ERROR [{split}]: Invalid class ID {cls_id} in {lbl_p.name}. Must be in [0, {NUM_CLASSES-1}].")
                    total_errors += 1
                else:
                    split_boxes_per_class[split][cls_id] += 1

                # Check box range
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    print(f"ERROR [{split}]: Center out of bounds (cx={cx}, cy={cy}) in {lbl_p.name}")
                    total_errors += 1
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    print(f"ERROR [{split}]: Width/Height out of bounds (w={w}, h={h}) in {lbl_p.name}")
                    total_errors += 1

    # 4. Clip Leakage Verification (CRITICAL)
    train_clips = split_clips.get("train", set())
    val_clips = split_clips.get("val", set())
    clip_intersection = train_clips.intersection(val_clips)

    print("\n--- Clip Leakage Analysis ---")
    if len(clip_intersection) > 0:
        print(f"FAILED: Clip leakage detected! {len(clip_intersection)} clip(s) present in BOTH train and val:")
        for c in sorted(list(clip_intersection))[:10]:
            print(f"  LEAKED CLIP: {c}")
        total_errors += len(clip_intersection)
    else:
        print(f"PASSED: Zero clip leakage detected. Train and Val splits are 100% clip-disjoint.")

    # 5. Check data.yaml exists
    if not data_yaml.exists():
        print(f"WARNING: data.yaml not found at {data_yaml}")
    else:
        print(f"PASSED: data.yaml found at {data_yaml}")

    # 6. Detailed Summary Audit Table
    train_count = len(split_images.get("train", []))
    val_count = len(split_images.get("val", []))
    total_images = train_count + val_count

    print("\n" + "=" * 65)
    print("DATASET VERIFICATION AUDIT SUMMARY")
    print("=" * 65)
    print(f"Images:  Train = {train_count:>5} | Val = {val_count:>5} | Total = {total_images:>5}")
    print(f"Labels:  Train = {len(split_labels.get('train', [])):>5} | Val = {len(split_labels.get('val', [])):>5}")
    print(f"Clips:   Train = {len(train_clips):>5} | Val = {len(val_clips):>5} | Total = {len(train_clips | val_clips):>5}")
    print(f"Overlapping Clips (Leakage): {len(clip_intersection)}")
    print("-" * 65)
    print("Class Distribution Across Splits:")
    print(f"{'ID':>3}  {'Class Name':<18}  {'Train Boxes':>12}  {'Val Boxes':>10}  {'Total':>10}")
    print("-" * 65)

    total_all_boxes = 0
    for cls_id in range(NUM_CLASSES):
        cls_name = BENCHMARK_CLASSES[cls_id]
        train_boxes = split_boxes_per_class["train"][cls_id]
        val_boxes = split_boxes_per_class["val"][cls_id]
        cls_total = train_boxes + val_boxes
        total_all_boxes += cls_total
        print(f"{cls_id:>3}  {cls_name:<18}  {train_boxes:>12}  {val_boxes:>10}  {cls_total:>10}")

    print("-" * 65)
    print(f"Total Bounding Boxes: {total_all_boxes}")
    print("=" * 65)

    if total_errors == 0:
        print("\nALL CHECKS PASSED: Dataset is verified, valid, and clip-disjoint.")
        return True
    else:
        print(f"\nAUDIT FAILED: Found {total_errors} error(s).")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Indian Road Dataset integrity and clip disjointness")
    parser.add_argument("--data-dir", type=str, default="data/indian_road_yolo",
                        help="Path to YOLO dataset directory (containing images/ and labels/)")
    args = parser.parse_args()

    success = verify_dataset(Path(args.data_dir))
    sys.exit(0 if success else 1)
