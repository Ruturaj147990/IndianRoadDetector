"""
Phase 1 Dataset Structure Audit for IRD V1
Analyzes:
- Class frequency and bounding box counts
- Objects per image & multi-object distributions
- Specific breakdowns for cars, motorcycles, riders, persons
- Bounding box scale/size distribution (tiny: <32x32, small: 32-96, medium: 96-256, large: >256)
- Aspect ratios
- Identifies diagnostic test subsets A through J
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

DATA_DIR = Path("data/indian_road_yolo")
TRAIN_LBL_DIR = DATA_DIR / "labels" / "train"
VAL_LBL_DIR = DATA_DIR / "labels" / "val"
TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
VAL_IMG_DIR = DATA_DIR / "images" / "val"

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

def audit_split(lbl_dir: Path, img_dir: Path, split_name: str):
    label_files = sorted(list(lbl_dir.glob("*.txt")))
    total_images = len(label_files)
    total_boxes = 0
    class_counts = Counter()
    boxes_per_image = []
    
    # Specific class per-image counters
    cars_per_image = []
    motorcycles_per_image = []
    riders_per_image = []
    persons_per_image = []
    
    # Size distributions (assuming 640x640 base)
    size_counts = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    aspect_ratios = []
    
    # Subset definitions
    subsets = defaultdict(list)
    
    for lf in label_files:
        img_name = f"{lf.stem}.jpg"
        img_path = img_dir / img_name
        
        with open(lf, "r") as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]
            
        n_obj = len(lines)
        boxes_per_image.append(n_obj)
        total_boxes += n_obj
        
        img_classes = []
        c_count = 0
        m_count = 0
        r_count = 0
        p_count = 0
        has_tiny = False
        
        for p in lines:
            cid = int(p[0])
            cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            cname = BENCHMARK_CLASSES[cid]
            class_counts[cname] += 1
            img_classes.append(cname)
            
            # Absolute pixel dims at 640x640
            px_w = w * 640.0
            px_h = h * 640.0
            area = px_w * px_h
            sqrt_area = np.sqrt(max(0, area))
            
            if sqrt_area < 32:
                size_counts["tiny"] += 1
                has_tiny = True
            elif sqrt_area < 96:
                size_counts["small"] += 1
            elif sqrt_area < 256:
                size_counts["medium"] += 1
            else:
                size_counts["large"] += 1
                
            if px_h > 0:
                aspect_ratios.append(px_w / px_h)
                
            if cname == "car":
                c_count += 1
            elif cname == "motorcycle":
                m_count += 1
            elif cname == "rider":
                r_count += 1
            elif cname == "person":
                p_count += 1

        cars_per_image.append(c_count)
        motorcycles_per_image.append(m_count)
        riders_per_image.append(r_count)
        persons_per_image.append(p_count)
        
        # Classify into subsets for validation
        if c_count >= 3:
            subsets["A_multi_car"].append(img_name)
        if m_count >= 2:
            subsets["B_multi_motorcycle"].append(img_name)
        if c_count >= 1 and m_count >= 1:
            subsets["C_car_plus_motorcycle"].append(img_name)
        if m_count >= 1 and r_count >= 1:
            subsets["D_motorcycle_plus_rider"].append(img_name)
        if n_obj >= 8:
            subsets["E_dense_traffic"].append(img_name)
        if has_tiny:
            subsets["F_small_objects"].append(img_name)
        if any(ar > 1.8 for ar in [float(p[3])/max(float(p[4]), 1e-4) for p in lines]):
            subsets["G_side_view"].append(img_name)
        if any(1.2 < ar <= 1.8 for ar in [float(p[3])/max(float(p[4]), 1e-4) for p in lines]):
            subsets["H_diagonal_view"].append(img_name)
            
    summary = {
        "split": split_name,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(float(np.mean(boxes_per_image)), 2),
        "max_boxes_per_image": int(np.max(boxes_per_image)) if boxes_per_image else 0,
        "multi_object_images_count": sum(1 for b in boxes_per_image if b > 1),
        "multi_object_ratio": round(sum(1 for b in boxes_per_image if b > 1) / max(total_images, 1), 3),
        "class_counts": dict(class_counts),
        "size_distribution": size_counts,
        "cars_stats": {
            "avg_per_image": round(float(np.mean(cars_per_image)), 2),
            "images_with_car": sum(1 for c in cars_per_image if c > 0),
            "multi_car_images": sum(1 for c in cars_per_image if c >= 3)
        },
        "motorcycles_stats": {
            "avg_per_image": round(float(np.mean(motorcycles_per_image)), 2),
            "images_with_motorcycle": sum(1 for m in motorcycles_per_image if m > 0),
            "multi_motorcycle_images": sum(1 for m in motorcycles_per_image if m >= 2)
        },
        "riders_stats": {
            "avg_per_image": round(float(np.mean(riders_per_image)), 2),
            "images_with_rider": sum(1 for r in riders_per_image if r > 0),
        },
        "subset_counts": {k: len(v) for k, v in subsets.items()}
    }
    return summary, subsets

if __name__ == "__main__":
    print("Running comprehensive dataset audit on train & val splits...")
    train_summary, _ = audit_split(TRAIN_LBL_DIR, TRAIN_IMG_DIR, "train")
    val_summary, val_subsets = audit_split(VAL_LBL_DIR, VAL_IMG_DIR, "val")
    
    out_file = Path("experiments/custom_model/dataset_audit.json")
    with open(out_file, "w") as f:
        json.dump({
            "train": train_summary,
            "val": val_summary
        }, f, indent=2)
        
    subsets_file = Path("experiments/custom_model/val_diagnostic_subsets.json")
    with open(subsets_file, "w") as f:
        json.dump(val_subsets, f, indent=2)
        
    print(f"Dataset audit saved to {out_file}")
    print(f"Validation diagnostic subsets saved to {subsets_file}")
    print("\n--- Summary Highlights ---")
    print(f"Train images: {train_summary['total_images']}, Total boxes: {train_summary['total_boxes']}")
    print(f"Val images:   {val_summary['total_images']}, Total boxes: {val_summary['total_boxes']}")
    print(f"Val average objects/image: {val_summary['avg_boxes_per_image']}")
    print(f"Val class counts: {val_summary['class_counts']}")
    print(f"Val size distribution: {val_summary['size_distribution']}")
    print(f"Val diagnostic subset sizes: {val_summary['subset_counts']}")
