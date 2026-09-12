# IRD — IndianRoadDetector (V1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Custom%20PyTorch-green.svg)](src/models/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, custom object detection architecture specifically engineered for the unique complexities of Indian road environments: high-density traffic, severe occlusions, diverse multi-scale agents, and heterogeneous vehicle compositions.

---

## 1. Project Overview

- **Official Model Name**: **IRD — IndianRoadDetection (V1)**
- **Baseline Comparison**: YOLOv8s baseline trained on 8,000 train / 2,000 validation images from the Indian Road Dataset achieved:
  - **mAP50**: 42.1%
  - **mAP50-95**: 33.1%
- **Architecture**: A pure PyTorch detector (~4.24M parameters) designed from first principles:
  - **Custom Backbone** (`IndianRoadBackbone`): Detail-preserving stem, dual-path anti-aliasing downsampling, multi-receptive blocks (MRB) with asymmetric strip convolutions and dilated context, and multi-scale context blocks (MSCB).
  - **Custom Multi-Scale Neck** (`IndianRoadNeck`): Adaptive scale fusion (ASF) with dynamic softmax gating, road context aggregator (RCA), and high-resolution detail enhancer for small objects.
  - **Custom Decoupled Head** (`IndianRoadHead`): Independent branches for bounding-box regression, objectness, and 12-class classification.
  - **Custom Training Loss** (`IndianRoadLoss`): Numerically stable CIoU localization, focal objectness, focal classification, and dynamic spatial matching (`MultiScaleSpatialMatcher`).
  - **100% Ultralytics-Free Model**: Zero dependencies on Ultralytics in the model architecture, head, or loss definitions.

---

## 2. Benchmark Dataset & Clip-Disjoint Splitting

### Dataset Source
- Hugging Face repository: [`thirdeyelabs/indian-road-dataset`](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset)
- Annotation standard: BDD100K-style JSON annotations where `json["name"]` has the structure `"<clip_id>/<frame_id>.jpg"`.

### Exact 12 Benchmark Classes (YOLOv8 & IRD V1)
To ensure a strictly apples-to-apples comparison against the YOLOv8s baseline, the class ID mapping matches the baseline ordering identically:

| Class ID | Class Name | Description |
|:---:|:---|:---|
| **0** | `person` | Pedestrians walking, standing, or crossing |
| **1** | `rider` | Drivers/passengers mounted on two-wheelers |
| **2** | `car` | Sedans, hatchbacks, SUVs |
| **3** | `truck` | Heavy freight, cargo vehicles, lorries |
| **4** | `bus` | Public transit, private, and interstate buses |
| **5** | `motorcycle` | Two-wheel motorized vehicles, scooters |
| **6** | `bicycle` | Non-motorized cycles |
| **7** | `autorickshaw` | Three-wheeled auto-rickshaws |
| **8** | `animal` | Stray cows, dogs, and working animals |
| **9** | `vehicle fallback` | Tractors, carts, and unclassified vehicles |
| **10** | `traffic light` | Overhead and post-mounted signals |
| **11** | `traffic sign` | Informational and regulatory road signs |

### Clip-Level Deterministic Split (Zero Clip Leakage)
In video-based road datasets, 30–60 consecutive frames belong to a single continuous video clip recorded under identical lighting, camera positioning, and road backdrop. Splitting by individual frames causes **clip leakage**—where near-identical frames from the same clip leak across train and validation splits, artificially inflating validation metrics.

**Guarantees of our pipeline:**
1. **Atomic Clip Assignment**: Every single frame belonging to a clip is assigned entirely to either `train` or `val`. A clip **never** appears in both splits.
2. **Deterministic SHA-256 Hashing**: Clip assignment uses a deterministic hash:
   $$\text{split} = \begin{cases} \text{train} & \text{if } \frac{\text{int}(\text{SHA256}(\text{seed} \parallel \text{clip\_id})[:8], 16)}{2^{32} - 1} < \text{split\_ratio} \\ \text{val} & \text{otherwise} \end{cases}$$
   This makes the split 100% reproducible across machines and independent of Python hash randomization.
3. **Clip Integrity Over Arbitrary Frame Counts**: When approaching the ~10,000-frame benchmark target (approx 8,000 train / 2,000 validation), the streaming pipeline completes the entire active video clip before terminating. No video clip is ever cut in half.
4. **RAM-Safe Streaming**: Direct streaming from Hugging Face via `IterableDataset` with immediate disk persistence (no memory exhaustion).

---

## 3. Dataset Conversion & Verification

### Running Dataset Conversion
Convert the Hugging Face dataset to YOLO format with deterministic clip-level splitting:
```bash
# Convert full benchmark dataset (~10,000 images, ~8,000 train / 2,000 val)
python src/data/convert_bdd_to_yolo.py \
    --output-dir data/indian_road_yolo \
    --target-total 10000 \
    --split-ratio 0.8 \
    --seed 42

# Or create a quick test subset (e.g. 100 images)
python src/data/convert_bdd_to_yolo.py \
    --output-dir data/test_yolo \
    --max-images 100 \
    --seed 42
```

The script generates:
```
data/indian_road_yolo/
├── images/
│   ├── train/     # <clip_id>__<frame>.jpg
│   └── val/       # <clip_id>__<frame>.jpg
├── labels/
│   ├── train/     # <clip_id>__<frame>.txt
│   └── val/       # <clip_id>__<frame>.txt
└── data.yaml      # YOLO dataset descriptor with 12 benchmark classes
```

### Auditing & Verification
Run the verification audit tool to prove dataset integrity:
```bash
python src/data/verify_dataset.py --data-dir data/indian_road_yolo
```

The audit automatically verifies and proves:
- **Zero Clip Leakage**: Computes $\text{Clips}_{\text{train}} \cap \text{Clips}_{\text{val}} = \emptyset$.
- **Image-Label Parity**: Verifies 100% matching image and `.txt` label file pairs with zero missing or orphan files.
- **Class ID Validation**: Verifies all annotations strictly belong to the 12 benchmark classes ($0 \le \text{id} \le 11$).
- **Bounding Box Bounds**: Confirms all coordinates $cx, cy, w, h \in [0.0, 1.0]$ with valid non-zero dimensions.
- **Statistical Breakdown**: Outputs exact frame, clip, and per-class object counts.

---

## 4. Training IRD V1

Train the custom detector using the production pipeline:
```bash
python scripts/train_custom.py \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/ird_v1 \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 640 \
    --workers 4
```

For quick verification / smoke testing:
```bash
python scripts/train_custom.py --smoke-test
```

---

## 5. Evaluating IRD V1 (Benchmark Evaluator)

Evaluate an IRD checkpoint against the benchmark validation dataset:
```bash
# Run standard evaluation on validation dataset
python scripts/evaluate_ird.py \
    --weights experiments/custom_model/ird_v1/ird_best.pt \
    --data data/indian_road_yolo/data.yaml \
    --imgsz 640 \
    --conf 0.001 \
    --iou 0.65 \
    --device auto

# Run embedded evaluation unit tests (IoU, NMS, matching, AP)
python scripts/evaluate_ird.py --run-unit-tests
```

### Evaluator Features:
- **Pure PyTorch**: 100% independent of Ultralytics evaluation routines.
- **Multi-Scale Head Decoding**: Stride 8 ($80 \times 80$), Stride 16 ($40 \times 40$), Stride 32 ($20 \times 20$).
- **Class-Aware NMS**: Spatial separation per class with pure PyTorch suppression.
- **Full COCO-Style Metrics**: Precision, Recall, mAP@0.50, and 10-threshold mAP@0.50:0.95 ($0.50:0.05:0.95$) with 101-point interpolated AP.
- **Latency & FPS Benchmarking**: Real-world timing per image.
- **Machine-Readable Export**: Automatically saves structured results to `experiments/custom_model/ird_evaluation.json`.

---

## 6. Model Architecture & Specifications

| Component | Class | Parameters | Spatial Outputs | Primary Role |
|---|---|---|---|---|
| **Backbone** | `IndianRoadBackbone` | ~3.43M | P3 (80x80), P4 (40x40), P5 (20x20) | Detail-preserving stem, asymmetric MRB strip convs, MSCB context |
| **Neck** | `IndianRoadNeck` | ~0.52M | N3 (80x80), N4 (40x40), N5 (20x20) | Dynamic softmax scale fusion (ASF), road context aggregator (RCA) |
| **Head** | `IndianRoadHead` | ~0.29M | Raw Decoupled (Box: 4, Obj: 1, Cls: 12) | Independent classification, box regression, spatial detail preserver |
| **Complete IRD** | `IndianRoadDetector` | **~4.24M** | Structured `HeadOutput` | End-to-end custom detection for Indian road conditions |
