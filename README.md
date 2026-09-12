# IRD — IndianRoadDetector (V1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Custom%20PyTorch-green.svg)](src/models/ARCHITECTURE.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, custom object detection architecture specifically engineered for the unique complexities of Indian road environments: high-density traffic, severe occlusions, diverse multi-scale agents, and heterogeneous vehicle compositions.

---

## 1. Project Overview & Final Empirical Findings

- **Official Model Name**: **IRD — IndianRoadDetection (V1)**
- **Trainable Parameters**: **4,241,529** (~4.24M, exactly 62% smaller than YOLOv8s)
- **Computational Complexity**: **18.4 GFLOPs** at $640 \times 640$ (vs 28.6 GFLOPs for YOLOv8s)
- **Architecture Highlights**:
  - **Custom Backbone** (`IndianRoadBackbone`): Detail-preserving stem, dual-path downsampling, multi-receptive blocks (MRB) with asymmetric strip convolutions ($1\times 5, 5\times 1$), and multi-scale context blocks (MSCB).
  - **Custom Neck** (`IndianRoadNeck`): Adaptive scale fusion (ASF) with dynamic softmax gating and road context aggregator (RCA).
  - **Decoupled 3-Branch Head** (`IndianRoadHead`): Strict decoupling of bounding-box regression, foreground presence (objectness), and multi-label classification.
  - **Target Assigner**: `ScaleAdaptiveTopKMatcher` allocating equal candidate capacity ($k=4$) per scale to eliminate candidate starvation on small vehicles.
  - **Loss System** (`IndianRoadLoss`): Continuous IoU-quality soft objectness targets + Class-Balanced Focal BCE + CIoU localization.
  - **100% Pure Native PyTorch**: Zero Ultralytics dependencies.

### Empirical Benchmarks on 10,001-Image Clip-Disjoint Dataset

All rapid screening experiments were strictly evaluated under **MAXIMUM 2 EPOCHS**:

| Metric | YOLOv8s Baseline (2 Ep) | IRD Baseline (2 Ep) | IRD Loop 2 — Best Model (2 Ep) | Historical IRD (5 Ep) |
| :--- | :---: | :---: | :---: | :---: |
| **Parameters** | 11,140,244 | 4,241,529 | **4,241,529** | 4,241,529 |
| **FLOPs** | 28.6 G | 18.4 G | **18.4 G** | 18.4 G |
| **Recall** | 0.384 | 0.282 | **0.308** (+2.6%) | **0.400** |
| **mAP@0.50** | 0.3615 | 0.1030 | **0.1050** | **0.2210** |
| **mAP@0.50:0.95** | 0.2743 | 0.0440 | **0.0460** | **0.1190** |
| **Car AP50** | 0.898 | 0.574 | 0.536 | **0.785** |
| **Motorcycle AP50** | 0.636 | 0.288 | 0.244 | **0.414** |
| **Rider AP50** | 0.679 | 0.252 | 0.177 | **0.362** |
| **Autorickshaw AP50** | 0.467 | 0.040 | **0.155** (+287%) | **0.210** |
| **Truck AP50** | 0.198 | 0.008 | **0.022** (+175%) | **0.124** |
| **Traffic Sign AP50** | 0.140 | 0.000 | **0.007** (learned) | **0.012** |
| **Model FPS (RX 7700 XT)** | 79.9 | 6.4 (no gate) | **26.4** (batch 1) / **83.8** (b16) | 55.1 |

---

## 2. Benchmark Dataset & Clip-Disjoint Splitting

### Dataset Source
- Hugging Face repository: [`thirdeyelabs/indian-road-dataset`](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset)
- Total Images: **10,001** (Train: 8,282, Val: 1,719 across 135 continuous video clips).
- **Clip Overlap Guarantee**: **0% (100% Clip-Disjoint)** via deterministic SHA-256 clip hashing.

### Exact 12 Benchmark Classes
```
0: person       1: rider         2: car               3: truck
4: bus          5: motorcycle    6: bicycle           7: autorickshaw
8: animal       9: vehicle fallback 10: traffic light  11: traffic sign
```

---

## 3. Quick Start & Execution

### Environment Setup (AMD ROCm / NVIDIA CUDA / CPU)
```bash
# Clone the repository
git clone https://github.com/<username>/IndianRoadDetector.git
cd IndianRoadDetector

# Install dependencies
pip install -r requirements.txt
```

### Running Hardware Benchmarks
```bash
# Cross-platform hardware benchmark (CPU & GPU)
python scripts/benchmark_hardware.py \
    --weights experiments/custom_model/exp_loop2_adaptive_topk/ird_best.pt \
    --output-json experiments/custom_model/hardware_benchmark.json
```

### Running Real-Time Video Inference
```bash
python scripts/infer_ird.py \
    --source data/test_clip.mp4 \
    --weights experiments/custom_model/exp_loop2_adaptive_topk/ird_best.pt \
    --output experiments/custom_model/final_video/inferred_video.mp4 \
    --conf 0.20 \
    --obj-gate 0.05 \
    --device auto
```

### Evaluating an IRD Checkpoint
```bash
python scripts/evaluate_ird.py \
    --weights experiments/custom_model/exp_loop2_adaptive_topk/ird_best.pt \
    --data data/indian_road_yolo/data.yaml \
    --obj-gate 0.05 \
    --conf 0.001 \
    --device auto
```

### Training IRD V1 (Scale-Adaptive Top-K & Class-Balanced Loss)
```bash
python scripts/train_custom.py \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/exp_loop2_adaptive_topk \
    --epochs 2 \
    --batch-size 16 \
    --lr 0.001 \
    --matcher-version topk_adaptive_v2 \
    --class-balanced-loss \
    --workers 0 \
    --device auto
```

---

## 4. Hardware Performance Summary

Measured on an **AMD Ryzen 5 7600X CPU** and an **AMD Radeon RX 7700 XT GPU (12 GB VRAM, ROCm 7.2.1)**:

- **GPU Batch 1 Latency:** **37.93 ms / frame (26.37 FPS)**
- **GPU Batch 4 Throughput:** **62.00 FPS**
- **GPU Batch 16 Throughput:** **83.78 FPS**
- **CPU Latency (Single Instance):** **78.44 ms / frame (12.75 FPS)**
- **Peak VRAM Consumption:** **8,392 MB**
- **Video Inference (1080p Clip):** **52.90 ms / frame (18.90 FPS model-only)**, **70.38 ms / frame (14.21 FPS end-to-end)**

---

## 5. Repository Documentation Directory

- [`experiments/custom_model/FINAL_STATUS.md`](experiments/custom_model/FINAL_STATUS.md): Complete final research and deployment status report.
- [`src/models/ARCHITECTURE.md`](src/models/ARCHITECTURE.md): Mathematical and structural breakdown of the model.
- [`experiments/custom_model/architecture_comparison.md`](experiments/custom_model/architecture_comparison.md): Detailed comparison against YOLOv8.
- [`experiments/custom_model/ablation_results.csv`](experiments/custom_model/ablation_results.csv): Full ablation experiment records.
- [`experiments/custom_model/final_visual_validation/`](experiments/custom_model/final_visual_validation/): Rendered visual validations across 10 diagnostic scenarios.
- [`experiments/custom_model/final_video/`](experiments/custom_model/final_video/): Real-time inferred video with HUD and latency breakdown.
