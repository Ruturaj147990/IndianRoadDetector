# IRD — IndianRoadDetector (V1.5 / IRD-Next)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Custom%20PyTorch-green.svg)](src/models/ARCHITECTURE.md)
[![Final Review](https://img.shields.io/badge/Status-Ready%20for%20Full%20Training-success.svg)](experiments/custom_model/FINAL_ARCHITECTURE_REVIEW.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, genuinely custom object detection architecture specifically engineered for the unique complexities of Indian road environments: high-density traffic, severe occlusions, lane-splitting two-wheelers, small traffic signs/pedestrians, and heterogeneous vehicle compositions.

---

## 1. Project Overview & Final Architectural Status

- **Official Model Name**: **IRD — IndianRoadDetection (V1.5 / IRD-Next)**
- **Trainable Parameters**: **4,441,989** (~4.44M, 60.1% smaller than YOLOv8s)
- **Computational Complexity**: **~19.1 GFLOPs** (6.137 GMACs) at $640 \times 640$ (vs. 28.6 GFLOPs for YOLOv8s, 33.2% compute reduction)
- **Architecture Highlights**:
  - **Custom Backbone** ([`IndianRoadBackbone`](src/models/backbone/custom_backbone.py)): Detail-preserving stem, dual-path anti-aliased downsampling, multi-receptive blocks (MRB) with asymmetric strip convolutions ($1\times 5, 5\times 1$), and multi-scale context blocks (MSCB) with dilations up to $d=4$.
  - **Custom Neck** ([`IndianRoadNeck`](src/models/neck/custom_neck.py)): 
    - **Adaptive Scale Fusion** (ASF) with dynamic softmax scale gating.
    - **Selective Spatial Detail Pathway** (SSDP): Direct stride-4 Laplacian edge routing from P2 ($160\times 160$) to N3 ($80\times 80$) via anti-aliased depthwise compression and salience gating (+12.8K params).
    - **Anisotropic Traffic Disentangler** (ATD): Orthogonal strip-convolution cross-gating ($1\times 7$ and $7\times 1$) on N3 and N4 (+104.2K params) to separate dense traffic queues and decouple rider torsos from motorcycle chassis.
    - **Road Context Aggregator** (RCA) on N5 ($20\times 20$).
  - **Decoupled 4-Branch Head** ([`IndianRoadHead`](src/models/head/custom_head.py)):
    - **Bounding-box regression** with **Fine-Grained Boundary Refiner** (FGBR) on N3 (+3.6K params) predicting zero-centered bounded residuals $\Delta b = 0.5 \cdot \tanh(\text{Conv}(\nabla F))$.
    - **Foreground presence (objectness)** with focal BCE and prior-bias initialization ($\pi = 0.01$).
    - **Multi-label classification** with **Class-Discriminative Gate** (CDG) (+74.3K params) applying orthogonal aspect-ratio conditioning to separate Truck vs. Car, Bus vs. Car, and Rider vs. Person.
    - **Localization-Quality Prediction Branch** (LQB) (+3.5K params) predicting continuous IoU alignment quality $q \in [0, 1]$.
  - **Authoritative Geometry Engine** ([`src/models/box_coder.py`](src/models/box_coder.py)):
    - Smooth non-saturating box parameterization ($v_2$ smooth) preventing gradient death.
    - Quality-calibrated confidence formulation: $\text{Score} = \text{Score}_{cls} \times \sqrt{\sigma(\text{Obj}) \cdot \sigma(\text{Quality})}$.
    - Early objectness gating in logit space (`obj-gate`) skipping $>90\%$ of background cells.
    - Class-aware pure PyTorch NMS.
  - **Custom Loss & Matching** ([`src/models/losses/custom_loss.py`](src/models/losses/custom_loss.py)):
    - `ScaleAdaptiveTopKMatcher` with small-object priority.
    - Training-only `AuxiliaryOneToOneMatcher` for peak sharpening and duplicate suppression.
    - Quality-Aware BCE + Class-Balanced Focal BCE + CIoU localization.
  - **100% Native PyTorch**: Zero Ultralytics dependencies, fully cross-platform (CUDA, ROCm, CPU).

---

## 2. Benchmark Dataset & Clip-Disjoint Splitting

### Dataset Source
- Hugging Face repository: [`thirdeyelabs/indian-road-dataset`](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset)
- Total Images: **10,001** (Train: 8,282, Val: 1,719 across 135 continuous video clips, 55,597 labeled boxes).
- **Clip Overlap Guarantee**: **0% (100% Clip-Disjoint)** via deterministic SHA-256 clip hashing.

### Exact 12 Benchmark Classes
```
0: person       1: rider         2: car               3: truck
4: bus          5: motorcycle    6: bicycle           7: autorickshaw
8: animal       9: vehicle fallback 10: traffic light  11: traffic sign
```

---

## 3. Master Verification Test Suite

Before advancing to the full Colab training run, the entire architecture was verified using a comprehensive CPU-only static and synthetic test suite covering all 20 research areas:

```bash
# Run master architectural verification suite
python tests/test_final_architecture.py
```

### Verified Test Suites:
1. **Model Instantiation & Parameter Budget**: Trainable parameter count verified at **4,441,989** (+4.7% over 4.24M baseline).
2. **Multi-Batch & Multi-Resolution**: Verified across batch sizes $B \in \{1, 2, 4\}$ and resolutions $512\times 512$, $640\times 640$, $768\times 768$.
3. **Backward Graph & Gradient Flow**: Verified end-to-end backpropagation through all 809 parameter tensors with zero NaNs.
4. **Numerical Stability & Determinism**: Bitwise identical outputs across repeated forward passes; zero NaNs on extreme $[-50, +50]$ dynamic range inputs.
5. **Checkpoint Serialization**: State dictionary save, reload, and bitwise output equivalence verified.
6. **11 Synthetic Indian Traffic Scenarios**: Validated across Single Car, Multi-Car Queues, Multi-Motorcycle Clusters, Vertical Rider-Motorcycle Pairs, Dense 10+ Objects, Tiny Sub-16px Objects, Occluded Vehicles, Large 300px+ Vehicles, Overlapping Pedestrian+Car, All 12 Classes Simultaneously, and Empty Road Images.
7. **Authoritative Decoder & NMS**: Boundary clamping $[0, 640]$, quality-calibrated scoring, and class-aware NMS consistency verified.
8. **Cross-Platform Operator Safety**: 11 model files scanned; zero hardware-specific hardcoded operators found.

---

## 4. Quick Start & Execution

### Environment Setup (CUDA / ROCm / CPU)
```bash
# Clone the repository
git clone https://github.com/<username>/IndianRoadDetector.git
cd IndianRoadDetector

# Install standard dependencies
pip install -r requirements.txt
```

### Running Authoritative Verification Tests
```bash
# Run decoder consistency tests
python tests/test_authoritative_decoder.py

# Run master final architecture tests
python tests/test_final_architecture.py
```

### Full Training Configuration (Colab / GPU)
```bash
python scripts/train_custom.py \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/final_training_run \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --matcher-version topk_adaptive_v2 \
    --class-balanced-loss \
    --obj-gate 0.05 \
    --device auto
```

---

## 5. Repository Documentation Directory

- [`experiments/custom_model/FINAL_ARCHITECTURE_REVIEW.md`](experiments/custom_model/FINAL_ARCHITECTURE_REVIEW.md): Exhaustive 22-section architecture review covering baseline, explored/rejected/retained changes, parameters, FLOPs, and class-specific strategies.
- [`src/models/ARCHITECTURE.md`](src/models/ARCHITECTURE.md): Structural and mathematical breakdown of the IRD V1.5 architecture.
- [`experiments/custom_model/FINAL_STATUS.md`](experiments/custom_model/FINAL_STATUS.md): Historical closed-loop research and deployment report.
- [`experiments/custom_model/closed_loop_results.csv`](experiments/custom_model/closed_loop_results.csv): Closed-loop ablation experiment records.

---

## 6. IRD V2 Task-Aligned System (DEVELOPMENT / NOT BENCHMARKED)

> [!NOTE]
> **Status: Development / Not Benchmarked.**  
> IRD V2 is an architectural and training system redesign developed to eliminate the 506k+ false positive problem and crowded-scene recall drop identified during the IRD V1.5 deep error analysis.

### IRD V2 Core Features:
- **Task-Aligned Assignor (TAL)**: Joint classification-localization metric $t = s^{0.5} \times \text{IoU}^{6.0}$ with in-box spatial gating and deterministic multi-GT conflict resolution ([`src/models/losses/task_aligned_assignor.py`](src/models/losses/task_aligned_assignor.py)).
- **Varifocal Classification Loss (VFL)**: Continuous IoU-aware focal loss for steep negative background suppression and calibrated quality targets ([`src/models/losses/task_aligned_loss.py`](src/models/losses/task_aligned_loss.py)).
- **Explicit Zero Background Supervision**: Prevents background quality and objectness drift by explicitly supervising negative cells with $0.0$ targets.
- **Calibrated Task-Aligned Inference Decoder**: Replaces square-root confidence inflation with linear score formulation $\text{Score} = \text{Cls}^{1.0} \times \text{Quality}^{1.0}$ and class-aware NMS at $\text{IoU} = 0.40$ ([`src/models/task_aligned_decoder.py`](src/models/task_aligned_decoder.py)).
- **Complete Design Specification**: [`experiments/custom_model/IRD_V2_DESIGN.md`](experiments/custom_model/IRD_V2_DESIGN.md).

### Verification & Hardware Safety:
- **Unit Tests**: 18/18 tests passed in [`tests/test_ird_v2_task_aligned.py`](tests/test_ird_v2_task_aligned.py).
- **Static Verification**: Exact parameter match (**4,441,989**) and 100% state-dict compatibility verified in [`scripts/verify_ird_v2.py`](scripts/verify_ird_v2.py).
- **Zero Local Workstation Training**: Local workstation is strictly used for code development and static verification; all V1.5 checkpoints remain untouched.

### Cloud GPU Training Command (External Cluster):
```bash
python train.py \
    --version v2 \
    --loss-type task_aligned \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/v2_training \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --device cuda \
    --amp
```

---

## 7. Architecture Status

**STATUS: V1.5 AUTHORITATIVE BASELINE PRESERVED | V2 SYSTEM INTEGRATED & VERIFIED (NOT BENCHMARKED)**

