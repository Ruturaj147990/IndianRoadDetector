# Indian Road Custom Detector Architecture

## 1. Motivation & Context
- **Official Model Name**: **IRD — IndianRoadDetection (V1.5 / IRD-Next)**
- **Internal PyTorch Class**: [`IndianRoadDetector`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/src/models/custom_detector.py)
- **Baseline**: YOLOv8s trained on the Indian Road Dataset (8,282 train / 1,719 val, 0 clip leakage) achieved 36.1% mAP50 and 27.4% mAP50-95.
- **Goal**: Build a genuinely custom PyTorch detector engineered specifically for the challenges of Indian road environments:
  - **Extreme Occlusion & High Density**: Overlapping vehicles, lane-splitting motorcycles/auto-rickshaws, crowded pedestrians.
  - **Small Object Preservation**: Traffic signs, distant pedestrians, cycles, small road debris.
  - **Asymmetric Aspect Ratios**: Tall objects (pedestrians, riders, poles) vs. wide objects (buses, trucks, multi-car queues).
  - **Hardware Efficiency**: Optimized for training and inference on Tesla T4 GPUs and edge deployment on Jetson/CPU/ROCm.

---

## 2. Component 1: Custom Backbone (`IndianRoadBackbone`)

### Key Design Pillars (No YOLOv8 C2f/CSP Copies):
1. **Detail-Preserving Stem (`DetailPreservingStem`)**:
   - Combines a convolutional gradient path with a max-pooling peak-contrast path at stride 2.
   - Preserves high-frequency spatial cues and contrast edges crucial for small road objects at 1/2 resolution ($320\times 320$).
2. **Dual-Path Anti-Aliasing Downsampling (`DualPathDownsample`)**:
   - Integrates depthwise strided $3\times 3$ convolution and $2\times 2$ max-pooling across stage transitions.
   - Prevents spatial feature loss and aliasing during resolution reduction.
3. **Multi-Receptive Feature Block (`MultiReceptiveBlock` - MRB)**:
   - Expands channels and splits feature flow into three specialized functional branches:
     - **Branch 1 (Local Geometry)**: $3\times 3$ depthwise convolution for fine contours and edges.
     - **Branch 2 (Asymmetric Strip Convolutions)**: Sequential $1\times 5$ and $5\times 1$ depthwise convolutions tailored to tall and wide objects.
     - **Branch 3 (Dilated Context)**: $3\times 3$ depthwise convolution with dilation=2 (effective RF $5\times 5$) capturing neighboring vehicle context.
   - **Occlusion Resilience**: Integrated Squeeze-and-Excitation (SE) channel gating.
4. **Multi-Scale Context Mechanism (`MultiScaleContextBlock` - MSCB)**:
   - Splits channels into 4 groups with cascaded depthwise convolutions of dilation rates $d \in \{1, 2, 4\}$.
   - Spans receptive fields from $3\times 3$ to $15\times 15$.

### Backbone Specifications:
| Stage | Output Map | Stride | Channels | Output Resolution (640x640 Input) | Target Detections |
|---|---|---|---|---|---|
| Stem | Stem | 2 | 32 | 320 x 320 | High-frequency detail preservation |
| Stage 1 | **P2** | 4 | 64 | 160 x 160 | High-res edge representation (routed to SSDP) |
| Stage 2 | **P3** | 8 | 128 | 80 x 80 | Small objects (signs, pedestrians, bikes) |
| Stage 3 | **P4** | 16 | 256 | 40 x 40 | Medium objects (cars, auto-rickshaws) |
| Stage 4 | **P5** | 32 | 512 | 20 x 20 | Large objects & context (buses, trucks) |

- **Trainable Parameters**: 3,431,550 (~3.43M)

---

## 3. Component 2: Custom Multi-Scale Feature-Fusion Neck (`IndianRoadNeck`)

### Key Design Pillars (No YOLOv8 PAN/FPN Copies):
1. **Adaptive Scale Fusion (`AdaptiveScaleFusion` - ASF)**:
   - Replaces fixed concatenation or uniform addition.
   - Dynamically learns per-channel and per-pixel scale gating weights via softmax across scale branches.
2. **Selective Spatial Detail Pathway (`SelectiveSpatialDetailPathway` - SSDP)**:
   - Extracts stride-4 high-pass edge cues from Backbone P2 ($160\times 160$) via Laplacian edge filtering $P_2 - \text{Blur}(P_2)$.
   - Compresses via depthwise strided conv ($3\times 3, s=2$, $64\to 64$) and $1\times 1$ projection to 128 channels.
   - Injects into N3 ($80\times 80$) modulated by learned spatial salience gating (+12,865 params).
3. **Anisotropic Traffic Disentangler (`AnisotropicTrafficDisentangler` - ATD)**:
   - Integrated into Neck N3 and N4 (+104,192 params).
   - Deploys parallel horizontal ($1\times 7$) and vertical ($7\times 1$) depthwise convolutions with mutual cross-gating:
     - Horizontal strip isolates adjacent lane-splitting motorcycles and multi-car queues.
     - Vertical strip isolates rider torsos from motorcycle chassis.
4. **Road Context Aggregator (`RoadContextAggregator` - RCA)**:
   - Lightweight horizontal ($1\times 7$) and vertical ($7\times 1$) context aggregation on N5 ($20\times 20$).
5. **High-Resolution Detail Enhancer (`HighResDetailEnhancer`)**:
   - Preserves high-frequency spatial gradients on N3.

### Neck Specifications:
| Output Level | Stride | Channels | Spatial Resolution (640x640 Input) | Primary Focus |
|---|---|---|---|---|
| **N3** | 8 | 128 | 80 x 80 | Small objects: signs, pedestrians, bicycles, motorcycles (with SSDP & ATD) |
| **N4** | 16 | 128 | 40 x 40 | Medium objects: cars, auto-rickshaws, riders (with ATD) |
| **N5** | 32 | 128 | 20 x 20 | Large objects & scene layout: buses, trucks, tractors (with RCA) |

- **Trainable Parameters**: 639,497 (~0.64M)

---

## 4. Component 3: Custom Decoupled Detection Head (`IndianRoadHead`)

### Key Design Pillars (No Ultralytics Detect Copies):
1. **Quad-Branch Decoupled Design (`ScaleDecoupledHead`)**:
   - For each scale (N3, N4, N5), independently predicts:
     - **Bounding-Box Regression**: 4 box parameters via lightweight depthwise-separable convs.
     - **Object Confidence / Objectness**: 1 presence score logit via dedicated conv stack.
     - **Classification**: 12 class logits for the target Indian road categories.
     - **Localization Quality**: 1 continuous IoU quality logit.
2. **Fine-Grained Boundary Refiner (`FineGrainedBoundaryRefiner` - FGBR)**:
   - Active on Head N3 ($80\times 80$, +3,590 params).
   - Computes high-pass spatial edge gradients and predicts bounded residual offsets $\Delta b = 0.5 \cdot \tanh(\text{Conv}(\nabla F))$ to sharpen box boundaries at high IoU thresholds.
3. **Localization-Quality Prediction Branch (`LocalizationQualityBranch` - LQB)**:
   - Active on Head N3, N4, N5 (+3,456 params total).
   - Predicts continuous IoU quality $q \in [0, 1]$ directly supervised by ground-truth CIoU during training.
   - Decoupled scoring formulation: $\text{Score} = \text{Score}_{cls} \times \sqrt{\sigma(\text{Obj}) \cdot \sigma(q)}$.
4. **Class-Discriminative Gate (`ClassDiscriminativeGate` - CDG)**:
   - Active on Head N3, N4, N5 (+74,304 params total).
   - Evaluates orthogonal strip convolutions ($1\times 5$ and $5\times 1$) with Squeeze-and-Excitation global context to separate Truck vs. Car, Bus vs. Car, and Rider vs. Person.
5. **Prior-Probability Bias Initialization**:
   - Classification and objectness prediction conv biases initialized to $\text{bias} = -\log((1 - \pi)/\pi) \approx -4.595$ ($\pi = 0.01$).

### Head Specifications:
| Scale Head | Input Map | Stride | Box Preds | Obj Preds | Cls Preds | Quality Preds |
|---|---|---|---|---|---|---|
| **Head N3** | N3 (128 ch) | 8 | [B, 4, 80, 80] | [B, 1, 80, 80] | [B, 12, 80, 80] | [B, 1, 80, 80] |
| **Head N4** | N4 (128 ch) | 16 | [B, 4, 40, 40] | [B, 1, 40, 40] | [B, 12, 40, 40] | [B, 1, 40, 40] |
| **Head N5** | N5 (128 ch) | 32 | [B, 4, 20, 20] | [B, 1, 20, 20] | [B, 12, 20, 20] | [B, 1, 20, 20] |

- **Trainable Parameters**: 370,942 (~0.37M)

---

## 5. Full Integrated Detector (`IndianRoadDetector` / **IRD V1.5**)

```
Input [B, 3, 640, 640]
       │
       ▼
IndianRoadBackbone
 ├── Stem (stride 2) ──► 320x320 (32 ch)
 ├── Stage 1 (P2, stride 4) ──► 160x160 (64 ch) ──► [SSDP High-Pass Path]
 ├── Stage 2 (P3, stride 8) ──► 80x80 (128 ch)
 ├── Stage 3 (P4, stride 16) ─► 40x40 (256 ch)
 └── Stage 4 (P5, stride 32) ─► 20x20 (512 ch)
       │
       ▼
IndianRoadNeck
 ├── Lateral Projections (128 ch)
 ├── Top-Down Adaptive Scale Fusion (ASF)
 ├── Selective Spatial Detail Pathway (SSDP) on N3
 ├── High-Res Detail Enhancer on N3
 ├── Bottom-Up Adaptive Scale Fusion (ASF)
 ├── Anisotropic Traffic Disentangler (ATD) on N3 & N4
 └── Road Context Aggregator (RCA) on N5
       ├── N3 (stride 8) ──► 80x80 (128 ch)
       ├── N4 (stride 16) ─► 40x40 (128 ch)
       └── N5 (stride 32) ─► 20x20 (128 ch)
       │
       ▼
IndianRoadHead
 ├── Head N3 (s8)  ──► Box+FGBR [4], Obj [1], Cls+CDG [12], Quality [1]
 ├── Head N4 (s16) ──► Box [4],      Obj [1], Cls+CDG [12], Quality [1]
 └── Head N5 (s32) ──► Box [4],      Obj [1], Cls+CDG [12], Quality [1]
```

### Parameter & Compute Breakdown:
| Component | Trainable Parameters | Share (%) | FLOPs (at 640x640) |
|---|---|---|---|
| **Backbone** (`IndianRoadBackbone`) | 3,431,550 (~3.43M) | 77.3% | ~12.8 GFLOPs |
| **Neck** (`IndianRoadNeck`) | 639,497 (~0.64M) | 14.4% | ~4.5 GFLOPs |
| **Head** (`IndianRoadHead`) | 370,942 (~0.37M) | 8.4% | ~1.8 GFLOPs |
| **Complete IRD Detector** | **4,441,989 (~4.44M)** | **100.0%** | **~19.1 GFLOPs** |

---

## 6. Component 4: Custom Loss & Target Assignment (`IndianRoadLoss`)

1. **Scale-Adaptive Top-K Matcher (`ScaleAdaptiveTopKMatcher`)**:
   - Matches ground truth objects dynamically across strides 8, 16, 32 using scale alignment and center-proximity sampling ($r=1.5$).
   - Top-$k$ candidates assigned per object with small-object priority to prevent small pedestrians and riders from being masked by adjacent large vehicles.
2. **Auxiliary One-to-One Matcher (`AuxiliaryOneToOneMatcher`)**:
   - Training-only supervision assigning strictly 1 anchor location per GT object.
   - Enforces sharp single-peak predictions and reduces duplicate candidates with zero inference cost.
3. **Localization Quality Loss**:
   - Binary Cross-Entropy on positive match locations between predicted quality logit and actual CIoU overlap.
4. **Complete IoU (CIoU) Box Loss**:
   - Penalizes overlap error ($1 - \text{IoU}$), normalized center distance ($\rho^2 / c^2$), and aspect ratio discrepancy ($\alpha v$).
5. **Class-Balanced Focal Classification Loss**:
   - Inverse-frequency focal weighting tailored to minority classes:
     $$\mathbf{w}_{cls} = [2.2, 1.8, 1.0, 2.4, 2.5, 1.8, 2.4, 2.3, 2.5, 2.3, 2.5, 2.4]$$

---

## 7. Component 5: Authoritative Box Decoding Engine (`src/models/box_coder.py`)

1. **Smooth Non-Saturating Box Parameterization (`v2_smooth`)**:
   $$c_x = (g_x + 2\sigma(t_x) - 0.5) \cdot \text{stride}$$
   $$c_y = (g_y + 2\sigma(t_y) - 0.5) \cdot \text{stride}$$
   $$w = \text{stride} \cdot \exp(3.0 \cdot \tanh(t_w / 3.0))$$
   $$h = \text{stride} \cdot \exp(3.0 \cdot \tanh(t_h / 3.0))$$
   - Eliminates gradient saturation/death during training.
2. **Early Objectness Gating in Logit Space (`obj-gate`)**:
   - Prunes negative cells before decoding coordinates, skipping $>90\%$ of background cells.
3. **Quality-Calibrated Confidence Formulation**:
   $$\text{Score} = \text{Score}_{cls} \times \sqrt{\sigma(\text{Obj}) \cdot \sigma(\text{Quality})}$$
4. **Class-Aware Pure PyTorch NMS**:
   - Pure PyTorch NMS with spatial offset by class ID to prevent cross-class suppression between riders and motorcycles.

---

## 8. Verification & Compatibility Summary

- **Static & Synthetic Test Suite**: 100% passed across all 8 verification suites in [`tests/test_final_architecture.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/tests/test_final_architecture.py).
- **CUDA / ROCm / CPU Ready**: Pure standard PyTorch ops, zero hardcoded `.cuda()` calls.
- **Status**: **V1.5 AUTHORITATIVE BASELINE READY**

---

## 9. Component 6: IRD V2 Task-Aligned System (DEVELOPMENT / NOT BENCHMARKED)

> [!NOTE]
> **Status: Under Active Development / Not Benchmarked.**
> IRD V2 is an architectural and training system redesign developed to eliminate the 506k+ false positive problem and crowded-scene recall drop identified during the IRD V1.5 deep error analysis.

### Core Enhancements:
1. **Task-Aligned Assignor (TAL)** ([`src/models/losses/task_aligned_assignor.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/src/models/losses/task_aligned_assignor.py)):
   - Replaces static geometric center radius with joint metric: $t = s^{0.5} \times \text{IoU}^{6.0}$.
   - Dynamically retains top-$k=10$ candidate anchors per ground truth with in-box spatial gating.
   - Deterministic multi-GT conflict resolution assigning contested anchors to $\arg\max_j t_{j, i}$.
2. **Varifocal Classification Loss (VFL)** ([`src/models/losses/task_aligned_loss.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/src/models/losses/task_aligned_loss.py)):
   - Positives ($q > 0$): Supervised with continuous alignment target $q = \text{IoU} \times \frac{t}{\max(t)}$.
   - Negatives ($q = 0$): Heavily downweighted with focal loss $-\alpha p^\gamma \log(1 - p)$ ($\alpha=0.75, \gamma=2.0$), driving background logits strongly negative.
3. **Quality-Weighted CIoU Loss**:
   - Weights bounding box CIoU loss by continuous alignment target $q_i$, downweighting sloppy boxes.
4. **Explicit Zero Background Supervision**:
   - All background grid cells receive explicit continuous targets of $0.0$ on both quality and objectness branches.
5. **Calibrated Task-Aligned Inference Decoder** ([`src/models/task_aligned_decoder.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/src/models/task_aligned_decoder.py)):
   - Primary score: $\text{Score} = \text{Cls}^{1.0} \times \text{Quality}^{1.0}$ (or direct $\text{Cls}$).
   - Completely eliminates square-root background noise inflation.
   - Class-aware NMS default: $\text{IoU} = 0.40, \text{conf} = 0.25$.
6. **Validation Status**:
   - 18/18 Unit Tests Passed in [`tests/test_ird_v2_task_aligned.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/tests/test_ird_v2_task_aligned.py).
   - Static verification passed with exact parameter match (4,441,989) and zero NaNs/Infs in [`scripts/verify_ird_v2.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/scripts/verify_ird_v2.py).
   - Checkpoint `ird_best.pt` 100% preserved. Zero training performed locally.