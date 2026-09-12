# Indian Road Custom Detector Architecture

## 1. Motivation & Context
- **Official Model Name**: **IRD — IndianRoadDetection (V1)**
- **Internal PyTorch Class**: `IndianRoadDetector`
- **Baseline**: YOLOv8s trained on the Indian Road Dataset (8,000 train / 2,000 val) achieved 42.1% mAP50 and 33.1% mAP50-95.
- **Goal**: Build a genuinely custom PyTorch detector engineered specifically for the challenges of Indian road environments:
  - **Extreme Occlusion & High Density**: Overlapping vehicles, lane-splitting motorcycles/auto-rickshaws, crowded pedestrians.
  - **Small Object Preservation**: Traffic signs, distant pedestrians, cycles, small debris/potholes.
  - **Asymmetric Aspect Ratios**: Tall objects (pedestrians, poles, traffic signals) vs. wide objects (buses, trucks, barricades).
  - **Hardware Efficiency**: Optimized for training and inference on Tesla T4 GPUs.

---

## 2. Component 1: Custom Backbone (`IndianRoadBackbone`)

### Key Design Pillars (No YOLOv8 C2f/CSP Copies):
1. **Detail-Preserving Stem (`DetailPreservingStem`)**:
   - Combines a convolutional gradient path with a max-pooling peak-contrast path at stride 2.
   - Preserves high-frequency spatial cues and contrast edges crucial for small road objects at 1/2 resolution (320x320).
2. **Dual-Path Anti-Aliasing Downsampling (`DualPathDownsample`)**:
   - Integrates depthwise strided 3x3 convolution and 2x2 max-pooling across stage transitions.
   - Prevents spatial feature loss and aliasing during resolution reduction.
3. **Multi-Receptive Feature Block (`MultiReceptiveBlock` - MRB)**:
   - Expands channels and splits feature flow into three specialized functional branches:
     - **Branch 1 (Local Geometry)**: 3x3 depthwise convolution for fine contours and edges.
     - **Branch 2 (Asymmetric Strip Convolutions)**: Sequential 1x5 and 5x1 depthwise convolutions tailored to tall and wide objects.
     - **Branch 3 (Dilated Context)**: 3x3 depthwise convolution with dilation=2 (effective RF 5x5) capturing neighboring vehicle context.
   - **Occlusion Resilience**: Integrated Squeeze-and-Excitation (SE) channel gating.
4. **Multi-Scale Context Mechanism (`MultiScaleContextBlock` - MSCB)**:
   - Splits channels into 4 groups with cascaded depthwise convolutions of dilation rates $d \in \{1, 2, 4\}$.
   - Spans receptive fields from 3x3 to 15x15.

### Backbone Specifications:
| Stage | Output Map | Stride | Channels | Output Resolution (640x640 Input) | Target Detections |
|---|---|---|---|---|---|
| Stem | Stem | 2 | 32 | 320 x 320 | High-frequency detail preservation |
| Stage 1 | P2 | 4 | 64 | 160 x 160 | High-res geometric representation |
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
   - Permits high-resolution boundaries (for small signs/pedestrians) to dominate at edge locations while allowing semantic context (for occluded vehicles) to dominate in ambiguous regions.
2. **Road Context Aggregator (`RoadContextAggregator` - RCA)**:
   - Exploit domain-specific road geometry using three lightweight parallel branches:
     - **Horizontal Strip (1x7)**: Traffic lanes, multi-vehicle queues, barricades.
     - **Vertical Strip (7x1)**: Pedestrians, traffic lights, electricity poles.
     - **Dilated 2D Context (3x3, d=2)**: Local surrounding context without edge blurring.
     - **Global Channel Excitation**: Dynamic scene descriptor modulation.
3. **High-Resolution Detail Enhancer (`HighResDetailEnhancer`)**:
   - Preserves fragile high-frequency gradients for N3 (80x80) via high-pass filtering and learnable residual injection.
4. **Deep Feature Refinement (`RoadFusionBlock`)**:
   - Multi-receptive feature refinement after each scale fusion step.
5. **Anti-Aliased Downsampling (`NeckDownsampler`)**:
   - Dual-path strided depthwise conv + pooling downsampling for bottom-up localization flow.

### Neck Specifications:
| Output Level | Stride | Channels | Spatial Resolution (640x640 Input) | Primary Focus |
|---|---|---|---|---|
| **N3** | 8 | 128 | 80 x 80 | Small objects: signs, pedestrians, bicycles, motorcycles |
| **N4** | 16 | 128 | 40 x 40 | Medium objects: cars, auto-rickshaws, riders |
| **N5** | 32 | 128 | 20 x 20 | Large objects & scene layout: buses, trucks, tractors |

- **Trainable Parameters**: 524,872 (~0.52M)

---

## 4. Component 3: Custom Decoupled Detection Head (`IndianRoadHead`)

### Key Design Pillars (No Ultralytics Detect Copies):
1. **Tri-Branch Decoupled Design (`ScaleDecoupledHead`)**:
   - For each scale (N3, N4, N5), independently predicts:
     - **Bounding-Box Regression**: 4 box parameters via lightweight depthwise-separable convs.
     - **Object Confidence / Objectness**: 1 presence score logit via dedicated conv stack.
     - **Classification**: 12 class logits for the target Indian road categories.
   - Eliminates classification-localization feature conflict in congested scenes.
2. **Spatial Detail Preserver (`SpatialDetailPreserver`)**:
   - Integrated into the N3 head (stride 8, 80x80 resolution) to preserve sharp spatial edge gradients for distant pedestrians, bicycle wheels, and small traffic lights/signs.
3. **Prior-Probability Bias Initialization**:
   - Classification and objectness prediction conv biases are initialized to $\text{bias} = -\log((1 - \pi)/\pi) \approx -4.595$ with $\pi = 0.01$.
   - Eliminates early loss spikes and stabilizes training under extreme class imbalance.
4. **Structured Head Output (`HeadOutput`)**:
   - Transparent, modular container supporting dict access, attribute access, and tuple unpacking.
   - Keeps raw predictions separate from decoding/NMS for clean integration with future custom loss functions.

### Head Specifications:
| Scale Head | Input Map | Stride | Box Preds | Obj Preds | Cls Preds | Spatial Output |
|---|---|---|---|---|---|---|
| **Head N3** | N3 (128 ch) | 8 | [B, 4, 80, 80] | [B, 1, 80, 80] | [B, 12, 80, 80] | 80 x 80 |
| **Head N4** | N4 (128 ch) | 16 | [B, 4, 40, 40] | [B, 1, 40, 40] | [B, 12, 40, 40] | 40 x 40 |
| **Head N5** | N5 (128 ch) | 32 | [B, 4, 20, 20] | [B, 1, 20, 20] | [B, 12, 20, 20] | 20 x 20 |

- **Trainable Parameters**: 285,107 (~0.29M)

---

## 5. Full Integrated Detector (`IndianRoadDetector` / **IRD V1**)

```
Input [B, 3, 640, 640]
       │
       ▼
IndianRoadBackbone
 ├── Stem (stride 2) ──► 320x320
 ├── Stage 1 (P2, stride 4) ──► 160x160
 ├── Stage 2 (P3, stride 8) ──► 80x80 (128 ch)
 ├── Stage 3 (P4, stride 16) ─► 40x40 (256 ch)
 └── Stage 4 (P5, stride 32) ─► 20x20 (512 ch)
       │
       ▼
IndianRoadNeck
 ├── Lateral Projections (128 ch)
 ├── Top-Down Adaptive Scale Fusion (ASF)
 ├── High-Res Detail Enhancer
 ├── Bottom-Up Adaptive Scale Fusion (ASF)
 └── Road Context Aggregator (RCA)
       ├── N3 (stride 8) ──► 80x80 (128 ch)
       ├── N4 (stride 16) ─► 40x40 (128 ch)
       └── N5 (stride 32) ─► 20x20 (128 ch)
       │
       ▼
IndianRoadHead
 ├── ScaleDecoupledHead (N3) ──► Box [B, 4, 80, 80], Obj [B, 1, 80, 80], Cls [B, 12, 80, 80]
 ├── ScaleDecoupledHead (N4) ──► Box [B, 4, 40, 40], Obj [B, 1, 40, 40], Cls [B, 12, 40, 40]
 └── ScaleDecoupledHead (N5) ──► Box [B, 4, 20, 20], Obj [B, 1, 20, 20], Cls [B, 12, 20, 20]
```

### Parameter Breakdown:
| Component | Trainable Parameters | Percentage of Total |
|---|---|---|
| **Backbone** (`IndianRoadBackbone`) | 3,431,550 (~3.43M) | 80.9% |
| **Neck** (`IndianRoadNeck`) | 524,872 (~0.52M) | 12.4% |
| **Head** (`IndianRoadHead`) | 285,107 (~0.29M) | 6.7% |
| **Complete IRD Pipeline** | **4,241,529 (~4.24M)** | **100.0%** |

---

## 6. Component 4: Custom Detection Loss & Target Assignment (`IndianRoadLoss`)

### Key Design Pillars (No Ultralytics Loss Copies):
1. **Multi-Scale Spatial Matcher (`MultiScaleSpatialMatcher`)**:
   - **Scale Assignment**: Dynamically matches targets based on scale $D = \sqrt{w \times h}$ across strides 8, 16, 32 with overlapping scale bounds.
   - **Center Proximity**: Samples candidate grid cells within radius $r=1.2$ inside the target box.
   - **Small-Object Priority**: Resolves multi-object spatial collision in congested traffic by prioritizing the smaller box area.
2. **Complete IoU (CIoU) Bounding-Box Loss**:
   - Jointly penalizes overlap error ($1 - \text{IoU}$), normalized center distance ($\rho^2 / c^2$), and aspect ratio discrepancy ($\alpha v$).
3. **Focal Objectness Loss**:
   - Binary Cross-Entropy with focal modulation ($\gamma = 2.0, \alpha = 0.25$) on all 8,400 grid cells.
4. **Multi-Label Focal Classification Loss**:
   - Evaluated on positive locations across the 12 Indian road classes.
5. **Structured Loss Output (`LossResult`)**:
   - Returns `total_loss`, `box_loss`, `objectness_loss`, `classification_loss`, and `number_of_positive_samples`.

---

## 7. Component 5: Tiny-Dataset Overfitting Verification Test

Verified via [`overfit_test.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/experiments/custom_model/overfit_test.py):
- **Training Total Loss Reduction**: **`96.93%`** (`142.45` $\to$ `4.37`).
- **Post-Training Evaluation (Same 16 Images)**:
  - Classification Loss dropped by **`98.93%`** (`4.51` $\to$ `0.048`).
  - Objectness Loss dropped by **`72.36%`** (`1.13` $\to$ `0.31`).
  - Box CIoU Loss dropped by **`20.08%`** (`0.971` $\to$ `0.776`).
  - Total Loss dropped by **`59.6%`** (`10.50` $\to$ `4.24`).

---

## 8. Component 6: Production Training Pipeline (`train_custom.py`)

Implementation: [`train_custom.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/scripts/train_custom.py)

### Pipeline Capabilities:
1. **Model Identification**: **IRD (IndianRoadDetection)**.
2. **Pure PyTorch Architecture**: 100% independent from Ultralytics training/loss components.
3. **Data Handling**: Dynamic dataset discovery (`/content/indian_road_yolo` with local auto-fallback), YOLO coordinate validation, PyTorch DataLoader integration.
4. **Training Optimization**: AdamW optimizer, Cosine Annealing scheduler, Automatic Mixed Precision (`torch.amp`), and gradient clipping ($10.0$).
5. **Transparent Validation**: Evaluates without gradients every epoch, recording validation loss breakdown, mean match IoU, and top-1 class accuracy.
6. **Robust Checkpoint Management**:
   - Best checkpoint: `ird_best.pt`
   - Latest checkpoint: `ird_last.pt`
   - Resumption: Full restoration of model weights, optimizer, scheduler, scaler, epoch count, and history.
   - History logs: `ird_history.csv`, `ird_history.json`, `ird_config.json`.

---

## 9. Component 7: Benchmark Dataset Pipeline & Clip-Disjoint Splitting

Implementations:
- Pipeline: [`src/data/convert_bdd_to_yolo.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/src/data/convert_bdd_to_yolo.py)
- Audit & Verification: [`src/data/verify_dataset.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/src/data/verify_dataset.py)

### Design & Benchmark Integrity:
1. **Identical Class Semantics with YOLOv8 Baseline**:
   - Exactly 12 classes in identical ordering (0: person, 1: rider, 2: car, 3: truck, 4: bus, 5: motorcycle, 6: bicycle, 7: autorickshaw, 8: animal, 9: vehicle fallback, 10: traffic light, 11: traffic sign).
   - Guarantees valid, apples-to-apples comparison between YOLOv8s and IRD V1.
2. **Elimination of Video Clip Leakage**:
   - In BDD100K-style road datasets (`thirdeyelabs/indian-road-dataset`), consecutive frames belong to continuous video clips.
   - Splitting at the frame level causes clip leakage (near-identical backgrounds/agents across train and val).
   - The pipeline enforces **atomic clip-level assignment** via deterministic SHA-256 hashing ($\text{train\_ratio} = 0.8$, $\text{seed} = 42$).
   - $\text{Clips}_{\text{train}} \cap \text{Clips}_{\text{val}} = \emptyset$ is mathematically and empirically guaranteed.
3. **RAM-Safe Direct Streaming**:
   - Streams from Hugging Face via `IterableDataset` and writes directly to disk, avoiding high-RAM crashes.
4. **Clip Boundary Integrity**:
   - Prioritizes clip completeness over exact sample boundaries. Terminating at clip boundaries ensures no video clip is truncated or partially written.
5. **Multi-Point Verification Audit**:
   - Automatically checks clip disjointness, image-label parity, coordinate bounds, and class distribution.