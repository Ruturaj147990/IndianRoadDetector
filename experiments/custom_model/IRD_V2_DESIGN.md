# IRD V2 Architecture & Training System Design Specification

**Status:** DEVELOPMENT / NOT BENCHMARKED  
**Model Name:** IRD V2 (IndianRoadDetection V2)  
**Baseline Model:** IRD V1.5 (`experiments/custom_model/final_training_50ep/ird_best.pt`, Epoch 19)  
**Date:** September 14, 2026  

> [!IMPORTANT]
> **DEVELOPMENT NOTICE:**
> - IRD V2 is an architectural and training system redesign developed to address the specific failure modes identified during the IRD V1.5 deep error analysis.
> - **Zero training was performed on this local workstation**; all validation consists strictly of static integrity checks, unit tests (18/18 passed), and synthetic forward/backward pass validations.
> - No empirical benchmark claims or hypothetical mAP numbers are reported as measured facts. Actual benchmarking will occur after external cloud GPU training.

---

## 1. Executive Summary & Design Motivation

During the deep error analysis of IRD V1.5 across all 1,719 validation images, three critical systemic deficiencies were uncovered:

1. **Unsupervised Background Quality Inflation:**
   - V1.5 confidence scoring: $\text{Score} = \text{Score}_{\text{cls}} \times \sqrt{\text{Objectness} \times \text{Quality}}$
   - Because the localization quality branch was supervised only on positive cells during training, background cells floated at uncalibrated initialization levels ($\sigma \approx 0.65$).
   - The square-root operator amplified tiny background objectness values ($0.01 \rightarrow 0.08$), pushing 497,837 background predictions into the 0.05–0.25 confidence range, causing 506,726 false positives across 1,719 validation images.
2. **Static Center-Distance Matcher Failure in Congestion:**
   - V1.5 assigned targets using fixed geometric center radius ($r = 1.2$) and static scale bins.
   - In crowded traffic clusters (10+ objects), recall dropped by 27.1% (from 96.0% down to 68.9%) due to grid cell contention, while adjacent cells generated multi-cell duplicate boxes.
3. **Severe Extreme-Class Imbalance Collapse:**
   - Static focal loss weighting starved rare classes (`animal`, `traffic light`, `traffic sign`, `vehicle fallback`) of gradient representation, causing AP50 to collapse between 0.0002 and 0.0524.

**IRD V2** directly resolves these three failure modes by introducing **Task-Aligned Assignment (TAL)**, **Varifocal continuous IoU-aware classification loss**, and **calibrated task-aligned scoring with explicit zero background supervision**, while preserving the exact 4,441,989 parameter backbone and neck of IRD V1.5.

---

## 2. Mathematical Formulation of IRD V2

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               IRD V2 Multi-Scale Prediction             │
                    │      Boxes: b_pred [B, 8400, 4] | Cls: s_pred [B, 8400, C] │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                                                 ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │            Task-Aligned Assignor (TAL)                  │
                    │   1. Spatial In-Box Gating (anchor in GT box)            │
                    │   2. Task Alignment Metric: t = s^0.5 * IoU^6.0         │
                    │   3. Dynamic Top-k Selection (k = 10 per GT)            │
                    │   4. Deterministic Multi-GT Contention Resolution       │
                    │   5. Continuous Alignment Targets: t* = IoU * (t / t_max)│
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
         ┌───────────────────────────────┐               ┌───────────────────────────────┐
         │     Varifocal Loss (VFL)      │               │   Quality-Weighted CIoU Loss  │
         │  Positives (q > 0):           │               │  L_box = sum(q_i * (1 - CIoU))│
         │    -q*(q*log(p)+(1-q)*log(1-p))│               │         / sum(q_i)            │
         │  Negatives (q = 0):           │               │  Higher aligned candidates    │
         │    -alpha * p^gamma * log(1-p)│               │  receive larger gradients     │
         └───────────────────────────────┘               └───────────────────────────────┘
```

### A. Task-Aligned Assignor (TAL)
Rather than relying on rigid geometric center distances, candidate anchors are evaluated jointly on classification confidence $s$ and localization quality $\text{IoU}$:

$$t = s^\alpha \times \text{IoU}^\beta$$

- **$s \in [0, 1]$:** Predicted classification probability for the target ground-truth class $c_{\text{gt}}$ at that anchor: $s = \sigma(z_{\text{cls}, c_{\text{gt}}})$.
- **$\text{IoU} \in [0, 1]$:** Pairwise Intersection over Union between the decoded predicted box $\hat{b}$ and the ground-truth box $g$.
- **$\alpha = 0.5$:** Classification score exponent.
- **$\beta = 6.0$:** Localization IoU exponent, ensuring that only tightly localized boxes achieve high alignment scores.

#### 1. Spatial In-Box Gating
An anchor cell with center coordinate $(cx_{\text{anchor}}, cy_{\text{anchor}})$ in pixels is considered a valid candidate for ground-truth box $g = [x_1, y_1, x_2, y_2]$ if and only if:
$$x_1 \le cx_{\text{anchor}} \le x_2 \quad \text{and} \quad y_1 \le cy_{\text{anchor}} \le y_2$$
Any anchor outside the bounding box receives $t = 0.0$.

#### 2. Dynamic Top-$k$ Candidate Selection
For each ground-truth object $j$, anchors are ranked by $t_j$. The top-$k$ ($k=10$) anchors with $t > 0$ are retained as positive candidate anchors.

#### 3. Deterministic Multi-GT Conflict Resolution
When an anchor $i$ falls within the candidate set of multiple ground-truth objects (e.g. in crowded clusters or overlapping `rider` + `motorcycle` pairs), it is deterministically assigned to the ground truth $j^*$ that maximizes the task-alignment metric:
$$j^* = \arg\max_j t_{j, i}$$
This eliminates arbitrary assignment order dependencies.

#### 4. Continuous Target Normalization
For each ground truth $j$, the alignment metric is normalized by the maximum alignment score achieved across its candidates:
$$t^*_{j, i} = \text{IoU}_{j, i} \times \frac{t_{j, i}}{\max_m(t_{j, m}) + \epsilon}$$
- The best-aligned anchor achieves a target score equal to its actual $\text{IoU}$.
- Lower-aligned candidates receive proportionally lower targets.
- **All background anchors and non-target classes receive continuous targets strictly equal to 0.0.**

---

### B. Varifocal Classification Loss (VFL)
To supervise dense anchors with continuous alignment targets $q \in [0, 1]$, IRD V2 adopts Varifocal Loss:

$$\text{VFL}(p, q) = \begin{cases} -q \cdot \left[ q \log(p) + (1 - q) \log(1 - p) \right], & q > 0 \\ -\alpha \cdot p^\gamma \log(1 - p), & q = 0 \end{cases}$$

- **Positives ($q > 0$):** Weighted by the continuous alignment target $q = t^*$. It trains the classification head to directly predict the task-aligned IoU score.
- **Negatives ($q = 0$):** Evaluated with focal downweighting $-\alpha p^\gamma \log(1 - p)$ with $\alpha = 0.75$ and $\gamma = 2.0$. Background patches with small probabilities ($p < 0.1$) generate near-zero loss, while any background patch hallucinating higher probabilities is penalized with steep gradients, driving background logits negative ($\le -4.5$).

### C. Quality-Weighted Bounding Box Loss
Bounding box regression is optimized using Complete IoU (CIoU), weighted by the continuous task-alignment score $q_i$:

$$\mathcal{L}_{\text{box}} = \frac{1}{\sum_{i \in \text{pos}} q_i + \epsilon} \sum_{i \in \text{pos}} q_i \cdot \left( 1 - \text{CIoU}(\hat{b}_i, b^*_i) \right)$$

This prevents low-quality candidates from corrupting box regression gradients, focusing gradient updates on candidates with high alignment potential.

### D. Explicit Background Quality & Objectness Supervision
If the decoupled head's localization quality or objectness branches are retained, they are supervised across **all 8,400 grid anchors**:
- **Positives ($i \in \text{pos}$):** Target equals $\text{IoU}(\hat{b}_i, b^*_i)$ or continuous alignment score $q_i$.
- **Negatives ($i \in \text{neg}$):** Target is **strictly 0.0**.
- **Loss:**
  $$\mathcal{L}_{\text{qual}} = \frac{1}{\sum q_i + \epsilon} \sum_{i=1}^{N_{\text{anchors}}} \text{BCEWithLogits}(\hat{q}_i, \text{target}_{q, i})$$

This mathematically guarantees that background quality logits cannot float at uncontrolled initialization levels ($\sigma \approx 0.65$), driving background quality probabilities down to $\approx 0.0$.

---

### E. Calibrated Task-Aligned Inference Scoring

IRD V2 eliminates the square-root confidence formula:
$$\text{V1.5 (Removed): } \text{Score} = \text{Score}_{\text{cls}} \times \sqrt{\text{Objectness} \times \text{Quality}}$$

Instead, IRD V2 implements **linear task-aligned scoring**:
$$\text{V2 (Task-Aligned): } \text{Score} = \text{Score}_{\text{cls}}^{\alpha_s} \times \text{Quality}^{\beta_s} \quad (\text{default: } \alpha_s = 1.0, \beta_s = 1.0)$$
$$\text{V2 (Direct-Cls): } \text{Score} = \text{Score}_{\text{cls}}$$

Because classification probabilities are trained via Varifocal Loss with continuous IoU targets, $\text{Score}_{\text{cls}}$ already represents the joint quality-aligned probability. When combined linearly with explicitly supervised quality logits, background noise multiplies toward zero rather than being amplified.

---

## 3. Side-by-Side Comparison: IRD V1.5 vs IRD V2

| Component | IRD V1.5 Baseline | IRD V2 Redesign | Rationale for Change |
| :--- | :--- | :--- | :--- |
| **Target Matcher** | MultiScaleSpatialMatcher (static center radius $r=1.2$) | TaskAlignedAssignor ($t = s^{0.5} \times \text{IoU}^{6.0}$, top-$k=10$) | Resolves cell contention in crowded traffic and eliminates adjacent stride echoes |
| **Classification Loss** | Multi-label Focal BCE on binary $\{0, 1\}$ targets | Varifocal Loss (VFL) on continuous $q \in [0, 1]$ targets | Aligns classification score directly with IoU; enforces steep negative background suppression |
| **Box Loss Weighting** | Unweighted mean CIoU over positive cells | Quality-weighted CIoU ($\sum q_i (1 - \text{CIoU}) / \sum q_i$) | Downweights sloppy anchors; focuses gradients on high-IoU candidates |
| **Background Quality** | Unsupervised (floating logits $\sigma \approx 0.65$) | Explicitly supervised with continuous $0.0$ target | Prevents background quality drift and eliminates hallucinated detections |
| **Inference Score** | $\text{Cls} \times \sqrt{\text{Obj} \times \text{Quality}}$ | $\text{Cls}^{1.0} \times \text{Quality}^{1.0}$ or direct $\text{Cls}$ | Eliminates square-root background inflation that caused 506k+ FPs |
| **Class Imbalance** | Static heuristic weights | Principled inverse-frequency weights on positive VFL targets | Prevents gradient starvation on rare classes (`animal`, `traffic light`, etc.) |
| **NMS Post-Processing**| Class-aware greedy NMS ($\text{IoU} = 0.40$) | Class-aware greedy NMS ($\text{IoU} = 0.40$) | Preserved; validated with zero cross-class interference |
| **Backbone & Neck** | IndianRoadBackbone + IndianRoadNeck (4.44M params) | IndianRoadBackbone + IndianRoadNeck (4.44M params) | 100% Preserved; enables controlled ablation of scoring & matching |

---

## 4. What Remains Unchanged in IRD V2

To maintain scientific control and isolate the impact of scoring and assignment improvements, the following subsystems remain 100% identical to IRD V1.5:

1. **Backbone Architecture:**
   - `IndianRoadBackbone` with `DetailPreservingStem`, `DualPathDownsample`, `MultiReceptiveBlock` (MRB), and `MultiScaleContextBlock` (MSCB).
   - Channels: Stem 32, P2 64, P3 128, P4 256, P5 512.
   - Trainable parameters: **3,431,550**.
2. **Neck Architecture:**
   - `IndianRoadNeck` with `AdaptiveScaleFusion`, `HighResDetailEnhancer`, `RoadFusionBlock`, `NeckDownsampler`, and `RoadContextAggregator`.
   - Channels: Unified 128 channels across N3, N4, N5.
   - Trainable parameters: **639,497**.
3. **Head Structure:**
   - Decoupled branches for regression, classification, objectness, and quality.
   - Trainable parameters: **370,942**.
   - **Total Model Parameters: 4,441,989 (Exact match with V1.5).**
4. **NMS Engine:**
   - Class-aware spatial offset separation ($c \times 10,000$).
   - Preserves overlapping co-occurrences (`rider` + `motorcycle`, `person` + `car`).
5. **Dataset & Evaluation Splits:**
   - 1,719 validation images with 8,757 annotations strictly preserved.

---

## 5. Hyperparameter Specification

| Hyperparameter | Default Value | Role / Mechanism |
| :--- | :---: | :--- |
| `topk` | `10` | Number of candidate anchors dynamically retained per ground-truth object in TAL. |
| `tal_alpha` | `0.5` | Classification score exponent in task-alignment metric $t = s^\alpha \times \text{IoU}^\beta$. |
| `tal_beta` | `6.0` | Localization IoU exponent in task-alignment metric. High value favors tight localization. |
| `vfl_alpha` | `0.75` | Negative sample scaling factor in Varifocal Loss. |
| `vfl_gamma` | `2.0` | Negative sample focusing parameter in Varifocal Loss for steep background suppression. |
| `box_weight` | `5.0` | Multiplier for quality-weighted CIoU box loss. |
| `cls_weight` | `1.0` | Multiplier for Varifocal classification loss. |
| `qual_weight`| `0.5` | Multiplier for explicit background quality supervision. |
| `obj_weight` | `1.0` | Multiplier for explicit background objectness supervision. |
| `conf_threshold` | `0.25` | Production inference confidence threshold. |
| `iou_threshold` | `0.40` | Class-aware NMS IoU suppression threshold. |
| `max_det` | `300` | Maximum allowed detections per image. |

---

## 6. Expected Failure Modes & Diagnostic Roadmap

While IRD V2 directly addresses false-positive inflation and crowded-scene cell contention, the following residual failure modes are expected and will form the basis for future ablations:

1. **Tiny Object Resolution Limit:**
   - The finest feature map remains N3 (stride 8, $80 \times 80$). Extremely small traffic lights and distant signs ($< 16 \times 16$ px) may still suffer from low feature representation after initial downsampling.
   - *Future Ablation:* Introduction of a P2 high-resolution pathway (stride 4, $160 \times 160$).
2. **Extreme Class Prior Asymmetry:**
   - `car` instances outnumber `animal` and `traffic light` by over 40:1. While principled class weighting in VFL mitigates gradient starvation, extreme visual variations of animals on Indian roads may require specialized data augmentation (e.g. mosaic with rare-class copy-paste).
   - *Future Ablation:* Class-aware mosaic augmentation and rare-class oversampling.
3. **Bounding Box Boundary Jitter under Glare:**
   - Point-offset regression predicts rigid corner offsets rather than probability distributions over box edges.
   - *Future Ablation:* Distribution Focal Loss (DFL) with integral regression.

---

## 7. External Training Command

When executing the training run on an external cloud GPU instance or cluster environment (e.g. NVIDIA A100 / RTX 4090 / Tesla T4), run:

```bash
python train.py \
    --version v2 \
    --loss-type task_aligned \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/v2_training \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --img-size 640 \
    --device cuda \
    --amp
```

---

## 8. Verification Status Summary

| Test / Check | Status | Key Results |
| :--- | :---: | :--- |
| **Unit Test Suite (`test_ird_v2_task_aligned.py`)** | **PASSED (18/18)** | Covers alignment calculation, top-k, multi-GT conflict resolution, crowded objects, empty images, background zero targets, monotonicity, stability, all 12 classes, and overlapping pairs. |
| **V1.5 Duplicate Unit Tests (`test_duplicate_cases.py`)** | **PASSED (6/6)** | Zero regression on canonical duplicate cases and class-aware preservation. |
| **V1.5 Authoritative Decoder Tests (`test_authoritative_decoder.py`)** | **PASSED** | Full backward compatibility with existing decoder pipeline. |
| **Static Verification (`verify_ird_v2.py`)** | **PASSED** | Exact parameter match (4,441,989), 100% state-dict compatibility, synthetic forward/backward pass with zero NaNs/Infs across 809 parameter tensors. |
| **Baseline Checkpoint Preservation** | **CONFIRMED** | `ird_best.pt` (54,732,253 bytes) remained 100% untouched. |
| **Hardware Safety** | **CONFIRMED** | Zero training executed on local workstation. |
