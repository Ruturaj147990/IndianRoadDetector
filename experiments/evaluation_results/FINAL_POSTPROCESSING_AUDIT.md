# Final IRD V1.5 Post-Processing Validation & Evaluator Audit Report

**Model Designation:** `IndianRoadDetector` (IRD V1.5 / IRD-Next)  
**Evaluation Checkpoint:** `experiments/custom_model/final_training_50ep/ird_best.pt` (Epoch 19 / Best Val Metric Checkpoint)  
**Evaluated Scope:** Full Validation Dataset (1,719 images, 100% complete)  
**Hardware Platform:** AMD Radeon RX 7700 XT (ROCm PyTorch 2.9.1 / ROCm 7.2)  
**Constraints Adherence:** Zero retraining, zero architectural modifications, zero weight changes, zero loss changes, zero benchmark overwriting.

---

## 1. Executive Summary

This final audit completes the post-processing validation and evaluator diagnostic investigation across all **1,719 images** of the validation set:

1. **Production Inference Configuration Validated**:
   - The default class-aware NMS IoU threshold is set to **`0.40`** in `src/models/box_coder.py`.
   - Confidence threshold is maintained at **`0.25`**.
   - Class-aware spatial separation is strictly preserved.
   - All 6 canonical duplicate unit tests passed with 100% accuracy (`tests/test_duplicate_cases.py`).
   - All authoritative decoder unit tests passed (`tests/test_authoritative_decoder.py`).
   - On the full 1,719 validation images, production inference outputs an average of **5.41 detections / image** (maximum 24, minimum 0), eliminating ceiling saturation, achieving **32.93% precision**, **mAP50 = 0.2647**, and **mAP50-95 = 0.1788**.

2. **Evaluator `max_det` Saturation Diagnosis on Full 1,719 Images**:
   - `max_det` was systematically varied across `[100, 300, 500, 1000, 2000, 5000]` at the authoritative evaluation threshold (`conf_threshold = 0.001`, `iou_threshold = 0.50`).
   - **Is `max_det=300` materially truncating useful detections?**
     **NO.** Expanding `max_det` from 300 all the way to 5,000 (which captures all surviving candidates across the entire network) dumps **3,117,197 extra predictions** (an average of 2,113.4 detections/image), but changes mAP50 by only **+0.0002** (0.2911 $\rightarrow$ 0.2913) and changes mAP50-95 by **0.0000** (0.1907 $\rightarrow$ 0.1907).
   - The extra 3.1 million predictions consist of **99.8% background noise** (precision collapses to 0.19%).
   - The 300-detection ceiling does **not** distort the benchmark mAP metric.

3. **Confidence Score Distribution Insights**:
   - Across the 1,719 images, the median candidate score is **0.0374**.
   - **95% of all candidates have confidence $< 0.108$**.
   - Only the top **1% (P99+)** have confidence $> 0.813$.
   - While an average of 2,113.4 candidates exist above 0.001, only **8.56 detections / image** exist above 0.25, and only **2.87 detections / image** exist above 0.50.

---

## 2. Confidence Score Distribution Analysis

A global statistical analysis was conducted on all decoded candidate predictions prior to confidence filtering and NMS:

### 2.1 Score Percentile Distribution
| Percentile | Score Value | Significance |
|:---:|:---:|---|
| **P10** | `0.0137` | 10% of candidates have score $< 0.0137$ |
| **P25** | `0.0231` | Lower quartile |
| **P50 (Median)** | `0.0374` | **Median candidate score is only 3.74%** |
| **P75** | `0.0593` | 75% of candidates have score $< 0.0593$ |
| **P90** | `0.0880` | 90% of candidates have score $< 0.0880$ |
| **P95** | `0.1085` | **95% of all candidates have score $< 0.1085$** |
| **P99** | `0.8132` | Top 1% boundary jumps sharply to 81.3% |
| **P99.5** | `0.8869` | High-confidence true positives |
| **P99.9** | `0.9296` | Peak confidence detections |

### 2.2 Detections Per Image Above Confidence Thresholds
| Confidence Floor | Average Dets / Image | Median Dets / Image | Max Dets / Image | Total Predictions (1,719 Images) |
|:---:|:---:|:---:|:---:|:---:|
| **$\ge 0.001$** | **2,113.38** | 2,110.0 | 2,378 | 3,632,897 |
| **$\ge 0.005$** | 2,060.77 | 2,062.0 | 2,348 | 3,542,466 |
| **$\ge 0.010$** | 1,991.67 | 1,992.0 | 2,289 | 3,423,687 |
| **$\ge 0.020$** | 1,693.71 | 1,696.0 | 2,012 | 2,911,480 |
| **$\ge 0.050$** | **691.91** | 578.0 | 1,432 | 1,189,396 |
| **$\ge 0.100$** | **120.04** | 96.0 | 697 | 206,349 |
| **$\ge 0.250$** (Production) | **8.56** | 8.0 | 34 | 14,716 |
| **$\ge 0.500$** | **2.87** | 2.0 | 13 | 4,930 |

*Analysis:*
The steep drop from **2,113.4** candidates at 0.001 down to **8.56** candidates at 0.25 mathematically illustrates the low-confidence tail phenomenon. Because the network was trained with BCE loss on unpenalized quality heads, background cells have an uncalibrated score floor between 0.01 and 0.05. Setting the production threshold to 0.25 filters out 99.6% of non-object candidates.

---

## 3. Full-Dataset `max_det` Evaluator Audit (1,719 Images)

All 1,719 images were evaluated under the official authoritative evaluator baseline (`conf_threshold = 0.001`, `iou_threshold = 0.50`), varying only `max_det`:

| `max_det` | Total Detections | Avg Dets / Img | Max Dets / Img | Precision | Recall | mAP50 | mAP50-95 | Eval Latency (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **100** | 171,900 | 100.0 | 100 (Sat) | **0.0296** | 0.4811 | 0.2908 | 0.1906 | 1.3s |
| **300** *(Official)* | 515,700 | 300.0 | 300 (Sat) | 0.0124 | 0.5133 | **0.2911** | **0.1907** | 2.7s |
| **500** | 859,500 | 500.0 | 500 (Sat) | 0.0085 | 0.5292 | **0.2912** | **0.1907** | 4.9s |
| **1000** | 1,719,000 | 1000.0 | 1000 (Sat) | 0.0049 | 0.5506 | **0.2913** | **0.1907** | 9.9s |
| **2000** | 3,431,367 | 1996.1 | 2000 (Sat) | 0.0021 | 0.5548 | **0.2913** | **0.1907** | 17.7s |
| **5000** *(Unbounded)*| 3,632,897 | 2113.4 | 2378 | 0.0019 | 0.5548 | **0.2913** | **0.1907** | 19.6s |

---

### 3.1 Per-Class AP50 Breakdown Across `max_det` Values

| Class | `max_det=100` | `max_det=300` *(Official)* | `max_det=500` | `max_det=1000` | `max_det=2000` | `max_det=5000` | Delta (300 $\rightarrow$ 5000) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`person`** | 0.2460 | 0.2463 | 0.2464 | 0.2465 | 0.2465 | 0.2465 | **+0.0002** |
| **`rider`** | 0.5034 | 0.5034 | 0.5039 | 0.5039 | 0.5039 | 0.5039 | **+0.0005** |
| **`car`** | 0.8427 | 0.8430 | 0.8430 | 0.8434 | 0.8434 | 0.8434 | **+0.0004** |
| **`truck`** | 0.1744 | 0.1749 | 0.1750 | 0.1751 | 0.1751 | 0.1751 | **+0.0002** |
| **`bus`** | 0.2756 | 0.2757 | 0.2758 | 0.2759 | 0.2759 | 0.2759 | **+0.0002** |
| **`motorcycle`** | 0.5504 | 0.5507 | 0.5507 | 0.5509 | 0.5509 | 0.5509 | **+0.0002** |
| **`bicycle`** | 0.3787 | 0.3794 | 0.3795 | 0.3796 | 0.3796 | 0.3796 | **+0.0002** |
| **`autorickshaw`** | 0.4498 | 0.4501 | 0.4500 | 0.4500 | 0.4501 | 0.4501 | **0.0000** |
| **`animal`** | 0.0009 | 0.0009 | 0.0009 | 0.0009 | 0.0009 | 0.0009 | **0.0000** |
| **`vehicle fallback`**| 0.0520 | 0.0524 | 0.0523 | 0.0524 | 0.0525 | 0.0525 | **+0.0001** |
| **`traffic light`** | 0.0000 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | **-0.0001** |
| **`traffic sign`** | 0.0162 | 0.0163 | 0.0163 | 0.0164 | 0.0164 | 0.0164 | **+0.0001** |
| **Mean (mAP50)** | **0.2908** | **0.2911** | **0.2912** | **0.2913** | **0.2913** | **0.2913** | **+0.0002** |

---

## 4. Key Determinations & Answers to Audit Questions

### 4.1 Does `max_det=300` Materially Truncate Useful Detections?
**NO.**
- Increasing `max_det` from 300 to 5,000 adds **3,117,197 additional predictions** across the 1,719 validation images (+604% more boxes).
- The resulting change in mAP50 is only **+0.0002** (from 0.2911 to 0.2913).
- The resulting change in mAP50-95 is **0.0000** (exactly 0.1907).
- Although recall nominally rises from 51.3% to 55.5% (+4.2%), precision collapses from 1.24% to **0.19%**, indicating that over 99.8% of the extra 3.1 million candidates are false positive noise.
- Therefore, `max_det=300` is **NOT** clipping meaningful object detections in any statistically significant way for mAP calculation.

### 4.2 Why Did the Evaluator Saturate at 300?
The evaluator saturates at 300 because:
1. `conf_threshold = 0.001` permits any grid cell with a non-zero activation to become a candidate.
2. The $\sqrt{\text{Obj} \times \text{Quality}}$ score formulation inflates background probabilities by $10\times$.
3. An average of **2,113 candidate boxes** survive class-aware NMS per image when unconstrained.
4. Because the standard COCO evaluator has an explicit cap of `max_det = 300`, the loop halts at exactly 300.
5. In production inference where `conf_threshold = 0.25`, the saturation disappears: images average **5.41 detections / image**, and never exceed 24 detections.

---

## 5. Three-Way Comparison: Current Evaluator vs `max_det` Diagnostic vs Production

| Metric | A. Current Official Evaluator | B. Unbounded `max_det=5000` Diagnostic | C. Production Config (`IoU=0.40`, `conf=0.25`) |
|---|:---:|:---:|:---:|
| **Confidence Threshold** | `0.001` | `0.001` | **`0.25`** |
| **NMS IoU Threshold** | `0.50` | `0.50` | **`0.40`** |
| **`max_det` Ceiling** | `300` | `5000` | **`300`** |
| **Total Detections (1,719 imgs)**| 515,700 | 3,632,897 | **9,306** |
| **Average Detections / Image** | 300.0 | 2,113.4 | **5.41** |
| **Maximum Detections / Image** | 300 (100% Sat) | 2,378 | **24** (0% Sat) |
| **Precision** | 0.0124 (1.24%) | 0.0019 (0.19%) | **0.3293 (32.93%)** |
| **Recall** | 0.5133 (51.33%) | 0.5548 (55.48%) | **0.3208 (32.08%)** |
| **mAP50** | **0.2911** | **0.2913** | **0.2647** |
| **mAP50-95** | **0.1907** | **0.1907** | **0.1788** |
| **Visual Quality & Overlap** | Dense overlapping noise | Extreme canvas clutter | **Crisp, non-overlapping HUD boxes** |

---

## 6. Official Recommendations & Evaluation Standard

1. **Production Inference Setting (Adopted)**:
   - Use **`iou_threshold = 0.40`** (now default in `src/models/box_coder.py`).
   - Use **`conf_threshold = 0.25`**.
   - Keep class-aware spatial separation enabled.
   - Result: 5.4 detections/image, 32.9% precision, clean non-duplicate rendering, preserving rider+motorcycle co-occurrences.

2. **Benchmark Evaluation Standard**:
   - The official authoritative benchmark mAP is **mAP50 = 0.2911**, **mAP50-95 = 0.1907** (with `max_det=300`, `conf=0.001`, `iou=0.50`).
   - **Do NOT overwrite or replace official benchmark results** with production numbers (`conf=0.25`), because COCO AP is defined by integrating the entire recall curve down to zero confidence.
   - The diagnostic evaluation proves that `max_det=300` does **not** harm AP (delta is only +0.0002 at `max_det=5000`), so `max_det=300` remains fully sound as an authoritative benchmark standard.

---

## 7. Artifacts Summary

- **Final Structured JSON:** [`experiments/evaluation_results/final_postprocessing_audit.json`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/experiments/evaluation_results/final_postprocessing_audit.json)
- **Final Markdown Report:** [`experiments/evaluation_results/FINAL_POSTPROCESSING_AUDIT.md`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/experiments/evaluation_results/FINAL_POSTPROCESSING_AUDIT.md)
- **Previous Audit Report:** [`experiments/evaluation_results/DUPLICATE_SUPPRESSION_AUDIT.md`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/experiments/evaluation_results/DUPLICATE_SUPPRESSION_AUDIT.md)
- **Unit Test Suite:** [`tests/test_duplicate_cases.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/tests/test_duplicate_cases.py)
- **Full Evaluator Script:** [`scripts/audit_final_evaluator.py`](file:///c:/Users/limbk/OneDrive/Desktop/YOLO/IndianRoadDetector/scripts/audit_final_evaluator.py)
