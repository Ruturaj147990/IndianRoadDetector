# IRD V2 External GPU Training & Evaluation Runbook

**Document:** External GPU Setup, Execution, and Verification Guide  
**Target Model:** IRD V2 (IndianRoadDetection V2)  
**Task:** Task-Aligned Assignment (TAL) + Varifocal Continuous Alignment Loss (VFL)  
**Baseline Reference (V1.5 Authoritative):**  
- **mAP50:** `0.2911` (29.11%)  
- **mAP50-95:** `0.1907` (19.07%)  
- **Evaluator:** 1,719 images, 8,757 GT boxes on corrected clip-disjoint split  
- **Notice:** These are the authoritative baseline figures. **Do not claim V2 improvements until external training and evaluation are completed and measured.**

---

## 1. Environment Installation (External Machine / Cloud VM)

On your external cloud GPU instance (e.g. Lambda Labs, RunPod, AWS EC2, GCP, or local multi-GPU workstation with NVIDIA A100, H100, RTX 4090, or Tesla T4/V100):

```bash
# 1. Clone the repository
git clone https://github.com/Ruturaj147990/IndianRoadDetector.git
cd IndianRoadDetector

# 2. Set up Python virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install PyTorch with appropriate CUDA support
# Example for CUDA 12.1+:
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install repository requirements
pip install -r requirements.txt
```

---

## 2. Dataset Placement & Verification

The Indian Road Dataset follows the standard YOLO detection directory structure.

### Placement
Ensure the dataset is located at `data/indian_road_yolo/`:
```text
data/indian_road_yolo/
├── data.yaml
├── images/
│   ├── train/     # 8,282 images
│   └── val/       # 1,719 images (corrected split)
└── labels/
    ├── train/     # 8,282 label txt files
    └── val/       # 1,719 label txt files
```

### Verification Command
Run the dataset integrity audit script to confirm zero missing files, 0% clip leakage, and all 12 classes populated:

```bash
python scripts/audit_dataset.py --data-dir data/indian_road_yolo
```

Expected validation output:
- **Train images:** 8,282
- **Validation images:** 1,719
- **Classes:** 12 (`person`, `rider`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `autorickshaw`, `animal`, `vehicle fallback`, `traffic light`, `traffic sign`)
- **Clip overlap:** 0.0%

---

## 3. GPU Verification & Architecture Sanity

Before launching long training jobs, execute the static verification script on the target GPU. This validates tensor shapes, forward/backward gradients, and 100% parameter compatibility without training:

```bash
python scripts/verify_ird_v2.py
```

Expected output:
- **Total parameters:** `4,441,989` (Exact match with V1.5)
- **Gradient flow:** 809 parameter tensors checked with 0 NaN/Inf and 0 missing gradients
- **Decoder checks:** PASSED

Also execute the unit test suite:
```bash
python tests/test_ird_v2_task_aligned.py
python tests/test_duplicate_cases.py
```
All 18/18 V2 tests and 6/6 duplicate suppression tests must report `[PASS]`.

---

## 4. Batch-Size Calibration Procedure

> [!IMPORTANT]
> **Do NOT assume a fixed batch size.**  
> Batch size determines throughput, gradient stability, and GPU memory saturation. Calibrate safe batch sizes on your external GPU before launching the 50-epoch run.

Run a lightweight calibration check:
```bash
python scripts/calibrate_vram.py --data-dir data/indian_road_yolo --img-size 640
```
Or test batch size limits manually using the smoke-test flag:
```bash
# Test Batch Size 16 (Typical for 16GB-24GB GPUs like RTX 4090 / A5000 / V100):
python train.py --version v2 --loss-type task_aligned --batch-size 16 --smoke-test

# If Out Of Memory (OOM), test Batch Size 12 or 8 (Safe for 12GB GPUs / T4):
python train.py --version v2 --loss-type task_aligned --batch-size 12 --smoke-test
```

**Selection Rule:** Choose the largest batch size that leaves at least **1.5 GB to 2.0 GB of VRAM headroom** to avoid mid-training OOM during peak augmentation steps.

---

## 5. The Final IRD V2 Training Command

Once the batch size `BATCH_SIZE` is chosen, launch the official 50-epoch IRD V2 training job:

```bash
python train.py \
    --version v2 \
    --loss-type task_aligned \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/v2_training \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --img-size 640 \
    --device auto \
    --amp
```

*(Replace `--batch-size 16` with your calibrated safe batch size if different).*

### Hyperparameter Summary:
- `--version v2`: Activates IRD V2 architecture hooks.
- `--loss-type task_aligned`: Activates `TaskAlignedLoss` (TAL dynamic matcher + Varifocal Loss).
- `--epochs 50`: Standard convergence schedule.
- `--lr 0.001`: AdamW initial learning rate with Cosine Annealing.
- `--weight-decay 0.0001`: Regularization parameter.
- `--img-size 640`: Standard $640 \times 640$ spatial resolution.
- `--amp`: Automatic Mixed Precision enabled for speed and memory efficiency.

---

## 6. Checkpoint Locations & Artifacts

All training artifacts will be saved in `--output-dir experiments/custom_model/v2_training/`:

- **Best Model Weights:** `experiments/custom_model/v2_training/ird_best.pt`
- **Latest Epoch Weights:** `experiments/custom_model/v2_training/ird_last.pt`
- **Epoch Training Log (CSV):** `experiments/custom_model/v2_training/ird_history.csv`
- **Full History JSON:** `experiments/custom_model/v2_training/ird_history.json`
- **Run Configuration:** `experiments/custom_model/v2_training/ird_config.json`

---

## 7. Resume Procedure (In Case of Interruption)

If the cloud VM pre-empts or training disconnects, resume seamlessly from the last saved state:

```bash
python train.py \
    --version v2 \
    --loss-type task_aligned \
    --data-dir data/indian_road_yolo \
    --output-dir experiments/custom_model/v2_training \
    --resume experiments/custom_model/v2_training/ird_last.pt \
    --epochs 50 \
    --batch-size 16 \
    --amp
```

The resume logic automatically restores the model weights, optimizer state, LR scheduler cycle, GradScaler state, and starting epoch index.

---

## 8. Authoritative Evaluation Procedure

Once training completes, evaluate the newly trained `ird_best.pt` on the authoritative 1,719-image validation set using the official COCO 10-IoU evaluation suite with IRD V2 task-aligned decoding:

```bash
python scripts/evaluate_ird.py \
    --checkpoint experiments/custom_model/v2_training/ird_best.pt \
    --data-dir data/indian_road_yolo \
    --split val \
    --conf-thresh 0.25 \
    --iou-thresh 0.40 \
    --max-det 300 \
    --score-mode task_aligned \
    --save-json experiments/evaluation_results/ird_v2_best_eval.json
```

---

## 9. V1.5 vs V2 Comparison Procedure

To compare the newly trained V2 model directly against the V1.5 baseline:

1. **Compare Summary Metrics:**
   - **V1.5 Baseline:** mAP50 = `0.2911`, mAP50-95 = `0.1907`
   - **V2 Output:** Check `experiments/evaluation_results/ird_v2_best_eval.json`

2. **Run Deep Diagnostic Comparison (Optional):**
   ```bash
   python scripts/run_deep_error_analysis.py \
       --checkpoint experiments/custom_model/v2_training/ird_best.pt \
       --score-mode task_aligned
   ```
   Inspect the confusion matrix and false positive count to verify that the 506k background false alarms observed in V1.5 have been suppressed.
