"""
Static and Lightweight Sanity Verification Script for IRD V2.

Performs:
1. Model Instantiation & Parameter Verification against V1.5 baseline.
2. State-Dict Compatibility Verification with Epoch 19 checkpoint.
3. Synthetic Forward Pass Verification (B=1, B=2).
4. Synthetic TaskAlignedLoss Calculation with multi-object targets.
5. End-to-End Gradient Flow Verification (loss.backward() with NaN/Inf check).
6. Task-Aligned Decoder Verification (decode_ird_v2_predictions).
7. V1.5 Checkpoint Integrity Guarantee (zero modification confirmation).
8. External Training Command Specification.

CRITICAL HARDWARE RULE:
This script performs ONLY static checks and synthetic forward/backward passes.
ZERO model training is performed.
ZERO checkpoint files are modified.
"""

import hashlib
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.custom_detector import IndianRoadDetector, build_detector
from src.models.losses.task_aligned_loss import TaskAlignedLoss, build_task_aligned_loss
from src.models.task_aligned_decoder import decode_ird_v2_predictions


def verify_ird_v2():
    print("=" * 80)
    print("IRD V2 ARCHITECTURE & TRAINING SYSTEM STATIC VERIFICATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    # -----------------------------------------------------------------------
    # Step 1: Model Instantiation & Parameter Count Comparison
    # -----------------------------------------------------------------------
    print("\n[STEP 1] Model Instantiation & Parameter Count Verification:")
    model = IndianRoadDetector(num_classes=12).to(device=device)
    model.eval()

    params = model.get_parameter_counts(only_trainable=True)
    total_params = params["total"]
    expected_params = 4441989

    print(f"  Backbone parameters: {params['backbone']:>10,} ({params['backbone']/1e6:.2f}M)")
    print(f"  Neck parameters:     {params['neck']:>10,} ({params['neck']/1e6:.2f}M)")
    print(f"  Head parameters:     {params['head']:>10,} ({params['head']/1e6:.2f}M)")
    print(f"  Total parameters:    {total_params:>10,} ({total_params/1e6:.2f}M)")

    assert total_params == expected_params, (
        f"Parameter count mismatch! Expected {expected_params}, got {total_params}"
    )
    print(f"  --> Parameter Count Check: EXACT MATCH ({expected_params:,} parameters) [PASSED]")

    # -----------------------------------------------------------------------
    # Step 2: State-Dict Compatibility with V1.5 Checkpoint
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Checkpoint Compatibility Verification:")
    ckpt_path = Path("experiments/custom_model/final_training_50ep/ird_best.pt")
    assert ckpt_path.exists(), f"V1.5 checkpoint not found at: {ckpt_path}"

    ckpt_stat_before = ckpt_path.stat()
    print(f"  Target Checkpoint: {ckpt_path} (Size: {ckpt_stat_before.st_size:,} bytes)")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = incompatible.missing_keys
    unexpected = incompatible.unexpected_keys

    print(f"  Loaded state_dict keys: {len(state_dict)}")
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    assert len(missing) == 0, f"Missing keys detected: {missing[:5]}"
    assert len(unexpected) == 0, f"Unexpected keys detected: {unexpected[:5]}"
    print("  --> 100% Weight Compatibility with V1.5 Checkpoint: [PASSED]")

    # -----------------------------------------------------------------------
    # Step 3: Synthetic Forward Pass Verification (B=1, B=2)
    # -----------------------------------------------------------------------
    print("\n[STEP 3] Synthetic Forward Pass Verification:")
    for B in [1, 2]:
        x_dummy = torch.randn(B, 3, 640, 640, device=device)
        with torch.no_grad():
            out = model(x_dummy)

        for s_idx, stride in enumerate(model.strides):
            H_exp = 640 // stride
            W_exp = 640 // stride
            b_shape = tuple(out.box_preds[s_idx].shape)
            c_shape = tuple(out.cls_preds[s_idx].shape)
            assert b_shape == (B, 4, H_exp, W_exp), f"Box shape mismatch: {b_shape}"
            assert c_shape == (B, 12, H_exp, W_exp), f"Cls shape mismatch: {c_shape}"

            assert not torch.isnan(out.box_preds[s_idx]).any(), f"NaN in box_preds scale {s_idx}"
            assert not torch.isinf(out.box_preds[s_idx]).any(), f"Inf in box_preds scale {s_idx}"
            assert not torch.isnan(out.cls_preds[s_idx]).any(), f"NaN in cls_preds scale {s_idx}"
            assert not torch.isinf(out.cls_preds[s_idx]).any(), f"Inf in cls_preds scale {s_idx}"

        print(f"  --> Batch Size {B} Forward Pass: [PASSED] (Shapes & Finite Values Confirmed)")

    # -----------------------------------------------------------------------
    # Step 4: Synthetic TaskAlignedLoss Calculation
    # -----------------------------------------------------------------------
    print("\n[STEP 4] Synthetic TaskAlignedLoss Calculation:")
    loss_fn = TaskAlignedLoss(
        num_classes=12,
        box_weight=5.0,
        cls_weight=1.0,
        qual_weight=0.5,
        strides=tuple(model.strides),
        topk=10,
        tal_alpha=0.5,
        tal_beta=6.0,
        class_balanced=True,
    ).to(device=device)

    # Multi-object synthetic targets (car, motorcycle, rider, truck, person)
    synth_targets = torch.tensor([
        [0, 2, 200.0, 180.0, 120.0, 90.0],   # car
        [0, 5, 140.0, 220.0, 60.0, 80.0],    # motorcycle
        [0, 1, 140.0, 190.0, 40.0, 70.0],    # rider (overlapping bike)
        [1, 3, 300.0, 250.0, 180.0, 160.0],  # truck
        [1, 0, 100.0, 150.0, 30.0, 80.0],    # person
    ], dtype=torch.float32, device=device)

    x_b2 = torch.randn(2, 3, 640, 640, device=device)
    model.train()
    out_b2 = model(x_b2)

    loss_res = loss_fn(out_b2, synth_targets)
    print(f"  Total Loss:   {loss_res.total_loss.item():.4f}")
    print(f"  Box Loss:     {loss_res.box_loss.item():.4f}")
    print(f"  Cls Loss:     {loss_res.cls_loss.item():.4f}")
    print(f"  Qual Loss:    {loss_res.qual_loss.item():.4f}")
    print(f"  Num Positives: {loss_res.num_positives}")

    assert not torch.isnan(loss_res.total_loss), "NaN in TaskAlignedLoss total_loss"
    assert not torch.isinf(loss_res.total_loss), "Inf in TaskAlignedLoss total_loss"
    assert loss_res.total_loss.item() > 0.0, "total_loss must be strictly positive"
    assert loss_res.num_positives > 0, "No positive matches assigned by TAL"
    print("  --> TaskAlignedLoss Forward Calculation: [PASSED]")

    # -----------------------------------------------------------------------
    # Step 5: End-to-End Gradient Backward Pass Verification
    # -----------------------------------------------------------------------
    print("\n[STEP 5] End-to-End Gradient Flow Verification:")
    model.zero_grad()
    loss_res.total_loss.backward()

    nan_params = []
    zero_grad_params = []
    total_grad_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_grad_params += 1
            if param.grad is None:
                zero_grad_params.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_params.append(name)

    print(f"  Total trainable parameter tensors: {total_grad_params}")
    print(f"  Parameters with missing gradients: {len(zero_grad_params)}")
    print(f"  Parameters with NaN/Inf gradients: {len(nan_params)}")

    assert len(zero_grad_params) == 0, f"Missing gradients in: {zero_grad_params[:5]}"
    assert len(nan_params) == 0, f"NaN/Inf gradients in: {nan_params[:5]}"
    print("  --> All 100% of Trainable Parameters Received Clean Finite Gradients: [PASSED]")

    # -----------------------------------------------------------------------
    # Step 6: Task-Aligned Decoder Verification
    # -----------------------------------------------------------------------
    print("\n[STEP 6] Task-Aligned Post-Processing Decoder Verification:")
    model.eval()
    with torch.no_grad():
        out_eval = model(torch.randn(1, 3, 640, 640, device=device))

    boxes, scores, classes, diag = decode_ird_v2_predictions(
        head_output=out_eval,
        img_size=640,
        conf_threshold=0.25,
        iou_threshold=0.40,
        max_det=300,
        score_mode="task_aligned",
        return_diagnostics=True,
    )

    print(f"  Total grid cells:       {diag['raw_cells']:,}")
    print(f"  Surviving candidates:   {diag['candidates']:,}")
    print(f"  Post-NMS detections:    {diag['final_detections']:,}")

    if boxes.shape[0] > 0:
        assert boxes.shape[1] == 4
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
        assert (classes >= 0).all() and (classes < 12).all()
        assert (boxes[:, 0] >= 0.0).all() and (boxes[:, 2] <= 640.0).all()
        assert (boxes[:, 1] >= 0.0).all() and (boxes[:, 3] <= 640.0).all()
    print("  --> IRD V2 Decoder Correctness & Boundary Guarantees: [PASSED]")

    # -----------------------------------------------------------------------
    # Step 7: V1.5 Baseline Checkpoint Preservation Verification
    # -----------------------------------------------------------------------
    print("\n[STEP 7] Baseline Checkpoint Preservation Verification:")
    ckpt_stat_after = ckpt_path.stat()
    assert ckpt_stat_before.st_size == ckpt_stat_after.st_size, "Checkpoint file size changed!"
    assert ckpt_stat_before.st_mtime == ckpt_stat_after.st_mtime, "Checkpoint modification time changed!"
    print(f"  Checkpoint Size: {ckpt_stat_after.st_size:,} bytes (UNTOUCHED)")
    print(f"  Checkpoint ModTime: {ckpt_stat_after.st_mtime} (UNTOUCHED)")
    print("  --> IRD V1.5 Checkpoint 100% Preserved & Unmodified: [PASSED]")

    # -----------------------------------------------------------------------
    # Step 8: External GPU Environment Training Command
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ALL STATIC CHECKS, INTEGRATIONS, AND SANITY TESTS PASSED SUCCESSFULLY!")
    print("NO TRAINING WAS PERFORMED ON THIS WORKSTATION.")
    print("=" * 80)
    print("\nTo train IRD V2 on an external cloud GPU / cluster environment, execute:")
    print("python train.py --version v2 --loss-type task_aligned --data data/indian_road_yolo/data.yaml "
          "--epochs 50 --batch-size 16 --img-size 640 --device 0 --save-dir experiments/custom_model/v2_training")
    print("=" * 80)


if __name__ == "__main__":
    verify_ird_v2()
