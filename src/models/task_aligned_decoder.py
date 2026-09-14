"""
Task-Aligned Post-Processing and Box Decoder for IRD V2 (IndianRoadDetector V2).

Replaces the flawed V1.5 confidence scoring:
    V1.5 (Flawed): Score = Cls * sqrt(Obj * Quality)
with the calibrated task-aligned formulation:
    V2 Task-Aligned: Score = Cls^alpha * Quality^beta  (default: alpha=1.0, beta=1.0)
    V2 Direct-Cls:   Score = Cls  (direct Varifocal continuous target)

Key Guarantees:
1. Zero Square-Root Noise Inflation: Eliminates the square-root function that
   amplified near-zero background logits into the 0.05-0.25 confidence range.
2. Explicit Background Quality Calibration: Relies on explicit zero-supervision
   for background quality, suppressing hallucinations.
3. Strict Class-Aware NMS: Uses class-spatial offsets (c * 10,000) with default
   IoU threshold = 0.40 to eliminate stride echoes while preserving valid
   co-occurrences (rider + motorcycle, person + vehicle).
4. Full Backward Compatibility: Works seamlessly with existing IndianRoadDetector
   head outputs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.head.custom_head import HeadOutput
from src.models.box_coder import (
    NUM_CLASSES,
    class_aware_nms,
    decode_boxes_smooth,
    pure_pytorch_nms,
)


def decode_ird_v2_predictions(
    head_output: HeadOutput,
    img_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.40,
    max_det: int = 300,
    score_mode: str = "task_aligned",
    score_alpha: float = 1.0,
    score_beta: float = 1.0,
    return_diagnostics: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """
    Authoritative Post-Processing and Decoder for IRD V2.

    Args:
        head_output: Output from IndianRoadDetector.head.
        img_size: Input image resolution (default: 640).
        conf_threshold: Minimum detection confidence threshold (default: 0.25).
        iou_threshold: Class-aware NMS IoU threshold (default: 0.40).
        max_det: Maximum detections allowed per image (default: 300).
        score_mode: Confidence computation mode:
          - "task_aligned": Score = Cls^alpha * Quality^beta (linear, uninflated)
          - "direct_cls":   Score = Cls (Varifocal continuous alignment score)
        score_alpha: Power factor for classification probability (default: 1.0).
        score_beta: Power factor for localization quality probability (default: 1.0).
        return_diagnostics: If True, returns auxiliary diagnostic metrics dict.

    Returns:
        Tuple of (boxes, scores, classes):
          - boxes:   [N, 4] Tensor of bounding boxes in (x1, y1, x2, y2)
          - scores:  [N] Tensor of confidence scores in [0, 1]
          - classes: [N] Tensor of class IDs in [0, 11] (torch.long)
        If return_diagnostics is True, returns (boxes, scores, classes, diag_dict).
    """
    device = head_output.box_preds[0].device
    strides = head_output.strides if hasattr(head_output, "strides") else [8, 16, 32]

    all_boxes_list: List[torch.Tensor] = []
    all_scores_list: List[torch.Tensor] = []
    all_classes_list: List[torch.Tensor] = []

    total_grid_cells = 0
    total_candidates = 0

    has_quality = (
        hasattr(head_output, "quality_preds")
        and head_output.quality_preds is not None
        and len(head_output.quality_preds) == len(strides)
    )

    for s_idx, stride in enumerate(strides):
        b_pred = head_output.box_preds[s_idx]  # [1, 4, H, W]
        c_pred = head_output.cls_preds[s_idx]  # [1, C, H, W]

        H, W = b_pred.shape[2], b_pred.shape[3]
        total_grid_cells += H * W

        # Grid Coordinates for all cell locations at this scale
        dev = b_pred.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=dev, dtype=torch.float32),
            torch.arange(W, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        grid_y_flat = grid_y.reshape(-1)
        grid_x_flat = grid_x.reshape(-1)

        # Raw predicted box parameters: [H*W, 4]
        b_flat = b_pred[0].permute(1, 2, 0).reshape(-1, 4)

        # Decode boxes using smooth parameterization
        decoded_boxes = decode_boxes_smooth(
            b_flat,
            grid_x_flat,
            grid_y_flat,
            stride=stride,
            version="v2_smooth",
        )

        # Strict boundary clamping to [0, img_size]
        x1 = decoded_boxes[:, 0].clamp(min=0.0, max=float(img_size))
        y1 = decoded_boxes[:, 1].clamp(min=0.0, max=float(img_size))
        x2 = decoded_boxes[:, 2].clamp(min=0.0, max=float(img_size))
        y2 = decoded_boxes[:, 3].clamp(min=0.0, max=float(img_size))
        clamped_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # Classification Probabilities: [H*W, num_classes]
        c_flat = c_pred[0].permute(1, 2, 0).reshape(-1, NUM_CLASSES)
        cls_prob = torch.sigmoid(c_flat)

        # Task-Aligned Scoring Computation
        if score_mode == "task_aligned" and has_quality and head_output.quality_preds[s_idx] is not None:
            q_flat = head_output.quality_preds[s_idx][0, 0].reshape(-1, 1)  # [H*W, 1]
            qual_prob = torch.sigmoid(q_flat)
            # Linear Task-Aligned Score: Cls^alpha * Quality^beta (No square-root!)
            comb_scores = (cls_prob.pow(score_alpha) * qual_prob.pow(score_beta)).clamp(min=0.0, max=1.0)
        else:
            # Direct classification score (Varifocal continuous target)
            comb_scores = cls_prob

        max_scores, class_ids = comb_scores.max(dim=-1)

        # Candidate Filter: score >= threshold and non-degenerate area
        valid = (max_scores >= conf_threshold) & ((x2 - x1) > 1.0) & ((y2 - y1) > 1.0)
        total_candidates += int(valid.sum().item())

        if valid.any():
            all_boxes_list.append(clamped_boxes[valid])
            all_scores_list.append(max_scores[valid])
            all_classes_list.append(class_ids[valid])

    if not all_boxes_list:
        empty_b = torch.empty((0, 4), device=device)
        empty_s = torch.empty((0,), device=device)
        empty_c = torch.empty((0,), dtype=torch.long, device=device)
        if return_diagnostics:
            return empty_b, empty_s, empty_c, {
                "raw_cells": total_grid_cells,
                "candidates": 0,
                "final_detections": 0,
            }
        return empty_b, empty_s, empty_c

    # Concatenate candidates across all scales
    cat_boxes = torch.cat(all_boxes_list, dim=0)
    cat_scores = torch.cat(all_scores_list, dim=0)
    cat_classes = torch.cat(all_classes_list, dim=0)

    # Class-Aware Greedy NMS
    keep_indices = class_aware_nms(
        boxes=cat_boxes,
        scores=cat_scores,
        class_ids=cat_classes,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )

    final_boxes = cat_boxes[keep_indices]
    final_scores = cat_scores[keep_indices]
    final_classes = cat_classes[keep_indices]

    if return_diagnostics:
        return final_boxes, final_scores, final_classes, {
            "raw_cells": total_grid_cells,
            "candidates": total_candidates,
            "final_detections": int(final_boxes.shape[0]),
        }

    return final_boxes, final_scores, final_classes
