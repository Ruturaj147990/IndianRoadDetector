"""
Authoritative Shared Box Decoder and Coordinate Geometry Engine for IRD V1.

Guarantees:
1. Exact mathematical identity across training loss, evaluation, and inference.
2. Smooth, non-saturating bounding-box dimension parameterization (zero gradient death).
3. Early objectness gating in logit space (--obj-gate) for maximum efficiency.
4. Class-aware Non-Maximum Suppression with max_det enforcement.
5. Numerical stability and strict boundary clamping to [0, img_size].
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 12


def decode_boxes_smooth(
    raw_box: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    stride: int,
    version: str = "v2_smooth",
) -> torch.Tensor:
    """
    Decode raw box regression parameters at given grid positions into (x1, y1, x2, y2).
    
    Formulations:
      v2_smooth (Recommended):
        cx = (grid_x + 2.0 * sigmoid(tx) - 0.5) * stride
        cy = (grid_y + 2.0 * sigmoid(ty) - 0.5) * stride
        w  = stride * exp(3.0 * tanh(tw / 3.0))
        h  = stride * exp(3.0 * tanh(th / 3.0))
        -> Derivative is strictly positive everywhere (no gradient saturation).
        
      v1_legacy (for reproducing overfit_test.pt):
        cx = (grid_x + 2.0 * sigmoid(tx) - 0.5) * stride
        cy = (grid_y + 2.0 * sigmoid(ty) - 0.5) * stride
        w  = stride * exp(clamp(tw, -4.0, 4.0))
        h  = stride * exp(clamp(th, -4.0, 4.0))
    """
    tx = raw_box[..., 0]
    ty = raw_box[..., 1]
    tw = raw_box[..., 2]
    th = raw_box[..., 3]

    grid_x = grid_x.to(device=raw_box.device, dtype=raw_box.dtype)
    grid_y = grid_y.to(device=raw_box.device, dtype=raw_box.dtype)

    cx = (grid_x + torch.sigmoid(tx) * 2.0 - 0.5) * stride
    cy = (grid_y + torch.sigmoid(ty) * 2.0 - 0.5) * stride

    if version == "v1_legacy":
        w = stride * torch.exp(tw.clamp(min=-4.0, max=4.0))
        h = stride * torch.exp(th.clamp(min=-4.0, max=4.0))
    else:  # v2_smooth
        w = stride * torch.exp(3.0 * torch.tanh(tw / 3.0))
        h = stride * torch.exp(3.0 * torch.tanh(th / 3.0))

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return torch.stack([x1, y1, x2, y2], dim=-1)


def pure_pytorch_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    max_det: int = 300,
) -> torch.Tensor:
    """Pure PyTorch NMS with max_det ceiling."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if len(keep) >= max_det or order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = (ovr <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.50,
    max_det: int = 300,
    max_coordinate: float = 10000.0,
) -> torch.Tensor:
    """Class-aware NMS applying spatial offsets per class ID."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    offsets = class_ids.float().unsqueeze(1) * max_coordinate
    offset_boxes = boxes + offsets
    return pure_pytorch_nms(offset_boxes, scores, iou_threshold=iou_threshold, max_det=max_det)


def decode_ird_predictions_authoritative(
    head_output: Any,
    img_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    max_det: int = 300,
    obj_gate: Optional[float] = None,
    decoder_version: str = "v2_smooth",
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Authoritative decoding pipeline for IRD multi-scale predictions.
    
    Features:
    1. Early Objectness Gating: Skips impossible candidate cells in logit space.
    2. Shared decode_boxes_smooth geometry engine.
    3. Strict boundary clamping to [0, img_size].
    4. Class-aware NMS with max_det enforcement.
    5. Post-NMS safety verification.
    """
    all_boxes_list: List[torch.Tensor] = []
    all_scores_list: List[torch.Tensor] = []
    all_classes_list: List[torch.Tensor] = []

    box_preds = head_output.box_preds
    obj_preds = head_output.obj_preds
    cls_preds = head_output.cls_preds
    strides = head_output.strides

    # Determine objectness gate threshold in logit space
    # If conf_threshold is 0.25 and obj_gate is None, obj_gate defaults to conf_threshold
    effective_gate = obj_gate if obj_gate is not None else conf_threshold
    gate_logit = math.log(effective_gate / max(1e-6, 1.0 - effective_gate)) if 0.0 < effective_gate < 1.0 else -10.0

    for s_idx, stride in enumerate(strides):
        b_pred = box_preds[s_idx][0]  # [4, H, W]
        o_pred = obj_preds[s_idx][0]  # [1, H, W]
        c_pred = cls_preds[s_idx][0]  # [12, H, W]

        H, W = b_pred.shape[1], b_pred.shape[2]

        # 1. Early Objectness Gating in logit space
        o_logit_flat = o_pred[0].reshape(-1)  # [H*W]
        surviving_mask = o_logit_flat >= gate_logit

        if not surviving_mask.any():
            continue

        surviving_indices = surviving_mask.nonzero(as_tuple=False).squeeze(1)

        device = b_pred.device
        # 2. Grid Coordinates for surviving locations only
        grid_y = (surviving_indices // W).to(dtype=torch.float32, device=device)
        grid_x = (surviving_indices % W).to(dtype=torch.float32, device=device)

        # Extract surviving raw boxes: [K, 4]
        b_flat = b_pred.permute(1, 2, 0).reshape(-1, 4)
        raw_boxes_surviving = b_flat[surviving_indices]

        # 3. Decode surviving boxes using authoritative engine
        decoded_boxes = decode_boxes_smooth(
            raw_boxes_surviving,
            grid_x,
            grid_y,
            stride=stride,
            version=decoder_version,
        )

        # 4. Strict boundary clamping
        x1 = decoded_boxes[:, 0].clamp(min=0.0, max=float(img_size))
        y1 = decoded_boxes[:, 1].clamp(min=0.0, max=float(img_size))
        x2 = decoded_boxes[:, 2].clamp(min=0.0, max=float(img_size))
        y2 = decoded_boxes[:, 3].clamp(min=0.0, max=float(img_size))
        clamped_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # 5. Probabilities
        obj_prob = torch.sigmoid(o_logit_flat[surviving_indices]).unsqueeze(1)  # [K, 1]
        c_flat = c_pred.permute(1, 2, 0).reshape(-1, NUM_CLASSES)
        cls_prob = torch.sigmoid(c_flat[surviving_indices])  # [K, 12]

        comb_scores = obj_prob * cls_prob  # [K, 12]
        max_scores, class_ids = comb_scores.max(dim=-1)

        # 6. Candidate Filter
        valid = (max_scores >= conf_threshold) & ((x2 - x1) > 1.0) & ((y2 - y1) > 1.0)

        if valid.any():
            all_boxes_list.append(clamped_boxes[valid])
            all_scores_list.append(max_scores[valid])
            all_classes_list.append(class_ids[valid])

    if not all_boxes_list:
        empty = torch.empty((0,), device=device)
        return torch.empty((0, 4), device=device), empty, torch.empty((0,), dtype=torch.long, device=device)

    all_boxes = torch.cat(all_boxes_list, dim=0)
    all_scores = torch.cat(all_scores_list, dim=0)
    all_classes = torch.cat(all_classes_list, dim=0)

    # 7. Class-aware NMS
    keep = class_aware_nms(
        all_boxes,
        all_scores,
        all_classes,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )

    retained_boxes = all_boxes[keep]
    retained_scores = all_scores[keep]
    retained_classes = all_classes[keep]

    # 8. Post-NMS Safety Verification
    safety_mask = retained_scores >= conf_threshold
    return retained_boxes[safety_mask], retained_scores[safety_mask], retained_classes[safety_mask]
