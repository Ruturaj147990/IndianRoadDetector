"""
Task-Aligned Detection Loss for IRD V2 (IndianRoadDetector V2).

Combines:
1. Task-Aligned Assignor (TAL): Dynamic positive sample matching based on joint
   classification and localization alignment: t = s^alpha * IoU^beta.
2. Varifocal Classification Loss (VFL): Continuous IoU-aware focal loss that provides:
   - IoU-quality regression on positive samples (q > 0)
   - Heavy focal downweighting and explicit suppression on background samples (q = 0)
3. Quality-Weighted CIoU Bounding Box Loss: Downweights poorly localized candidates
   while giving high gradient weight to tightly aligned boxes.
4. Explicit Background Quality Supervision: Background anchors receive continuous
   targets strictly equal to 0.0, eliminating uncalibrated background logit drift.
5. Principled Class-Imbalance Handling: Configurable class weighting for rare categories
   (animal, traffic light, traffic sign, vehicle fallback).
"""

import math
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.head.custom_head import HeadOutput
from src.models.box_coder import decode_boxes_smooth
from src.models.losses.custom_loss import bbox_ciou
from src.models.losses.task_aligned_assignor import (
    TaskAlignedAssignor,
    box_iou_pairwise,
    generate_anchor_grid,
)


class TaskAlignedLossResult(NamedTuple):
    """Structured container for IRD V2 loss outputs."""
    total_loss: torch.Tensor
    box_loss: torch.Tensor
    cls_loss: torch.Tensor
    qual_loss: torch.Tensor
    obj_loss: torch.Tensor
    num_positives: int

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([self.total_loss, self.box_loss, self.cls_loss, self.qual_loss, self.obj_loss])


def varifocal_loss(
    pred_logits: torch.Tensor,
    target_scores: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Varifocal Loss (VFL) for training dense object detectors with continuous IoU targets.

    Formula:
        For q > 0: -q * (q * log(p) + (1 - q) * log(1 - p))
        For q = 0: -alpha * p^gamma * log(1 - p)

    Args:
        pred_logits: Raw prediction classification logits [B, N, C].
        target_scores: Continuous alignment targets [B, N, C] in [0, 1].
        alpha: Negative sample scaling factor (default: 0.75).
        gamma: Focusing parameter for negative samples (default: 2.0).
        class_weights: Optional class weighting tensor [C].

    Returns:
        Scalar loss tensor (sum over elements).
    """
    pred_prob = torch.sigmoid(pred_logits)
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")

    # Weight factors:
    # For positives (q > 0): weight = q
    # For negatives (q = 0): weight = alpha * (pred_prob ^ gamma)
    pos_mask = target_scores > 0.0
    weight = torch.where(
        pos_mask,
        target_scores,
        alpha * pred_prob.pow(gamma),
    )

    loss = weight * bce_loss

    if class_weights is not None:
        # Apply class weights: [1, 1, C]
        cw = class_weights.unsqueeze(0).unsqueeze(0).to(dtype=loss.dtype, device=loss.device)
        # Apply class weights only to positive targets, keep background uniformly suppressed
        pos_weight = torch.where(pos_mask, cw, torch.ones_like(cw))
        loss = loss * pos_weight

    return loss.sum()


class TaskAlignedLoss(nn.Module):
    """
    Complete Task-Aligned Detection Loss for IRD V2.

    Replaces the static center-distance assignment, uncalibrated quality branch,
    and disconnected objectness loss of IRD V1.5.

    Args:
        num_classes: Number of detection classes (default: 12).
        box_weight: Weight multiplier for bounding box CIoU loss (default: 5.0).
        cls_weight: Weight multiplier for Varifocal classification loss (default: 1.0).
        qual_weight: Weight multiplier for explicit quality supervision (default: 0.5).
        strides: Multi-scale detection strides (default: (8, 16, 32)).
        topk: Number of candidates per ground truth in TAL (default: 10).
        tal_alpha: Power factor for classification score in TAL (default: 0.5).
        tal_beta: Power factor for IoU in TAL (default: 6.0).
        vfl_alpha: Scaling factor for negative samples in Varifocal Loss (default: 0.75).
        vfl_gamma: Focusing parameter for negative samples in Varifocal Loss (default: 2.0).
        class_balanced: Whether to apply principled class weighting to rare classes (default: True).
    """
    def __init__(
        self,
        num_classes: int = 12,
        box_weight: float = 5.0,
        cls_weight: float = 1.0,
        qual_weight: float = 0.5,
        obj_weight: float = 1.0,
        strides: Tuple[int, int, int] = (8, 16, 32),
        topk: int = 10,
        tal_alpha: float = 0.5,
        tal_beta: float = 6.0,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        class_balanced: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.qual_weight = qual_weight
        self.obj_weight = obj_weight
        self.strides = list(strides)
        self.topk = topk
        self.tal_alpha = tal_alpha
        self.tal_beta = tal_beta
        self.vfl_alpha = vfl_alpha
        self.vfl_gamma = vfl_gamma
        self.class_balanced = class_balanced

        # Task-Aligned Assignor instance
        self.assignor = TaskAlignedAssignor(
            topk=topk,
            num_classes=num_classes,
            alpha=tal_alpha,
            beta=tal_beta,
        )

        # Principled class weights (normalized inverse frequencies based on 1,719 validation distribution)
        # Class order: [person, rider, car, truck, bus, motorcycle, bicycle, autorickshaw, animal, veh_fallback, traf_light, traf_sign]
        if class_balanced:
            self.register_buffer(
                "class_weights",
                torch.tensor([2.0, 1.5, 1.0, 2.5, 2.5, 1.5, 2.5, 2.0, 3.0, 2.5, 3.0, 2.5], dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: Union[HeadOutput, Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]],
        targets: Union[torch.Tensor, List[torch.Tensor]],
        img_size: Tuple[int, int] = (640, 640),
    ) -> TaskAlignedLossResult:
        """
        Compute total multi-task detection loss for IRD V2.

        Args:
            predictions: HeadOutput instance or tuple of prediction lists.
            targets: Tensor of shape [N, 6] with [batch_idx, class_id, cx, cy, w, h] or List of [M, 5].
            img_size: Image dimensions (H, W) in pixels (default: (640, 640)).

        Returns:
            TaskAlignedLossResult with total_loss, box_loss, cls_loss, qual_loss, and num_positives.
        """
        # 1. Unpack predictions across all scales
        # 1. Unpack predictions across all scales
        quality_preds = None
        obj_preds = None
        if isinstance(predictions, HeadOutput):
            box_preds = predictions.box_preds      # List of [B, 4, H_i, W_i]
            obj_preds = predictions.obj_preds      # List of [B, 1, H_i, W_i]
            cls_preds = predictions.cls_preds      # List of [B, C, H_i, W_i]
            quality_preds = predictions.quality_preds  # Optional list of [B, 1, H_i, W_i]
        elif len(predictions) == 4:
            box_preds, obj_preds, cls_preds, quality_preds = predictions
        elif len(predictions) == 3:
            box_preds, obj_preds, cls_preds = predictions
        else:
            box_preds, cls_preds = predictions

        device = box_preds[0].device
        # Always compute loss strictly in float32 to prevent float16 underflow/overflow in CIoU and TAL
        dtype = torch.float32
        B = box_preds[0].shape[0]
        h_img, w_img = img_size

        # Cast predictions to float32
        box_preds = [p.float() for p in box_preds]
        cls_preds = [p.float() for p in cls_preds]
        if obj_preds is not None:
            obj_preds = [p.float() if p is not None else None for p in obj_preds]
        if quality_preds is not None:
            quality_preds = [p.float() if p is not None else None for p in quality_preds]

        grid_shapes = [(p.shape[2], p.shape[3]) for p in box_preds]

        # 2. Generate anchor points across all scales [N_anchors, 2]
        anchor_points, stride_tensor = generate_anchor_grid(grid_shapes, self.strides, device=device, dtype=dtype)
        N_anchors = anchor_points.shape[0]

        # 3. Flatten and decode predicted boxes across all scales: [B, N_anchors, 4]
        decoded_boxes_list = []
        cls_logits_list = []
        qual_logits_list = []
        obj_logits_list = []

        start_idx = 0
        for s_idx, stride in enumerate(self.strides):
            h_g, w_g = grid_shapes[s_idx]
            num_cells = h_g * w_g
            end_idx = start_idx + num_cells

            b_pred = box_preds[s_idx].permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, H*W, 4]
            c_pred = cls_preds[s_idx].permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)  # [B, H*W, C]

            # Scale anchors
            s_anchors = anchor_points[start_idx:end_idx]  # [H*W, 2]
            gx = s_anchors[:, 0] / stride - 0.5
            gy = s_anchors[:, 1] / stride - 0.5

            # Decode raw boxes: [B, H*W, 4] in (x1, y1, x2, y2)
            dec_b = decode_boxes_smooth(b_pred, gx, gy, stride=stride, version="v2_smooth")
            decoded_boxes_list.append(dec_b)
            cls_logits_list.append(c_pred)

            if obj_preds is not None and len(obj_preds) > s_idx and obj_preds[s_idx] is not None:
                o_pred = obj_preds[s_idx].permute(0, 2, 3, 1).reshape(B, -1, 1)  # [B, H*W, 1]
                obj_logits_list.append(o_pred)

            if quality_preds is not None and len(quality_preds) > s_idx and quality_preds[s_idx] is not None:
                q_pred = quality_preds[s_idx].permute(0, 2, 3, 1).reshape(B, -1, 1)  # [B, H*W, 1]
                qual_logits_list.append(q_pred)

            start_idx = end_idx

        all_pred_boxes = torch.cat(decoded_boxes_list, dim=1)    # [B, N_anchors, 4]
        all_cls_logits = torch.cat(cls_logits_list, dim=1)        # [B, N_anchors, C]
        all_pred_scores = torch.sigmoid(all_cls_logits)           # [B, N_anchors, C]

        has_quality = len(qual_logits_list) == len(self.strides)
        all_qual_logits = torch.cat(qual_logits_list, dim=1) if has_quality else None

        has_objectness = len(obj_logits_list) == len(self.strides)
        all_obj_logits = torch.cat(obj_logits_list, dim=1) if has_objectness else None

        # 4. Standardize ground-truth targets into [B, max_gt, 4] and [B, max_gt, 1]
        gt_bboxes_list: List[torch.Tensor] = []
        gt_labels_list: List[torch.Tensor] = []
        max_gt = 0

        if isinstance(targets, list):
            for b_i in range(B):
                t = targets[b_i] if b_i < len(targets) else None
                if t is not None and t.numel() > 0:
                    cls_id = t[:, 0:1].long()
                    cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
                    if (cx <= 1.0).all() and (cy <= 1.0).all():
                        cx, cy, w, h = cx * w_img, cy * h_img, w * w_img, h * h_img
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    gt_bboxes_list.append(boxes)
                    gt_labels_list.append(cls_id)
                    max_gt = max(max_gt, boxes.shape[0])
                else:
                    gt_bboxes_list.append(torch.empty((0, 4), dtype=dtype, device=device))
                    gt_labels_list.append(torch.empty((0, 1), dtype=torch.long, device=device))
        else:
            # targets is [N, 6]: [batch_idx, class_id, cx, cy, w, h]
            for b_i in range(B):
                mask_b = targets[:, 0] == b_i
                t_b = targets[mask_b]
                if t_b.numel() > 0:
                    cls_id = t_b[:, 1:2].long()
                    cx, cy, w, h = t_b[:, 2], t_b[:, 3], t_b[:, 4], t_b[:, 5]
                    if (cx <= 1.0).all() and (cy <= 1.0).all():
                        cx, cy, w, h = cx * w_img, cy * h_img, w * w_img, h * h_img
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    gt_bboxes_list.append(boxes)
                    gt_labels_list.append(cls_id)
                    max_gt = max(max_gt, boxes.shape[0])
                else:
                    gt_bboxes_list.append(torch.empty((0, 4), dtype=dtype, device=device))
                    gt_labels_list.append(torch.empty((0, 1), dtype=torch.long, device=device))

        # Pad ground truths into batched tensors
        if max_gt > 0:
            padded_gt_bboxes = torch.zeros((B, max_gt, 4), dtype=dtype, device=device)
            padded_gt_labels = torch.zeros((B, max_gt, 1), dtype=torch.long, device=device)
            mask_gt = torch.zeros((B, max_gt, 1), dtype=torch.bool, device=device)

            for b_i in range(B):
                m_count = gt_bboxes_list[b_i].shape[0]
                if m_count > 0:
                    padded_gt_bboxes[b_i, :m_count] = gt_bboxes_list[b_i]
                    padded_gt_labels[b_i, :m_count] = gt_labels_list[b_i]
                    mask_gt[b_i, :m_count] = True
        else:
            padded_gt_bboxes = torch.zeros((B, 0, 4), dtype=dtype, device=device)
            padded_gt_labels = torch.zeros((B, 0, 1), dtype=torch.long, device=device)
            mask_gt = torch.zeros((B, 0, 1), dtype=torch.bool, device=device)

        # 5. Execute Task-Aligned Assignor (TAL)
        target_bboxes, target_scores, fg_mask, _ = self.assignor(
            pred_scores=all_pred_scores.detach(),
            pred_bboxes=all_pred_boxes.detach(),
            anchor_points=anchor_points,
            gt_labels=padded_gt_labels,
            gt_bboxes=padded_gt_bboxes,
            mask_gt=mask_gt,
        )

        num_positives = int(fg_mask.sum().item())
        target_score_sum = target_scores.sum().clamp(min=1.0)

        # 6. Classification Loss: Varifocal Loss across all anchors
        cls_loss = varifocal_loss(
            pred_logits=all_cls_logits,
            target_scores=target_scores,
            alpha=self.vfl_alpha,
            gamma=self.vfl_gamma,
            class_weights=self.class_weights if self.class_balanced else None,
        ) / target_score_sum

        # 7. Bounding Box Loss: Quality-Weighted CIoU Loss on positive anchors
        if num_positives > 0:
            pos_pred_boxes = all_pred_boxes[fg_mask]      # [N_pos, 4]
            pos_target_boxes = target_bboxes[fg_mask]    # [N_pos, 4]
            pos_target_scores = target_scores[fg_mask]  # [N_pos, C]

            # Quality weight q_i = max target score for this anchor
            qual_weights = pos_target_scores.max(dim=-1)[0]  # [N_pos]

            ciou = bbox_ciou(pos_pred_boxes, pos_target_boxes)  # [N_pos]
            box_loss = (qual_weights * (1.0 - ciou)).sum() / target_score_sum
        else:
            box_loss = all_pred_boxes.sum() * 0.0

        # 8. Explicit Background Quality Supervision
        # Positive anchors get target = IoU(pred, target)
        # Background anchors get target = 0.0
        qual_loss = torch.zeros(1, dtype=dtype, device=device).squeeze()
        if has_quality and all_qual_logits is not None:
            target_qual = torch.zeros((B, N_anchors, 1), dtype=dtype, device=device)
            if num_positives > 0:
                pos_pred_boxes_det = all_pred_boxes[fg_mask].detach()
                pos_target_boxes_det = target_bboxes[fg_mask]
                ious_pos = bbox_ciou(pos_pred_boxes_det, pos_target_boxes_det).clamp(min=0.0, max=1.0)
                target_qual[fg_mask, 0] = ious_pos

            # BCE with logits across ALL anchors: drives background quality logits negative
            qual_bce = F.binary_cross_entropy_with_logits(all_qual_logits, target_qual, reduction="sum")
            qual_loss = qual_bce / target_score_sum

        # 8b. Explicit Background Objectness Supervision (if objectness branch is present)
        # Positive anchors get target = max continuous alignment score (q_obj in (0, 1])
        # Background anchors get target = 0.0
        obj_loss = torch.zeros(1, dtype=dtype, device=device).squeeze()
        if has_objectness and all_obj_logits is not None:
            target_obj = target_scores.max(dim=-1, keepdim=True)[0]  # [B, N_anchors, 1]
            obj_bce = F.binary_cross_entropy_with_logits(all_obj_logits, target_obj, reduction="sum")
            obj_loss = obj_bce / target_score_sum

        # 9. Total Multi-Task Loss
        total_loss = (
            self.box_weight * box_loss +
            self.cls_weight * cls_loss +
            self.qual_weight * qual_loss +
            self.obj_weight * obj_loss
        )

        return TaskAlignedLossResult(
            total_loss=total_loss,
            box_loss=box_loss,
            cls_loss=cls_loss,
            qual_loss=qual_loss,
            obj_loss=obj_loss,
            num_positives=num_positives,
        )


def build_task_aligned_loss(
    num_classes: int = 12,
    box_weight: float = 5.0,
    cls_weight: float = 1.0,
    qual_weight: float = 0.5,
    strides: Tuple[int, int, int] = (8, 16, 32),
    topk: int = 10,
    tal_alpha: float = 0.5,
    tal_beta: float = 6.0,
    class_balanced: bool = True,
    **kwargs: Any,
) -> TaskAlignedLoss:
    """Helper factory function to construct an IRD V2 TaskAlignedLoss module."""
    return TaskAlignedLoss(
        num_classes=num_classes,
        box_weight=box_weight,
        cls_weight=cls_weight,
        qual_weight=qual_weight,
        strides=strides,
        topk=topk,
        tal_alpha=tal_alpha,
        tal_beta=tal_beta,
        class_balanced=class_balanced,
        **kwargs,
    )
