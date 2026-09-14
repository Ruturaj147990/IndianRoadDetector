"""
Task-Aligned Assignor (TAL) for IRD V2 (IndianRoadDetector V2).

Replaces the static center-distance matcher from IRD V1.5 with dynamic,
quality-aware task-aligned assignment (TOOD / TAL design):
  t = s^alpha * IoU^beta

Key Properties:
1. Joint Alignment: Considers both classification confidence s and localization
   IoU dynamically to select the best candidates.
2. Dynamic Top-k Selection: Selects the top-k highest aligned candidates per GT.
3. Spatial In-Box Gating: Candidates must fall inside the ground-truth bounding box.
4. Deterministic Conflict Resolution: Resolves overlapping GT candidate contentions
   by assigning the contested anchor to the GT with the highest alignment metric t.
5. Explicit Zero Background: Unassigned anchors receive continuous score targets
   strictly equal to 0.0 across all classes.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute pairwise Intersection-over-Union (IoU) between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4] in (x1, y1, x2, y2).
        boxes2: Tensor of shape [M, 4] in (x1, y1, x2, y2).
        eps: Small constant to avoid division by zero.

    Returns:
        Tensor of shape [N, M] with IoU values in [0, 1].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype, device=boxes1.device)

    # Coordinates: [N, 1, 4] and [1, M, 4]
    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0)

    inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area1 = ((b1[..., 2] - b1[..., 0]).clamp(min=0.0) * (b1[..., 3] - b1[..., 1]).clamp(min=0.0))
    area2 = ((b2[..., 2] - b2[..., 0]).clamp(min=0.0) * (b2[..., 3] - b2[..., 1]).clamp(min=0.0))

    union_area = area1 + area2 - inter_area + eps
    return (inter_area / union_area).clamp(min=0.0, max=1.0)


def generate_anchor_grid(
    grid_shapes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points (cell centers in pixels) and stride factors across multi-scale feature maps.

    Args:
        grid_shapes: List of (H_i, W_i) tuples for each scale.
        strides: List of stride integers (e.g. [8, 16, 32]).
        device: Target torch device.
        dtype: Target data type.

    Returns:
        anchor_points: Tensor [N_anchors, 2] of (cx, cy) center coordinates in pixels.
        stride_tensor: Tensor [N_anchors, 1] of stride values.
    """
    anchor_points_list = []
    stride_list = []

    for (h, w), s in zip(grid_shapes, strides):
        # Grid coordinates
        shift_y, shift_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        # Center in pixel coordinates: (x + 0.5) * stride, (y + 0.5) * stride
        cx = (shift_x + 0.5) * s
        cy = (shift_y + 0.5) * s
        points = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1)  # [H*W, 2]
        strides_scale = torch.full((h * w, 1), s, device=device, dtype=dtype)

        anchor_points_list.append(points)
        stride_list.append(strides_scale)

    anchor_points = torch.cat(anchor_points_list, dim=0)  # [Total_anchors, 2]
    stride_tensor = torch.cat(stride_list, dim=0)        # [Total_anchors, 1]
    return anchor_points, stride_tensor


class TaskAlignedAssignor(nn.Module):
    """
    Task-Aligned Assignor for dynamic sample matching in object detection.

    Computes task-alignment metric:
        t = s^alpha * IoU^beta
    where:
      - s: predicted classification probability for the GT category
      - IoU: predicted bounding box overlap with the GT box
      - alpha: classification score power factor (default: 0.5)
      - beta: localization IoU power factor (default: 6.0)
      - topk: number of candidate anchors to retain per GT (default: 10)

    Args:
        topk: Top-k candidates per ground-truth object (default: 10).
        num_classes: Number of object detection classes (default: 12).
        alpha: Weight power for classification score (default: 0.5).
        beta: Weight power for IoU (default: 6.0).
        eps: Small constant for numerical stability (default: 1e-7).
    """
    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 12,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign ground-truth objects to prediction anchors.

        Args:
            pred_scores: Predicted class probabilities [B, N_anchors, num_classes] in [0, 1].
                         If logits are supplied, callers must apply sigmoid beforehand.
            pred_bboxes: Predicted bounding boxes [B, N_anchors, 4] in (x1, y1, x2, y2) pixels.
            anchor_points: Anchor center coordinates [N_anchors, 2] in (cx, cy) pixels.
            gt_labels: Ground truth class indices [B, N_gt, 1] or [B, N_gt] (long).
            gt_bboxes: Ground truth bounding boxes [B, N_gt, 4] in (x1, y1, x2, y2) pixels.
            mask_gt: Optional validity mask for padded ground truths [B, N_gt, 1] (boolean or 0/1).

        Returns:
            Tuple of 4 tensors:
              - target_bboxes:   [B, N_anchors, 4] Target boxes for positive anchors (0 for background).
              - target_scores:   [B, N_anchors, num_classes] Continuous alignment scores (0 for background).
              - fg_mask:         [B, N_anchors] Boolean mask indicating assigned positive anchors.
              - target_gt_idx:   [B, N_anchors] Index of assigned GT object (-1 for background).
        """
        device = pred_scores.device
        dtype = pred_scores.dtype
        B, N_anchors, C = pred_scores.shape
        N_gt = gt_bboxes.shape[1]

        if gt_labels.dim() == 2:
            gt_labels = gt_labels.unsqueeze(-1)  # [B, N_gt, 1]

        if mask_gt is None:
            # Check if boxes are non-degenerate (w > 0 and h > 0)
            valid_box = (gt_bboxes[..., 2] > gt_bboxes[..., 0]) & (gt_bboxes[..., 3] > gt_bboxes[..., 1])
            mask_gt = valid_box.unsqueeze(-1)  # [B, N_gt, 1]

        target_bboxes = torch.zeros((B, N_anchors, 4), dtype=dtype, device=device)
        target_scores = torch.zeros((B, N_anchors, C), dtype=dtype, device=device)
        fg_mask = torch.zeros((B, N_anchors), dtype=torch.bool, device=device)
        target_gt_idx = torch.full((B, N_anchors), -1, dtype=torch.long, device=device)

        # Process each image in batch independently to ensure complete deterministic isolation
        for b in range(B):
            b_mask_gt = mask_gt[b, :, 0].bool()
            valid_gt_count = int(b_mask_gt.sum().item())

            if valid_gt_count == 0 or N_gt == 0:
                continue

            b_gt_boxes = gt_bboxes[b][b_mask_gt]    # [M, 4]
            b_gt_labels = gt_labels[b][b_mask_gt, 0] # [M]
            M = b_gt_boxes.shape[0]

            b_pred_boxes = pred_bboxes[b]  # [N_anchors, 4]
            b_pred_scores = pred_scores[b]  # [N_anchors, C]

            # 1. Spatial In-Box Constraint:
            # Check if anchor point (cx, cy) is inside the ground truth box [M, N_anchors]
            cx = anchor_points[:, 0].unsqueeze(0)  # [1, N_anchors]
            cy = anchor_points[:, 1].unsqueeze(0)  # [1, N_anchors]
            gx1 = b_gt_boxes[:, 0].unsqueeze(1)    # [M, 1]
            gy1 = b_gt_boxes[:, 1].unsqueeze(1)    # [M, 1]
            gx2 = b_gt_boxes[:, 2].unsqueeze(1)    # [M, 1]
            gy2 = b_gt_boxes[:, 3].unsqueeze(1)    # [M, 1]

            in_box_mask = (cx >= gx1) & (cx <= gx2) & (cy >= gy1) & (cy <= gy2)  # [M, N_anchors]

            # 2. Pairwise IoU calculation: [M, N_anchors]
            ious = box_iou_pairwise(b_gt_boxes, b_pred_boxes)  # [M, N_anchors]

            # 3. Extract predicted scores for the corresponding GT classes: [M, N_anchors]
            # b_gt_labels: [M] -> gather across class dimension for each GT
            cls_scores = b_pred_scores[:, b_gt_labels].transpose(0, 1)  # [M, N_anchors]
            cls_scores = cls_scores.clamp(min=0.0, max=1.0)

            # 4. Task-Alignment Metric computation:
            # t = s^alpha * IoU^beta
            alignment_metric = (cls_scores.pow(self.alpha) * ious.pow(self.beta)).clamp(min=0.0)

            # Mask out anchors outside the ground truth bounding box
            alignment_metric = alignment_metric * in_box_mask.to(dtype=dtype)

            # 5. Dynamic Top-k Candidate Selection per GT:
            # Select top-k anchors with highest alignment metric for each GT object
            k = min(self.topk, N_anchors)
            topk_metrics, topk_indices = torch.topk(alignment_metric, k=k, dim=-1, largest=True)  # [M, k]

            # Binary mask for selected top-k candidates
            topk_mask = torch.zeros_like(alignment_metric, dtype=torch.bool)
            # Only keep top-k candidates that have positive alignment metric and are inside box
            valid_topk = (topk_metrics > 0.0)
            for m in range(M):
                sel_idx = topk_indices[m][valid_topk[m]]
                topk_mask[m, sel_idx] = True

            # Candidates must satisfy both top-k and in-box
            candidate_mask = topk_mask & in_box_mask

            # 6. Multi-GT Conflict Resolution:
            # An anchor may be selected by multiple GT objects (e.g. dense clusters, rider + motorcycle).
            # Contention is resolved deterministically by assigning the anchor to the GT with the
            # LARGEST alignment metric.
            candidate_metrics = alignment_metric * candidate_mask.to(dtype=dtype)  # [M, N_anchors]
            max_metric_per_anchor, best_gt_for_anchor = candidate_metrics.max(dim=0)  # [N_anchors]

            # Valid foreground anchors are those with positive best metric
            pos_anchor_mask = max_metric_per_anchor > 0.0  # [N_anchors]

            if not pos_anchor_mask.any():
                continue

            fg_mask[b] = pos_anchor_mask
            assigned_gt_idx = best_gt_for_anchor[pos_anchor_mask]  # [N_pos]
            target_gt_idx[b, pos_anchor_mask] = assigned_gt_idx

            # 7. Target Bounding Boxes:
            target_bboxes[b, pos_anchor_mask] = b_gt_boxes[assigned_gt_idx]

            # 8. Normalized Target Score:
            # For each GT, normalize alignment metric:
            #   t* = IoU * (t / max(t))
            # ensuring that the best aligned candidate's target equals its actual IoU.
            # Normalizing factor per GT: [M]
            max_t_per_gt = alignment_metric.max(dim=-1, keepdim=True)[0] + self.eps  # [M, 1]
            normalized_scores = (ious * (alignment_metric / max_t_per_gt)).clamp(min=0.0, max=1.0)

            # Assign continuous target score for the assigned GT class
            pos_indices = pos_anchor_mask.nonzero(as_tuple=False).squeeze(1)
            for p_idx, gt_i in zip(pos_indices, assigned_gt_idx):
                target_cls = b_gt_labels[gt_i].item()
                target_scores[b, p_idx, target_cls] = normalized_scores[gt_i, p_idx]

        return target_bboxes, target_scores, fg_mask, target_gt_idx
