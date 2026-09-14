"""
Custom Detection Loss and Target-Assignment package for IndianRoadDetector.
"""

from src.models.losses.custom_loss import (
    IndianRoadLoss,
    LossResult,
    MultiScaleSpatialMatcher,
    bbox_ciou,
    build_loss,
    decode_boxes_at_indices,
)
from src.models.losses.task_aligned_assignor import (
    TaskAlignedAssignor,
    box_iou_pairwise,
    generate_anchor_grid,
)
from src.models.losses.task_aligned_loss import (
    TaskAlignedLoss,
    TaskAlignedLossResult,
    build_task_aligned_loss,
    varifocal_loss,
)

__all__ = [
    # V1.5 Loss & Matcher
    "IndianRoadLoss",
    "build_loss",
    "LossResult",
    "MultiScaleSpatialMatcher",
    "bbox_ciou",
    "decode_boxes_at_indices",
    # V2 Task-Aligned Loss & Assignor
    "TaskAlignedAssignor",
    "TaskAlignedLoss",
    "TaskAlignedLossResult",
    "build_task_aligned_loss",
    "varifocal_loss",
    "box_iou_pairwise",
    "generate_anchor_grid",
]

