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

__all__ = [
    "IndianRoadLoss",
    "build_loss",
    "LossResult",
    "MultiScaleSpatialMatcher",
    "bbox_ciou",
    "decode_boxes_at_indices",
]
