"""
Custom Indian Road Object Detector package.

Standalone PyTorch object detection architecture tailored for Indian road environments.
"""

from src.models.backbone import (
    ConvBNAct,
    DetailPreservingStem,
    DualPathDownsample,
    IndianRoadBackbone,
    MultiReceptiveBlock,
    MultiScaleContextBlock,
    build_backbone,
)
from src.models.custom_detector import (
    IndianRoadDetector,
    build_detector,
)
from src.models.head import (
    DecoupledConvBlock,
    HeadOutput,
    IndianRoadHead,
    ScaleDecoupledHead,
    SpatialDetailPreserver,
    build_head,
)
from src.models.losses import (
    IndianRoadLoss,
    LossResult,
    MultiScaleSpatialMatcher,
    TaskAlignedAssignor,
    TaskAlignedLoss,
    TaskAlignedLossResult,
    bbox_ciou,
    build_loss,
    build_task_aligned_loss,
    decode_boxes_at_indices,
)
from src.models.neck import (
    AdaptiveScaleFusion,
    HighResDetailEnhancer,
    IndianRoadNeck,
    NeckDownsampler,
    RoadContextAggregator,
    RoadFusionBlock,
    build_neck,
)
from src.models.task_aligned_decoder import decode_ird_v2_predictions

__all__ = [
    # Full Integrated Model
    "IndianRoadDetector",
    "build_detector",
    # Backbone
    "IndianRoadBackbone",
    "build_backbone",
    "DetailPreservingStem",
    "DualPathDownsample",
    "MultiReceptiveBlock",
    "MultiScaleContextBlock",
    # Neck
    "IndianRoadNeck",
    "build_neck",
    "AdaptiveScaleFusion",
    "RoadContextAggregator",
    "HighResDetailEnhancer",
    "RoadFusionBlock",
    "NeckDownsampler",
    # Head
    "IndianRoadHead",
    "build_head",
    "ScaleDecoupledHead",
    "HeadOutput",
    "SpatialDetailPreserver",
    "DecoupledConvBlock",
    "ConvBNAct",
    # Loss & Target Assignment
    "IndianRoadLoss",
    "build_loss",
    "LossResult",
    "MultiScaleSpatialMatcher",
    "bbox_ciou",
    "decode_boxes_at_indices",
    # V2 Task-Aligned Components
    "TaskAlignedAssignor",
    "TaskAlignedLoss",
    "TaskAlignedLossResult",
    "build_task_aligned_loss",
    "decode_ird_v2_predictions",
]
