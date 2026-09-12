"""
Custom Detection Loss and Target-Assignment System for IndianRoadDetector.

Designed specifically for Indian road conditions:
1. Dense traffic and extreme occlusion:
   - Multi-scale spatial target assignment assigns ground-truth objects based on geometric
     scale and spatial proximity, prioritizing smaller objects in cases of spatial overlap.
2. High scale disparity (tiny signs/pedestrians vs. large buses/trucks):
   - Dynamic stride matching maps objects to appropriate strides (8, 16, 32) with overlapping
     scale boundaries to avoid boundary discretization artifacts.
3. Severe class and foreground/background imbalance (8,400 grid cells vs. ~10-30 objects):
   - Focal binary cross-entropy on objectness downweights easy background predictions.
   - Decoupled multi-label classification loss handles co-occurring categories (e.g. rider + bike).
4. Numerically stable CIoU bounding box regression loss:
   - Penalizes overlap error, normalized center distance, and aspect ratio discrepancy.
   - Employs clamped coordinate decoding to prevent numerical instability.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.head.custom_head import HeadOutput
from src.models.box_coder import decode_boxes_smooth


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete Intersection over Union (CIoU) between two sets of bounding boxes.
    
    Penalizes:
      1. Overlap deficiency: (1 - IoU)
      2. Normalized center distance: rho^2 / c^2
      3. Aspect ratio consistency: alpha * v
      
    Args:
        box1: Predicted boxes [N, 4] in (x1, y1, x2, y2) format.
        box2: Target boxes [N, 4] in (x1, y1, x2, y2) format.
        eps: Small epsilon for numerical stability.
        
    Returns:
        CIoU values [N] clamped to [-1.0, 1.0].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    # Box dimensions
    w1, h1 = (b1_x2 - b1_x1).clamp(min=0), (b1_y2 - b1_y1).clamp(min=0)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=0), (b2_y2 - b2_y1).clamp(min=0)

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps
    iou = inter_area / union_area

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_diag_sq = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2) + eps

    # Center distance
    b1_cx = (b1_x1 + b1_x2) / 2.0
    b1_cy = (b1_y1 + b1_y2) / 2.0
    b2_cx = (b2_x1 + b2_x2) / 2.0
    b2_cy = (b2_y1 + b2_y2) / 2.0
    rho_sq = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2)

    # Aspect ratio term
    v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    ciou = iou - (rho_sq / c_diag_sq) - (alpha * v)
    return ciou.clamp(min=-1.0, max=1.0)


def decode_boxes_at_indices(
    raw_box_preds: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    stride: int,
    version: str = "v2_smooth",
) -> torch.Tensor:
    """
    Authoritative shared box decoder entrypoint for training and validation metrics.
    
    Delegates to src.models.box_coder.decode_boxes_smooth to guarantee 100% mathematical
    identity across loss calculation, evaluation, and inference.
    """
    return decode_boxes_smooth(
        raw_box=raw_box_preds,
        grid_x=grid_x,
        grid_y=grid_y,
        stride=stride,
        version=version,
    )


class MultiScaleSpatialMatcher:
    """
    Custom positive-sample target assignment strategy for Indian road scenes.
    
    Assignment Principles:
      1. Geometric Scale Appropriateness:
         Matches each ground-truth object to scale level(s) based on its characteristic
         scale D = sqrt(w * h) in pixels:
           - Stride 8  (N3): D in [0, 80]     (small objects: signs, pedestrians, bikes)
           - Stride 16 (N4): D in [32, 224]   (medium objects: cars, auto-rickshaws, riders)
           - Stride 32 (N5): D in [128, inf)  (large objects: buses, trucks, tractors)
         Overlapping ranges provide multi-scale supervision for boundary objects.
         Every object is guaranteed assignment to at least its primary scale.
         
      2. Center-Proximity Sampling:
         For an assigned scale, candidates are grid cells within center radius r = 1.2
         whose cell center lies within the object's spatial bounds.
         
      3. Ambiguity Resolution (Small-Object Priority):
         If multiple ground-truth objects match the same grid location, the cell is
         assigned to the object with the smaller area, ensuring small pedestrians/two-wheelers
         are not masked by large adjacent vehicles.
    """
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        center_radius: float = 1.2,
    ) -> None:
        self.strides = strides
        self.center_radius = center_radius

    def match(
        self,
        targets: torch.Tensor,
        grid_shapes: List[Tuple[int, int]],
        img_size: Tuple[int, int] = (640, 640),
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Assign ground truth targets to prediction grid locations across all scales.
        
        Args:
            targets: Tensor of shape [N, 6] with columns [batch_idx, class_id, cx, cy, w, h].
                     Coordinates can be normalized [0, 1] or absolute pixel values.
            grid_shapes: List of (H_i, W_i) for each scale.
            img_size: Image height and width (H, W).
            
        Returns:
            List of 3 dictionaries (one per scale), each containing:
              - 'batch_idx': Tensor of batch indices [N_pos]
              - 'grid_y':    Tensor of grid y indices [N_pos]
              - 'grid_x':    Tensor of grid x indices [N_pos]
              - 'gt_boxes':  Tensor of ground truth boxes [N_pos, 4] in (cx, cy, w, h) pixels
              - 'gt_classes': Tensor of class IDs [N_pos]
        """
        device = targets.device if targets.numel() > 0 else torch.device("cpu")
        num_scales = len(self.strides)
        h_img, w_img = img_size

        matches_per_scale: List[Dict[str, List[Any]]] = [
            {"batch_idx": [], "grid_y": [], "grid_x": [], "gt_boxes": [], "gt_classes": [], "areas": []}
            for _ in range(num_scales)
        ]

        if targets.numel() == 0:
            return [
                {
                    "batch_idx": torch.empty(0, dtype=torch.long, device=device),
                    "grid_y": torch.empty(0, dtype=torch.long, device=device),
                    "grid_x": torch.empty(0, dtype=torch.long, device=device),
                    "gt_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "gt_classes": torch.empty(0, dtype=torch.long, device=device),
                }
                for _ in range(num_scales)
            ]

        # Convert normalized coordinates to absolute pixels if needed
        cx = targets[:, 2]
        cy = targets[:, 3]
        w = targets[:, 4]
        h = targets[:, 5]

        is_normalized = (cx <= 1.0).all() and (cy <= 1.0).all() and (w <= 1.0).all() and (h <= 1.0).all()
        if is_normalized:
            cx_px = cx * w_img
            cy_px = cy * h_img
            w_px = w * w_img
            h_px = h * h_img
        else:
            cx_px, cy_px, w_px, h_px = cx, cy, w, h

        # Object scale (geometric mean)
        obj_scale = torch.sqrt(w_px * h_px + 1e-6)
        obj_areas = w_px * h_px

        # For each ground-truth object
        for k in range(targets.shape[0]):
            b_idx = int(targets[k, 0].item())
            cls_id = int(targets[k, 1].item())
            k_cx = float(cx_px[k].item())
            k_cy = float(cy_px[k].item())
            k_w = float(w_px[k].item())
            k_h = float(h_px[k].item())
            k_scale = float(obj_scale[k].item())
            k_area = float(obj_areas[k].item())

            # Determine eligible scales
            eligible_scales = []
            if k_scale <= 80.0:
                eligible_scales.append(0)  # Stride 8 (N3)
            if 32.0 <= k_scale <= 224.0:
                eligible_scales.append(1)  # Stride 16 (N4)
            if k_scale >= 128.0:
                eligible_scales.append(2)  # Stride 32 (N5)

            # Fallback to guaranteed primary scale if empty
            if not eligible_scales:
                if k_scale < 64.0:
                    eligible_scales = [0]
                elif k_scale < 160.0:
                    eligible_scales = [1]
                else:
                    eligible_scales = [2]

            for s_idx in eligible_scales:
                stride = self.strides[s_idx]
                h_grid, w_grid = grid_shapes[s_idx]

                # Map center to grid units
                gx = k_cx / stride
                gy = k_cy / stride
                gw = k_w / stride
                gh = k_h / stride

                center_j = int(gx)
                center_i = int(gy)

                # Search candidate neighborhood around center
                r = self.center_radius
                j_min = max(0, int(math.floor(gx - r)))
                j_max = min(w_grid - 1, int(math.ceil(gx + r)))
                i_min = max(0, int(math.floor(gy - r)))
                i_max = min(h_grid - 1, int(math.ceil(gy + r)))

                assigned_any = False
                for i_cand in range(i_min, i_max + 1):
                    for j_cand in range(j_min, j_max + 1):
                        cell_cx = j_cand + 0.5
                        cell_cy = i_cand + 0.5

                        # Check center radius distance
                        dist = math.sqrt((cell_cx - gx) ** 2 + (cell_cy - gy) ** 2)
                        if dist <= r:
                            # Check if inside expanded object box
                            in_box_x = (gx - gw / 2.0 - 0.2) <= cell_cx <= (gx + gw / 2.0 + 0.2)
                            in_box_y = (gy - gh / 2.0 - 0.2) <= cell_cy <= (gy + gh / 2.0 + 0.2)
                            if in_box_x and in_box_y:
                                matches_per_scale[s_idx]["batch_idx"].append(b_idx)
                                matches_per_scale[s_idx]["grid_y"].append(i_cand)
                                matches_per_scale[s_idx]["grid_x"].append(j_cand)
                                matches_per_scale[s_idx]["gt_boxes"].append([k_cx, k_cy, k_w, k_h])
                                matches_per_scale[s_idx]["gt_classes"].append(cls_id)
                                matches_per_scale[s_idx]["areas"].append(k_area)
                                assigned_any = True

                # Guaranteed assignment of center grid cell if none matched
                if not assigned_any and 0 <= center_i < h_grid and 0 <= center_j < w_grid:
                    matches_per_scale[s_idx]["batch_idx"].append(b_idx)
                    matches_per_scale[s_idx]["grid_y"].append(center_i)
                    matches_per_scale[s_idx]["grid_x"].append(center_j)
                    matches_per_scale[s_idx]["gt_boxes"].append([k_cx, k_cy, k_w, k_h])
                    matches_per_scale[s_idx]["gt_classes"].append(cls_id)
                    matches_per_scale[s_idx]["areas"].append(k_area)

        # Resolve overlapping assignments (tie-breaking: smaller area wins)
        final_matches = []
        for s_idx in range(num_scales):
            raw = matches_per_scale[s_idx]
            if not raw["batch_idx"]:
                final_matches.append({
                    "batch_idx": torch.empty(0, dtype=torch.long, device=device),
                    "grid_y": torch.empty(0, dtype=torch.long, device=device),
                    "grid_x": torch.empty(0, dtype=torch.long, device=device),
                    "gt_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "gt_classes": torch.empty(0, dtype=torch.long, device=device),
                })
                continue

            # Dict mapping (batch_idx, grid_y, grid_x) -> index of smallest area match
            cell_map: Dict[Tuple[int, int, int], int] = {}
            for idx in range(len(raw["batch_idx"])):
                key = (raw["batch_idx"][idx], raw["grid_y"][idx], raw["grid_x"][idx])
                if key not in cell_map or raw["areas"][idx] < raw["areas"][cell_map[key]]:
                    cell_map[key] = idx

            keep_indices = list(cell_map.values())
            b_arr = torch.tensor([raw["batch_idx"][i] for i in keep_indices], dtype=torch.long, device=device)
            y_arr = torch.tensor([raw["grid_y"][i] for i in keep_indices], dtype=torch.long, device=device)
            x_arr = torch.tensor([raw["grid_x"][i] for i in keep_indices], dtype=torch.long, device=device)
            box_arr = torch.tensor([raw["gt_boxes"][i] for i in keep_indices], dtype=torch.float32, device=device)
            cls_arr = torch.tensor([raw["gt_classes"][i] for i in keep_indices], dtype=torch.long, device=device)

            final_matches.append({
                "batch_idx": b_arr,
                "grid_y": y_arr,
                "grid_x": x_arr,
                "gt_boxes": box_arr,
                "gt_classes": cls_arr,
            })

        return final_matches


class ScaleAdaptiveTopKMatcher:
    """
    Scale-Adaptive Top-K Candidate Assigner for IRD V1 (Matcher Version 2).
    
    Eliminates area-based candidate inflation where large vehicles receive 20+ anchors
    and small motorcycles/pedestrians receive only 1. Guarantees top-k spatial anchors
    per object based on receptive center alignment and scale awareness.
    """
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        topk: int = 4,
        max_dist_radius: float = 2.2,
    ) -> None:
        self.strides = strides
        self.topk = topk
        self.max_dist_radius = max_dist_radius

    def match(
        self,
        targets: torch.Tensor,
        grid_shapes: List[Tuple[int, int]],
        img_size: Tuple[int, int] = (640, 640),
    ) -> List[Dict[str, torch.Tensor]]:
        device = targets.device if targets.numel() > 0 else torch.device("cpu")
        num_scales = len(self.strides)
        h_img, w_img = img_size

        matches_per_scale: List[Dict[str, List[Any]]] = [
            {"batch_idx": [], "grid_y": [], "grid_x": [], "gt_boxes": [], "gt_classes": [], "areas": []}
            for _ in range(num_scales)
        ]

        if targets.numel() == 0:
            return [
                {
                    "batch_idx": torch.empty(0, dtype=torch.long, device=device),
                    "grid_y": torch.empty(0, dtype=torch.long, device=device),
                    "grid_x": torch.empty(0, dtype=torch.long, device=device),
                    "gt_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "gt_classes": torch.empty(0, dtype=torch.long, device=device),
                }
                for _ in range(num_scales)
            ]

        cx = targets[:, 2]
        cy = targets[:, 3]
        w = targets[:, 4]
        h = targets[:, 5]

        is_normalized = (cx <= 1.0).all() and (cy <= 1.0).all() and (w <= 1.0).all() and (h <= 1.0).all()
        if is_normalized:
            cx_px = cx * w_img
            cy_px = cy * h_img
            w_px = w * w_img
            h_px = h * h_img
        else:
            cx_px, cy_px, w_px, h_px = cx, cy, w, h

        obj_scale = torch.sqrt(w_px * h_px + 1e-6)
        obj_areas = w_px * h_px

        for k in range(targets.shape[0]):
            b_idx = int(targets[k, 0].item())
            cls_id = int(targets[k, 1].item())
            k_cx = float(cx_px[k].item())
            k_cy = float(cy_px[k].item())
            k_w = float(w_px[k].item())
            k_h = float(h_px[k].item())
            k_scale = float(obj_scale[k].item())
            k_area = float(obj_areas[k].item())

            # Determine eligible scales
            eligible_scales = []
            if k_scale <= 96.0:
                eligible_scales.append(0)  # Stride 8 (N3: detail / small objects)
            if 48.0 <= k_scale <= 256.0:
                eligible_scales.append(1)  # Stride 16 (N4: medium objects)
            if k_scale >= 160.0:
                eligible_scales.append(2)  # Stride 32 (N5: large vehicles)

            if not eligible_scales:
                eligible_scales = [0] if k_scale < 80.0 else ([1] if k_scale < 200.0 else [2])

            for s_idx in eligible_scales:
                stride = self.strides[s_idx]
                h_grid, w_grid = grid_shapes[s_idx]

                gx = k_cx / stride
                gy = k_cy / stride

                center_j = int(gx)
                center_i = int(gy)

                r = self.max_dist_radius
                j_min = max(0, int(math.floor(gx - r)))
                j_max = min(w_grid - 1, int(math.ceil(gx + r)))
                i_min = max(0, int(math.floor(gy - r)))
                i_max = min(h_grid - 1, int(math.ceil(gy + r)))

                candidates = []
                for i_cand in range(i_min, i_max + 1):
                    for j_cand in range(j_min, j_max + 1):
                        cell_cx = j_cand + 0.5
                        cell_cy = i_cand + 0.5
                        dist = math.sqrt((cell_cx - gx) ** 2 + (cell_cy - gy) ** 2)
                        if dist <= r:
                            candidates.append((dist, i_cand, j_cand))

                candidates.sort(key=lambda x: x[0])
                chosen = candidates[:self.topk]

                if not chosen and 0 <= center_i < h_grid and 0 <= center_j < w_grid:
                    chosen = [(0.0, center_i, center_j)]

                for _, i_c, j_c in chosen:
                    matches_per_scale[s_idx]["batch_idx"].append(b_idx)
                    matches_per_scale[s_idx]["grid_y"].append(i_c)
                    matches_per_scale[s_idx]["grid_x"].append(j_c)
                    matches_per_scale[s_idx]["gt_boxes"].append([k_cx, k_cy, k_w, k_h])
                    matches_per_scale[s_idx]["gt_classes"].append(cls_id)
                    matches_per_scale[s_idx]["areas"].append(k_area)

        # Resolve overlaps: smaller area wins
        final_matches = []
        for s_idx in range(num_scales):
            raw = matches_per_scale[s_idx]
            if not raw["batch_idx"]:
                final_matches.append({
                    "batch_idx": torch.empty(0, dtype=torch.long, device=device),
                    "grid_y": torch.empty(0, dtype=torch.long, device=device),
                    "grid_x": torch.empty(0, dtype=torch.long, device=device),
                    "gt_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "gt_classes": torch.empty(0, dtype=torch.long, device=device),
                })
                continue

            cell_map: Dict[Tuple[int, int, int], int] = {}
            for idx in range(len(raw["batch_idx"])):
                key = (raw["batch_idx"][idx], raw["grid_y"][idx], raw["grid_x"][idx])
                if key not in cell_map or raw["areas"][idx] < raw["areas"][cell_map[key]]:
                    cell_map[key] = idx

            keep_indices = list(cell_map.values())
            b_arr = torch.tensor([raw["batch_idx"][i] for i in keep_indices], dtype=torch.long, device=device)
            y_arr = torch.tensor([raw["grid_y"][i] for i in keep_indices], dtype=torch.long, device=device)
            x_arr = torch.tensor([raw["grid_x"][i] for i in keep_indices], dtype=torch.long, device=device)
            box_arr = torch.tensor([raw["gt_boxes"][i] for i in keep_indices], dtype=torch.float32, device=device)
            cls_arr = torch.tensor([raw["gt_classes"][i] for i in keep_indices], dtype=torch.long, device=device)

            final_matches.append({
                "batch_idx": b_arr,
                "grid_y": y_arr,
                "grid_x": x_arr,
                "gt_boxes": box_arr,
                "gt_classes": cls_arr,
            })

        return final_matches


class LossResult(dict):
    """
    Structured result container for IndianRoadDetector training losses.
    
    Supports:
      1. Attribute access: res.total_loss, res.box_loss, res.objectness_loss, res.classification_loss, res.number_of_positive_samples
      2. Dict access: res['total_loss'], res['box_loss'], etc.
      3. Tuple unpacking: total_loss, box_loss, obj_loss, cls_loss = res
    """
    def __init__(
        self,
        total_loss: torch.Tensor,
        box_loss: torch.Tensor,
        objectness_loss: torch.Tensor,
        classification_loss: torch.Tensor,
        number_of_positive_samples: int,
    ) -> None:
        super().__init__(
            total_loss=total_loss,
            box_loss=box_loss,
            objectness_loss=objectness_loss,
            classification_loss=classification_loss,
            number_of_positive_samples=number_of_positive_samples,
        )
        self.total_loss = total_loss
        self.box_loss = box_loss
        self.objectness_loss = objectness_loss
        self.classification_loss = classification_loss
        self.number_of_positive_samples = number_of_positive_samples

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([self.total_loss, self.box_loss, self.objectness_loss, self.classification_loss])


class IndianRoadLoss(nn.Module):
    """
    Custom Detection Loss Module for IndianRoadDetector.
    
    Combines:
      1. Bounding-box loss: CIoU localization loss computed on positive match locations.
      2. Objectness loss: Focal binary cross-entropy across all prediction locations.
      3. Classification loss: Multi-label focal BCE computed on positive locations.
      
    Args:
        num_classes: Number of object categories (default: 12).
        box_weight: Weight multiplier for bounding-box CIoU loss (default: 5.0).
        obj_weight: Weight multiplier for objectness loss (default: 1.0).
        cls_weight: Weight multiplier for classification loss (default: 1.0).
        strides: Model strides for [N3, N4, N5] (default: (8, 16, 32)).
        focal_gamma: Focusing parameter for focal loss (default: 2.0).
        focal_alpha: Class balance factor for positive samples (default: 0.25).
        decoder_version: 'v2_smooth' or 'v1_legacy'.
        quality_aware_obj: Whether to use IoU soft objectness targets.
        matcher_version: 'v1_spatial' or 'topk_adaptive_v2'.
        class_balanced_loss: Whether to apply inverse-frequency focal weighting.
    """
    def __init__(
        self,
        num_classes: int = 12,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        strides: Tuple[int, int, int] = (8, 16, 32),
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        decoder_version: str = "v2_smooth",
        quality_aware_obj: bool = True,
        matcher_version: str = "v1_spatial",
        class_balanced_loss: bool = False,
        small_obj_floor: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.strides = list(strides)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.decoder_version = decoder_version
        self.quality_aware_obj = quality_aware_obj
        self.matcher_version = matcher_version
        self.class_balanced_loss = class_balanced_loss
        self.small_obj_floor = small_obj_floor

        if matcher_version == "topk_adaptive_v2":
            self.matcher = ScaleAdaptiveTopKMatcher(strides=self.strides, topk=4)
        else:
            self.matcher = MultiScaleSpatialMatcher(strides=self.strides)

    def forward(
        self,
        predictions: Union[HeadOutput, Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]],
        targets: Union[torch.Tensor, List[torch.Tensor]],
        img_size: Tuple[int, int] = (640, 640),
    ) -> LossResult:
        """
        Compute total multi-task detection loss.
        
        Args:
            predictions: HeadOutput or tuple of (box_preds, obj_preds, cls_preds).
            targets: Tensor [N, 6] of [batch_idx, class_id, cx, cy, w, h] or List of [M, 5] tensors.
            img_size: Image height and width (default: (640, 640)).
            
        Returns:
            LossResult with total_loss, box_loss, objectness_loss, classification_loss, and num_positives.
        """
        # Unpack predictions
        if isinstance(predictions, HeadOutput):
            box_preds = predictions.box_preds
            obj_preds = predictions.obj_preds
            cls_preds = predictions.cls_preds
        else:
            box_preds, obj_preds, cls_preds = predictions

        device = box_preds[0].device
        dtype = box_preds[0].dtype
        batch_size = box_preds[0].shape[0]

        # Standardize targets into a single [N, 6] tensor
        if isinstance(targets, list):
            target_list = []
            for b_i, t in enumerate(targets):
                if t is not None and t.numel() > 0:
                    b_col = torch.full((t.shape[0], 1), b_i, dtype=t.dtype, device=t.device)
                    target_list.append(torch.cat([b_col, t], dim=1))
            if target_list:
                targets_tensor = torch.cat(target_list, dim=0).to(device=device)
            else:
                targets_tensor = torch.empty((0, 6), dtype=torch.float32, device=device)
        else:
            targets_tensor = targets.to(device=device)

        grid_shapes = [(p.shape[2], p.shape[3]) for p in box_preds]

        # Perform target assignment
        scale_matches = self.matcher.match(targets_tensor, grid_shapes, img_size=img_size)

        total_positives = sum(len(m["batch_idx"]) for m in scale_matches)
        norm_factor = max(total_positives, 1)

        total_box_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        total_cls_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        total_obj_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()

        # Compute loss per scale
        for s_idx, stride in enumerate(self.strides):
            pred_b = box_preds[s_idx]  # [B, 4, H, W]
            pred_o = obj_preds[s_idx]  # [B, 1, H, W]
            pred_c = cls_preds[s_idx]  # [B, num_classes, H, W]

            h_grid, w_grid = grid_shapes[s_idx]
            match = scale_matches[s_idx]
            n_pos = len(match["batch_idx"])

            # Create objectness target grid
            target_obj = torch.zeros_like(pred_o)  # [B, 1, H, W]

            if n_pos > 0:
                b_idx = match["batch_idx"]
                g_y = match["grid_y"]
                g_x = match["grid_x"]
                gt_boxes = match["gt_boxes"]      # [N_pos, 4] in (cx, cy, w, h)
                gt_classes = match["gt_classes"]  # [N_pos]

                # 1. Bounding-box regression loss (CIoU)
                # Extract raw box predictions at positive grid positions: [N_pos, 4]
                raw_boxes_pos = pred_b[b_idx, :, g_y, g_x]

                # Decode into absolute corners [N_pos, 4] in (x1, y1, x2, y2)
                decoded_pred_boxes = decode_boxes_smooth(
                    raw_boxes_pos, g_x, g_y, stride=stride, version=self.decoder_version
                )

                # Convert ground-truth (cx, cy, w, h) to (x1, y1, x2, y2)
                gt_cx = gt_boxes[:, 0]
                gt_cy = gt_boxes[:, 1]
                gt_w = gt_boxes[:, 2]
                gt_h = gt_boxes[:, 3]
                decoded_gt_boxes = torch.stack([
                    gt_cx - gt_w / 2.0,
                    gt_cy - gt_h / 2.0,
                    gt_cx + gt_w / 2.0,
                    gt_cy + gt_h / 2.0,
                ], dim=-1)

                ciou = bbox_ciou(decoded_pred_boxes, decoded_gt_boxes)
                scale_box_loss = (1.0 - ciou).sum()
                total_box_loss = total_box_loss + scale_box_loss

                # Mark positive locations in objectness target
                if self.quality_aware_obj:
                    # Quality-aware objectness target: smooth soft target based on localization IoU
                    iou_quality = ciou.detach().clamp(min=0.0, max=1.0)
                    base_target = 0.5 + 0.5 * iou_quality
                    if self.small_obj_floor:
                        # Guaranteed Small-Object Presence Supervision (GSO):
                        # Objects with pixel scale < 96.0px (70.6% of Indian Road Dataset)
                        # receive a guaranteed target floor of 0.80 instead of being suppressed
                        # by weak initial CIoU targets (~0.55) against 8400 negative cells.
                        obj_scale_pos = torch.sqrt(gt_w * gt_h + 1e-6)
                        floor_val = torch.tensor(0.80, dtype=dtype, device=device)
                        target_val = torch.where(obj_scale_pos < 96.0, torch.maximum(base_target, floor_val), base_target)
                        target_obj[b_idx, 0, g_y, g_x] = target_val.to(dtype=dtype)
                    else:
                        target_obj[b_idx, 0, g_y, g_x] = base_target.to(dtype=dtype)
                else:
                    target_obj[b_idx, 0, g_y, g_x] = torch.tensor(1.0, dtype=dtype, device=device)

                # 2. Classification loss (Multi-label Focal BCE)
                # Extract class logits at positive locations: [N_pos, num_classes]
                pred_cls_pos = pred_c[b_idx, :, g_y, g_x]
                one_hot_cls = F.one_hot(gt_classes, num_classes=self.num_classes).to(dtype=dtype)

                cls_bce = F.binary_cross_entropy_with_logits(pred_cls_pos, one_hot_cls, reduction="none")
                p_cls = torch.sigmoid(pred_cls_pos)
                p_t_cls = p_cls * one_hot_cls + (1.0 - p_cls) * (1.0 - one_hot_cls)
                cls_focal = (1.0 - p_t_cls).clamp(min=0.0).pow(self.focal_gamma)

                if self.class_balanced_loss:
                    cls_weights_tensor = torch.tensor(
                        [2.2, 1.8, 1.0, 2.4, 2.5, 1.8, 2.4, 2.3, 2.5, 2.3, 2.5, 2.4],
                        device=device, dtype=dtype
                    )
                    cw = cls_weights_tensor[gt_classes].unsqueeze(1)
                    cls_weight_mod = 1.0 + (cw - 1.0) * one_hot_cls
                    scale_cls_loss = (cls_weight_mod * cls_focal * cls_bce).sum()
                else:
                    scale_cls_loss = (cls_focal * cls_bce).sum()

                total_cls_loss = total_cls_loss + scale_cls_loss

            # 3. Objectness loss (Focal BCE across all grid cells on this scale)
            obj_bce = F.binary_cross_entropy_with_logits(pred_o, target_obj, reduction="none")
            p_obj = torch.sigmoid(pred_o)
            p_t_obj = p_obj * target_obj + (1.0 - p_obj) * (1.0 - target_obj)
            alpha_factor = self.focal_alpha * target_obj + (1.0 - self.focal_alpha) * (1.0 - target_obj)
            obj_focal = alpha_factor * (1.0 - p_t_obj).clamp(min=0.0).pow(self.focal_gamma)
            scale_obj_loss = (obj_focal * obj_bce).sum()
            total_obj_loss = total_obj_loss + scale_obj_loss

        # Normalize losses
        final_box_loss = total_box_loss / norm_factor
        final_cls_loss = total_cls_loss / norm_factor
        final_obj_loss = total_obj_loss / norm_factor

        # Weighted total loss
        total_loss = (
            self.box_weight * final_box_loss +
            self.obj_weight * final_obj_loss +
            self.cls_weight * final_cls_loss
        )

        return LossResult(
            total_loss=total_loss,
            box_loss=final_box_loss,
            objectness_loss=final_obj_loss,
            classification_loss=final_cls_loss,
            number_of_positive_samples=total_positives,
        )


def build_loss(
    num_classes: int = 12,
    box_weight: float = 5.0,
    obj_weight: float = 1.0,
    cls_weight: float = 1.0,
    **kwargs: Any,
) -> IndianRoadLoss:
    """Helper factory function to construct an IndianRoadLoss instance."""
    return IndianRoadLoss(
        num_classes=num_classes,
        box_weight=box_weight,
        obj_weight=obj_weight,
        cls_weight=cls_weight,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 78)
    print(f"{'Indian Road Custom Detection Loss: Standalone Verification':^78}")
    print("=" * 78)

    loss_fn = IndianRoadLoss(num_classes=12, box_weight=5.0, obj_weight=1.0, cls_weight=1.0)
    loss_param_count = sum(p.numel() for p in loss_fn.parameters())
    print(f"Loss module parameter count: {loss_param_count} (Non-parametric loss function)")
    print(f"Box weight: {loss_fn.box_weight}, Obj weight: {loss_fn.obj_weight}, Cls weight: {loss_fn.cls_weight}")
    print("-" * 78)

    # 1. Test Batch Size = 1 with Multiple Diverse Objects (Small, Medium, Large)
    print("\n[TEST 1] Batch Size = 1 with Diverse Objects (Small, Medium, Large):")
    # Synthetic HeadOutput for B=1
    preds_b1 = HeadOutput(
        box_preds=[
            torch.randn(1, 4, 80, 80, requires_grad=True),
            torch.randn(1, 4, 40, 40, requires_grad=True),
            torch.randn(1, 4, 20, 20, requires_grad=True),
        ],
        obj_preds=[
            torch.randn(1, 1, 80, 80, requires_grad=True),
            torch.randn(1, 1, 40, 40, requires_grad=True),
            torch.randn(1, 1, 20, 20, requires_grad=True),
        ],
        cls_preds=[
            torch.randn(1, 12, 80, 80, requires_grad=True),
            torch.randn(1, 12, 40, 40, requires_grad=True),
            torch.randn(1, 12, 20, 20, requires_grad=True),
        ],
        strides=[8, 16, 32],
    )

    # Ground truth targets: [batch_idx, class_id, cx, cy, w, h] (normalized in [0, 1])
    targets_b1 = torch.tensor([
        [0, 7, 0.15, 0.20, 0.03, 0.03],  # Small: Traffic Sign (approx 19x19 px in 640x640)
        [0, 5, 0.35, 0.45, 0.05, 0.12],  # Small/Medium: Pedestrian (approx 32x76 px)
        [0, 0, 0.50, 0.60, 0.22, 0.18],  # Medium: Car (approx 140x115 px)
        [0, 3, 0.70, 0.70, 0.45, 0.35],  # Large: Bus (approx 288x224 px)
    ], dtype=torch.float32)

    res_b1 = loss_fn(preds_b1, targets_b1, img_size=(640, 640))
    print(f"  Total Loss:             {res_b1.total_loss.item():.4f}")
    print(f"  Box CIoU Loss:          {res_b1.box_loss.item():.4f}")
    print(f"  Objectness Loss:        {res_b1.objectness_loss.item():.4f}")
    print(f"  Classification Loss:    {res_b1.classification_loss.item():.4f}")
    print(f"  Positive Samples:       {res_b1.number_of_positive_samples}")

    assert torch.isfinite(res_b1.total_loss), "Total loss is not finite in Test 1"
    assert torch.isfinite(res_b1.box_loss), "Box loss is not finite in Test 1"
    assert torch.isfinite(res_b1.objectness_loss), "Objectness loss is not finite in Test 1"
    assert torch.isfinite(res_b1.classification_loss), "Classification loss is not finite in Test 1"
    assert res_b1.number_of_positive_samples > 0, "No positive samples assigned in Test 1"

    # Backward pass
    res_b1.total_loss.backward()
    for s_i in range(3):
        assert preds_b1.box_preds[s_i].grad is not None and torch.isfinite(preds_b1.box_preds[s_i].grad).all()
        assert preds_b1.obj_preds[s_i].grad is not None and torch.isfinite(preds_b1.obj_preds[s_i].grad).all()
        assert preds_b1.cls_preds[s_i].grad is not None and torch.isfinite(preds_b1.cls_preds[s_i].grad).all()
    print("  --> Backward pass and gradient check: PASSED (All gradients finite)")

    # 2. Test Batch Size = 2 (Image 0: Objects, Image 1: Zero Objects)
    print("\n[TEST 2] Batch Size = 2 (Including an Image with Zero Objects):")
    preds_b2 = HeadOutput(
        box_preds=[
            torch.randn(2, 4, 80, 80, requires_grad=True),
            torch.randn(2, 4, 40, 40, requires_grad=True),
            torch.randn(2, 4, 20, 20, requires_grad=True),
        ],
        obj_preds=[
            torch.randn(2, 1, 80, 80, requires_grad=True),
            torch.randn(2, 1, 40, 40, requires_grad=True),
            torch.randn(2, 1, 20, 20, requires_grad=True),
        ],
        cls_preds=[
            torch.randn(2, 12, 80, 80, requires_grad=True),
            torch.randn(2, 12, 40, 40, requires_grad=True),
            torch.randn(2, 12, 20, 20, requires_grad=True),
        ],
        strides=[8, 16, 32],
    )

    # Targets only for batch image 0 (image 1 has 0 objects)
    targets_b2 = torch.tensor([
        [0, 1, 0.25, 0.30, 0.08, 0.15],  # Motorcycle in image 0
        [0, 2, 0.60, 0.50, 0.18, 0.20],  # Auto-rickshaw in image 0
    ], dtype=torch.float32)

    res_b2 = loss_fn(preds_b2, targets_b2, img_size=(640, 640))
    print(f"  Total Loss:             {res_b2.total_loss.item():.4f}")
    print(f"  Box CIoU Loss:          {res_b2.box_loss.item():.4f}")
    print(f"  Objectness Loss:        {res_b2.objectness_loss.item():.4f}")
    print(f"  Classification Loss:    {res_b2.classification_loss.item():.4f}")
    print(f"  Positive Samples:       {res_b2.number_of_positive_samples}")

    assert torch.isfinite(res_b2.total_loss)
    res_b2.total_loss.backward()
    print("  --> Batch Size 2 with partial zero-object image: PASSED")

    # 3. Test Entire Batch with Zero Objects
    print("\n[TEST 3] Entire Batch with Zero Ground-Truth Objects:")
    preds_b0 = HeadOutput(
        box_preds=[
            torch.randn(1, 4, 80, 80, requires_grad=True),
            torch.randn(1, 4, 40, 40, requires_grad=True),
            torch.randn(1, 4, 20, 20, requires_grad=True),
        ],
        obj_preds=[
            torch.randn(1, 1, 80, 80, requires_grad=True),
            torch.randn(1, 1, 40, 40, requires_grad=True),
            torch.randn(1, 1, 20, 20, requires_grad=True),
        ],
        cls_preds=[
            torch.randn(1, 12, 80, 80, requires_grad=True),
            torch.randn(1, 12, 40, 40, requires_grad=True),
            torch.randn(1, 12, 20, 20, requires_grad=True),
        ],
        strides=[8, 16, 32],
    )
    targets_empty = torch.empty((0, 6), dtype=torch.float32)

    res_b0 = loss_fn(preds_b0, targets_empty, img_size=(640, 640))
    print(f"  Total Loss:             {res_b0.total_loss.item():.4f}")
    print(f"  Box CIoU Loss:          {res_b0.box_loss.item():.4f} (Expected: 0.0)")
    print(f"  Objectness Loss:        {res_b0.objectness_loss.item():.4f}")
    print(f"  Classification Loss:    {res_b0.classification_loss.item():.4f} (Expected: 0.0)")
    print(f"  Positive Samples:       {res_b0.number_of_positive_samples} (Expected: 0)")

    assert res_b0.box_loss.item() == 0.0
    assert res_b0.classification_loss.item() == 0.0
    assert res_b0.number_of_positive_samples == 0
    assert torch.isfinite(res_b0.total_loss)
    res_b0.total_loss.backward()
    print("  --> Entirely empty ground truth test: PASSED (No crash, zero NaNs)")

    # 4. Full End-to-End Integration Test: Detector -> Loss -> Backward
    print("\n[TEST 4] End-to-End Integration: IndianRoadDetector -> Custom Loss -> Backward:")
    from src.models.custom_detector import IndianRoadDetector

    detector = IndianRoadDetector(num_classes=12)
    detector.train()

    dummy_images = torch.randn(2, 3, 640, 640)
    dummy_targets = torch.tensor([
        [0, 0, 0.20, 0.30, 0.10, 0.12],  # Car in image 0
        [0, 5, 0.45, 0.50, 0.04, 0.08],  # Pedestrian in image 0
        [1, 1, 0.30, 0.40, 0.06, 0.10],  # Motorcycle in image 1
        [1, 3, 0.70, 0.60, 0.35, 0.25],  # Bus in image 1
    ], dtype=torch.float32)

    raw_preds = detector(dummy_images)
    loss_out = loss_fn(raw_preds, dummy_targets, img_size=(640, 640))
    print(f"  Detector Total Loss:    {loss_out.total_loss.item():.4f}")
    print(f"  Detector Box Loss:      {loss_out.box_loss.item():.4f}")
    print(f"  Detector Obj Loss:      {loss_out.objectness_loss.item():.4f}")
    print(f"  Detector Cls Loss:      {loss_out.classification_loss.item():.4f}")
    print(f"  Detector Positives:     {loss_out.number_of_positive_samples}")

    assert torch.isfinite(loss_out.total_loss)
    loss_out.total_loss.backward()

    # Verify detector parameter gradients are non-empty and finite
    nan_grads = [name for name, p in detector.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    none_grads = [name for name, p in detector.named_parameters() if p.grad is None]
    assert len(nan_grads) == 0, f"NaN gradients in detector: {nan_grads[:5]}"
    assert len(none_grads) == 0, f"None gradients in detector: {none_grads[:5]}"
    print(f"  --> End-to-end backprop through all {sum(1 for _ in detector.parameters())} detector parameters: PASSED")

    print("\n" + "=" * 78)
    print(" ALL CUSTOM LOSS TESTS PASSED SUCCESSFULLY! ")
    print("=" * 78)
