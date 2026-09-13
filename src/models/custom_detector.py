"""
Custom Indian Road Object Detector.

An end-to-end, standalone PyTorch object detector designed specifically for Indian
road environments (dense traffic, high occlusion, small objects, diverse aspect ratios).

Architecture Dataflow:
  Input Image [B, 3, H, W]
       │
       ▼
  Custom Backbone (IndianRoadBackbone)
       ├── DetailPreservingStem
       ├── DualPathDownsample
       ├── MultiReceptiveBlock (MRB)
       └── MultiScaleContextBlock (MSCB)
       │
       ├── P3: [B, 128, H/8,  W/8]   (1/8 scale)
       ├── P4: [B, 256, H/16, W/16]  (1/16 scale)
       └── P5: [B, 512, H/32, W/32]  (1/32 scale)
       │
       ▼
  Custom Multi-Scale Neck (IndianRoadNeck)
       ├── Lateral Projections (128 ch)
       ├── Top-Down Semantic Flow (AdaptiveScaleFusion + RoadFusionBlock)
       ├── HighResDetailEnhancer (N3)
       ├── Bottom-Up Localization Flow (NeckDownsampler + AdaptiveScaleFusion + RoadFusionBlock)
       └── RoadContextAggregator (RCA)
       │
       ├── N3: [B, 128, H/8,  W/8]   (stride 8)
       ├── N4: [B, 128, H/16, W/16]  (stride 16)
       └── N5: [B, 128, H/32, W/32]  (stride 32)
       │
       ▼
  Custom Decoupled Detection Head (IndianRoadHead)
       ├── ScaleDecoupledHead (N3, stride 8,  with SpatialDetailPreserver)
       ├── ScaleDecoupledHead (N4, stride 16)
       └── ScaleDecoupledHead (N5, stride 32)
       │
       └── Structured HeadOutput:
             ├── box_preds: [box_n3, box_n4, box_n5]   (each [B, 4, H_i, W_i])
             ├── obj_preds: [obj_n3, obj_n4, obj_n5]   (each [B, 1, H_i, W_i])
             ├── cls_preds: [cls_n3, cls_n4, cls_n5]   (each [B, 12, H_i, W_i])
             └── strides:   [8, 16, 32]
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

# Ensure project root is in sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.backbone.custom_backbone import IndianRoadBackbone
from src.models.head.custom_head import HeadOutput, IndianRoadHead
from src.models.neck.custom_neck import IndianRoadNeck


class IndianRoadDetector(nn.Module):
    """
    Standalone PyTorch Object Detector for Indian Road Environments.
    
    Integrates:
      1. Custom Backbone: Preserves edge/contrast details and extracts multi-receptive features.
      2. Custom Neck: Learned adaptive scale fusion + road geometry context aggregation.
      3. Custom Head: Decoupled tri-branch prediction (regression, objectness, classification).
      
    Args:
        num_classes: Number of object detection classes (default: 12).
        in_channels: Input image channels (default: 3).
        stem_channels: Backbone stem output channels (default: 32).
        stage_channels: Backbone stage channels for [P2, P3, P4, P5] (default: (64, 128, 256, 512)).
        stage_depths: Backbone block counts for [P2, P3, P4, P5] (default: (2, 3, 4, 3)).
        neck_channels: Unified feature dimension in neck (default: 128).
        neck_refine_blocks: Feature refinement blocks per neck stage (default: 1).
        head_dim: Intermediate feature dimension in decoupled head branches (default: 128).
        head_layers: Number of conv layers per head branch (default: 2).
        strides: Detection strides for [N3, N4, N5] (default: (8, 16, 32)).
    """
    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        stage_depths: Tuple[int, int, int, int] = (2, 3, 4, 3),
        neck_channels: int = 128,
        neck_refine_blocks: int = 1,
        head_dim: int = 128,
        head_layers: int = 2,
        strides: Tuple[int, int, int] = (8, 16, 32),
        use_atd: bool = True,
        use_ssdp: bool = True,
        use_fgbr: bool = True,
        use_quality: bool = True,
        use_cdg: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.neck_channels = neck_channels
        self.strides = list(strides)
        self.use_atd = use_atd
        self.use_ssdp = use_ssdp
        self.use_fgbr = use_fgbr
        self.use_quality = use_quality
        self.use_cdg = use_cdg
        
        # 1. Custom Backbone
        self.backbone = IndianRoadBackbone(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage_channels=stage_channels,
            stage_depths=stage_depths,
        )
        
        # 2. Custom Multi-Scale Feature-Fusion Neck
        backbone_out_channels = tuple(self.backbone.out_channels)  # (128, 256, 512)
        self.neck = IndianRoadNeck(
            in_channels=backbone_out_channels,
            neck_channels=neck_channels,
            num_refine_blocks=neck_refine_blocks,
            use_atd=use_atd,
            use_ssdp=use_ssdp,
        )
        
        # 3. Custom Decoupled Detection Head (with FGBR, LQB, and CDG)
        self.head = IndianRoadHead(
            in_channels=neck_channels,
            head_dim=head_dim,
            num_classes=num_classes,
            strides=strides,
            num_layers=head_layers,
            use_fgbr=use_fgbr,
            use_quality=use_quality,
            use_cdg=use_cdg,
        )

    def forward_features(
        self,
        x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extract backbone and neck multi-scale feature maps.
        
        Args:
            x: Input image tensor [B, 3, H, W].
            
        Returns:
            Tuple of:
              - backbone_features: (P3, P4, P5)
              - neck_features:     (N3, N4, N5)
        """
        if self.use_ssdp:
            p2, p3, p4, p5 = self.backbone(x, return_p2=True)
            n3, n4, n5 = self.neck(p3, p4, p5, p2=p2)
        else:
            p3, p4, p5 = self.backbone(x)
            n3, n4, n5 = self.neck(p3, p4, p5)
        return (p3, p4, p5), (n3, n4, n5)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[HeadOutput, Tuple[HeadOutput, Dict[str, Any]]]:
        """
        Full detector forward pass.
        
        Args:
            x: Input image tensor of shape [B, 3, H, W].
            return_features: If True, returns intermediate backbone & neck features alongside predictions.
            
        Returns:
            HeadOutput containing:
              - box_preds: List of [B, 4, H_i, W_i] for each scale
              - obj_preds: List of [B, 1, H_i, W_i] for each scale
              - cls_preds: List of [B, num_classes, H_i, W_i] for each scale
              - strides:   [8, 16, 32]
            If return_features is True, returns (HeadOutput, features_dict).
        """
        # 1. Feature extraction through custom backbone
        p2 = None
        if self.use_ssdp:
            p2, p3, p4, p5 = self.backbone(x, return_p2=True)
            n3, n4, n5 = self.neck(p3, p4, p5, p2=p2)
        else:
            p3, p4, p5 = self.backbone(x)
            n3, n4, n5 = self.neck(p3, p4, p5)
        
        # 2. Prediction through decoupled detection head
        head_output = self.head(n3, n4, n5)
        
        if return_features:
            features = {
                "backbone": {"p3": p3, "p4": p4, "p5": p5},
                "neck": {"n3": n3, "n4": n4, "n5": n5},
            }
            return head_output, features
            
        return head_output

    def get_parameter_counts(self, only_trainable: bool = True) -> Dict[str, int]:
        """
        Compute parameter counts broken down by component.
        
        Args:
            only_trainable: Whether to count only trainable parameters.
            
        Returns:
            Dictionary with counts for 'backbone', 'neck', 'head', and 'total'.
        """
        def count_params(module: nn.Module) -> int:
            if only_trainable:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        backbone_count = count_params(self.backbone)
        neck_count = count_params(self.neck)
        head_count = count_params(self.head)
        total_count = backbone_count + neck_count + head_count

        return {
            "backbone": backbone_count,
            "neck": neck_count,
            "head": head_count,
            "total": total_count,
        }

    def get_model_summary(
        self,
        input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)
    ) -> str:
        """
        Generate a detailed summary string of the full detector architecture.
        
        Args:
            input_size: Shape of dummy input tensor [B, C, H, W].
            
        Returns:
            Formatted summary table as string.
        """
        params = self.get_parameter_counts(only_trainable=True)
        total_p = params["total"]

        # Run dummy forward pass to extract actual shapes
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(*input_size, dtype=torch.float32, device=next(self.parameters()).device)
            if self.use_ssdp:
                p2, p3, p4, p5 = self.backbone(dummy, return_p2=True)
                n3, n4, n5 = self.neck(p3, p4, p5, p2=p2)
            else:
                p3, p4, p5 = self.backbone(dummy)
                n3, n4, n5 = self.neck(p3, p4, p5)
            out = self.head(n3, n4, n5)
        if was_training:
            self.train()

        lines = [
            "=" * 78,
            f"{'Indian Road Custom Detector Model Summary':^78}",
            "=" * 78,
            f"Input Shape:          {tuple(input_size)}",
            f"Number of Classes:    {self.num_classes}",
            f"Detection Strides:    {self.strides}",
            "-" * 78,
            f"{'Stage':<14}{'Output Name':<14}{'Output Shape':<24}{'Stride':<10}{'Scale Role':<16}",
            "-" * 78,
            f"{'Backbone':<14}{'P3':<14}{str(tuple(p3.shape)):<24}{'8':<10}{'Small objects':<16}",
            f"{'Backbone':<14}{'P4':<14}{str(tuple(p4.shape)):<24}{'16':<10}{'Medium objects':<16}",
            f"{'Backbone':<14}{'P5':<14}{str(tuple(p5.shape)):<24}{'32':<10}{'Large objects':<16}",
            "-" * 78,
            f"{'Neck':<14}{'N3':<14}{str(tuple(n3.shape)):<24}{'8':<10}{'Small objects':<16}",
            f"{'Neck':<14}{'N4':<14}{str(tuple(n4.shape)):<24}{'16':<10}{'Medium objects':<16}",
            f"{'Neck':<14}{'N5':<14}{str(tuple(n5.shape)):<24}{'32':<10}{'Large objects':<16}",
            "-" * 78,
            f"{'Head N3':<14}{'box / obj / cls':<14}{str(tuple(out.box_preds[0].shape))[:22]:<24}{'8':<10}{'[4, 1, 12] ch':<16}",
            f"{'Head N4':<14}{'box / obj / cls':<14}{str(tuple(out.box_preds[1].shape))[:22]:<24}{'16':<10}{'[4, 1, 12] ch':<16}",
            f"{'Head N5':<14}{'box / obj / cls':<14}{str(tuple(out.box_preds[2].shape))[:22]:<24}{'32':<10}{'[4, 1, 12] ch':<16}",
            "=" * 78,
            f"{'Component':<24}{'Trainable Parameters':<28}{'Share (%)':<16}",
            "-" * 78,
            f"{'Custom Backbone':<24}{params['backbone']:>14,} ({params['backbone']/1e6:.2f}M)   {params['backbone']/total_p*100:>8.1f}%",
            f"{'Custom Neck':<24}{params['neck']:>14,} ({params['neck']/1e6:.2f}M)   {params['neck']/total_p*100:>8.1f}%",
            f"{'Custom Head':<24}{params['head']:>14,} ({params['head']/1e6:.2f}M)   {params['head']/total_p*100:>8.1f}%",
            "-" * 78,
            f"{'Total Complete Detector':<24}{total_p:>14,} ({total_p/1e6:.2f}M)   {'100.0%':>9}",
            "=" * 78,
        ]
        return "\n".join(lines)

    def print_summary(
        self,
        input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)
    ) -> None:
        """Print the formatted model summary to stdout."""
        print(self.get_model_summary(input_size=input_size))


def build_detector(
    num_classes: int = 12,
    in_channels: int = 3,
    neck_channels: int = 128,
    head_dim: int = 128,
    **kwargs: Any,
) -> IndianRoadDetector:
    """Helper factory function to construct an IndianRoadDetector instance."""
    return IndianRoadDetector(
        num_classes=num_classes,
        in_channels=in_channels,
        neck_channels=neck_channels,
        head_dim=head_dim,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 78)
    print(f"{'Indian Road Custom Detector: Full Integration Verification':^78}")
    print("=" * 78)

    # 1. Instantiate the integrated model
    model = IndianRoadDetector(num_classes=12)
    model.eval()

    # 2. Print parameter counts and summary
    model.print_summary(input_size=(1, 3, 640, 640))
    params = model.get_parameter_counts(only_trainable=True)

    # 3. Test Batch Size 1
    print("\n[TEST 1] Testing Batch Size = 1 (Input: [1, 3, 640, 640]):")
    x1 = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out1, feats1 = model(x1, return_features=True)

    # Confirm Backbone resolutions
    p3_1, p4_1, p5_1 = feats1["backbone"]["p3"], feats1["backbone"]["p4"], feats1["backbone"]["p5"]
    print(f"  Backbone P3 shape: {tuple(p3_1.shape)} (Expected: [1, 128, 80, 80])")
    print(f"  Backbone P4 shape: {tuple(p4_1.shape)} (Expected: [1, 256, 40, 40])")
    print(f"  Backbone P5 shape: {tuple(p5_1.shape)} (Expected: [1, 512, 20, 20])")
    assert p3_1.shape == (1, 128, 80, 80), f"P3 shape mismatch: {p3_1.shape}"
    assert p4_1.shape == (1, 256, 40, 40), f"P4 shape mismatch: {p4_1.shape}"
    assert p5_1.shape == (1, 512, 20, 20), f"P5 shape mismatch: {p5_1.shape}"
    print("  --> Backbone resolution check [80x80, 40x40, 20x20]: PASSED")

    # Confirm Neck resolutions
    n3_1, n4_1, n5_1 = feats1["neck"]["n3"], feats1["neck"]["n4"], feats1["neck"]["n5"]
    print(f"  Neck N3 shape:     {tuple(n3_1.shape)} (Expected: [1, 128, 80, 80])")
    print(f"  Neck N4 shape:     {tuple(n4_1.shape)} (Expected: [1, 128, 40, 40])")
    print(f"  Neck N5 shape:     {tuple(n5_1.shape)} (Expected: [1, 128, 20, 20])")
    assert n3_1.shape == (1, 128, 80, 80), f"N3 shape mismatch: {n3_1.shape}"
    assert n4_1.shape == (1, 128, 40, 40), f"N4 shape mismatch: {n4_1.shape}"
    assert n5_1.shape == (1, 128, 20, 20), f"N5 shape mismatch: {n5_1.shape}"
    print("  --> Neck resolution check [80x80, 40x40, 20x20]: PASSED")

    # Confirm Head outputs (box=4, obj=1, cls=12)
    box1, obj1, cls1 = out1
    for idx, s in enumerate(model.strides):
        h_exp = 640 // s
        w_exp = 640 // s
        print(f"  Head Scale N{idx+3} (stride {s}):")
        print(f"    Box: {tuple(box1[idx].shape)} (Expected: [1, 4, {h_exp}, {w_exp}])")
        print(f"    Obj: {tuple(obj1[idx].shape)} (Expected: [1, 1, {h_exp}, {w_exp}])")
        print(f"    Cls: {tuple(cls1[idx].shape)} (Expected: [1, 12, {h_exp}, {w_exp}])")
        assert box1[idx].shape == (1, 4, h_exp, w_exp)
        assert obj1[idx].shape == (1, 1, h_exp, w_exp)
        assert cls1[idx].shape == (1, 12, h_exp, w_exp)
    print("  --> Head channel dimensions [box=4, obj=1, cls=12]: PASSED")

    # NaN / Inf Check for Batch 1
    for i in range(3):
        assert not torch.isnan(box1[i]).any(), f"NaN in box[{i}] (batch 1)"
        assert not torch.isinf(box1[i]).any(), f"Inf in box[{i}] (batch 1)"
        assert not torch.isnan(obj1[i]).any(), f"NaN in obj[{i}] (batch 1)"
        assert not torch.isinf(obj1[i]).any(), f"Inf in obj[{i}] (batch 1)"
        assert not torch.isnan(cls1[i]).any(), f"NaN in cls[{i}] (batch 1)"
        assert not torch.isinf(cls1[i]).any(), f"Inf in cls[{i}] (batch 1)"
    print("  --> NaN / Inf check (Batch 1): PASSED (All finite values)")

    # 4. Test Batch Size 2
    print("\n[TEST 2] Testing Batch Size = 2 (Input: [2, 3, 640, 640]):")
    x2 = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out2 = model(x2)

    box2, obj2, cls2 = out2
    for idx, s in enumerate(model.strides):
        h_exp = 640 // s
        w_exp = 640 // s
        assert box2[idx].shape == (2, 4, h_exp, w_exp)
        assert obj2[idx].shape == (2, 1, h_exp, w_exp)
        assert cls2[idx].shape == (2, 12, h_exp, w_exp)
        assert not torch.isnan(box2[idx]).any()
        assert not torch.isinf(box2[idx]).any()
        assert not torch.isnan(obj2[idx]).any()
        assert not torch.isinf(obj2[idx]).any()
        assert not torch.isnan(cls2[idx]).any()
        assert not torch.isinf(cls2[idx]).any()
    print("  --> Batch Size 2 verification: PASSED")

    # 5. Gradient Backward Pass Test
    print("\n[TEST 3] Testing End-to-End Gradient Backward Pass:")
    model.train()
    x_grad = torch.randn(1, 3, 640, 640, requires_grad=True)
    out_g = model(x_grad)

    dummy_loss = (
        sum(b.sum() for b in out_g.box_preds) +
        sum(o.sum() for o in out_g.obj_preds) +
        sum(c.sum() for c in out_g.cls_preds) +
        (sum(q.sum() for q in out_g.quality_preds) if out_g.quality_preds is not None else 0)
    )
    dummy_loss.backward()

    assert x_grad.grad is not None, "Input tensor gradient is None"
    assert not torch.isnan(x_grad.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x_grad.grad).any(), "Inf in input gradient"
    print("  --> Input tensor gradient computed cleanly without NaNs: PASSED")

    # Verify gradients for all parameters
    uncomputed = []
    has_nan_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                uncomputed.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_nan_grad.append(name)

    assert len(uncomputed) == 0, f"Uncomputed gradients in {len(uncomputed)} parameters: {uncomputed[:5]}"
    assert len(has_nan_grad) == 0, f"NaN/Inf gradients in {len(has_nan_grad)} parameters: {has_nan_grad[:5]}"
    print(f"  --> All {sum(1 for _ in model.parameters() if _.requires_grad)} parameter gradients computed without NaNs: PASSED")

    # 6. Configurable class test (e.g. num_classes=20)
    print("\n[TEST 4] Testing Configurable Classes (e.g. num_classes=20):")
    model_20 = IndianRoadDetector(num_classes=20)
    model_20.eval()
    with torch.no_grad():
        out_20 = model_20(torch.randn(1, 3, 640, 640))
    for idx, s in enumerate(model_20.strides):
        assert out_20.cls_preds[idx].shape[1] == 20
    print("  --> num_classes=20 configuration verified: PASSED")

    print("\n" + "=" * 78)
    print(" ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 78)
