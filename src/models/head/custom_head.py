"""
Custom Decoupled Detection Head for Indian Road Object Detection.

Designed specifically for Indian road conditions:
- Heavy occlusion and dense traffic: Decoupling classification, bounding-box regression,
  and object confidence avoids feature misalignment where background patches or occluded
  edges produce false positives.
- Extreme scale variation: Dedicated decoupled prediction heads for N3 (80x80), N4 (40x40),
  and N5 (20x20) allow each scale to specialize.
- High-resolution spatial detail preservation on N3: Protects thin boundaries of traffic signs,
  distant pedestrians, motorcycles, and bicycles.
- Predicts 12 Indian road classes:
  [car, motorcycle, auto_rickshaw, bus, truck, pedestrian, bicycle, traffic_sign,
   traffic_light, animal, rider, barricade].

Output structure:
For each scale (N3, N4, N5), returns:
  - box_preds: [B, 4, H, W]  (raw box regression parameters)
  - obj_preds: [B, 1, H, W]  (raw objectness confidence logits)
  - cls_preds: [B, 12, H, W] (raw classification logits)
"""

import math
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """
    Standard Convolution + BatchNorm + Activation helper block.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Stride of the convolution.
        padding: Padding size (auto-calculated if None).
        dilation: Dilation rate.
        groups: Number of blocked connections from input to output channels.
        act: Whether to apply activation (SiLU).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size // 2) if dilation == 1 else dilation * (kernel_size // 2)
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DecoupledConvBlock(nn.Module):
    """
    Lightweight depthwise-separable convolution block for detection head branches.
    
    Combines 3x3 depthwise convolution with 1x1 pointwise projection to provide
    rich spatial feature processing at minimal computational and parameter overhead
    on Tesla T4 GPUs.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_channels, in_channels, kernel_size=3, groups=in_channels)
        self.pw = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SpatialDetailPreserver(nn.Module):
    """
    Preserves and sharpens high-resolution spatial details specifically for N3 (80x80).
    
    In Indian road scenes, distant pedestrians, small traffic lights, and two-wheeler
    silhouettes depend on crisp spatial edge gradients that standard convolution stacks
    can blur. This block maintains an explicit identity path coupled with learnable
    depthwise residual sharpening.
    
    Args:
        channels: Feature dimension.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge_dw = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.pw = ConvBNAct(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.pw(self.edge_dw(x))


class FineGrainedBoundaryRefiner(nn.Module):
    """
    Fine-Grained Boundary Refiner (FGBR) for High-Resolution Regression.
    
    Predicts residual spatial corrections for bounding box coordinates from high-frequency
    gradients, sharpening localization boundaries at high IoU thresholds (0.75 and 0.90)
    specifically for small and distant traffic participants (two-wheelers, pedestrians, signs).
    """
    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()
        mid_channels = max(32, in_channels // 4)
        self.refine_conv = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=3, groups=mid_channels),
            nn.Conv2d(mid_channels, 4, kernel_size=1),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.ones(1, 4, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.refine_conv(x)


class LocalizationQualityBranch(nn.Module):
    """
    Localization Quality Branch (LQB) for Decoupled Detection Heads.
    
    Predicts continuous IoU localization quality q in [0, 1] conditioned on the bounding-box
    regression feature representation. When supervised with actual CIoU / IoU during training,
    it allows the model to differentiate between tightly localized and sloppy boxes.
    At inference time, it calibrates final detection confidence to penalize jittery boxes.
    """
    def __init__(self, head_dim: int = 128) -> None:
        super().__init__()
        mid_dim = max(32, head_dim // 4)
        self.quality_conv = nn.Sequential(
            ConvBNAct(head_dim, mid_dim, kernel_size=3, groups=mid_dim),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )
        # Initialize bias to 0.0 so sigmoid(0) = 0.5 (neutral initial quality)
        nn.init.constant_(self.quality_conv[1].bias, 0.0)

    def forward(self, reg_feat: torch.Tensor) -> torch.Tensor:
        return self.quality_conv(reg_feat)


class ClassDiscriminativeGate(nn.Module):
    """
    Class-Discriminative Gate (CDG) for Classification Branch.
    
    In dense Indian traffic scenes, vehicle categories exhibit strong aspect-ratio priors:
    - Slender/Vertical: Person (0), Rider (1), Motorcycle (5), Bicycle (6) [H > W]
    - Compact/Square: Car (2), Autorickshaw (7), Animal (8) [H ~ W]
    - Elongated/Massive: Truck (3), Bus (4) [W >> H or large area]
    
    CDG combines orthogonal strip depthwise convolutions (1x5 and 5x1) with global
    squeeze-and-excitation channel modulation to decouple confusing pairs (e.g. truck vs car,
    bus vs car, rider vs pedestrian).
    """
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        mid_dim = max(32, channels // 4)
        # Aspect-ratio horizontal & vertical strip depthwise convs
        self.h_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.v_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        # Global channel context excitation
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj = ConvBNAct(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strip_mix = self.h_strip(x) + self.v_strip(x)
        gated = strip_mix * self.channel_gate(strip_mix)
        return self.gamma * self.proj(gated)


class ScaleDecoupledHead(nn.Module):
    """
    Decoupled prediction head for a single feature scale.
    
    Contains four decoupled branches:
      1. Bounding-box regression branch: Predicts 4 box parameters.
      2. Objectness confidence branch: Predicts 1 object presence logit.
      3. Classification branch: Predicts 12 class logits with ClassDiscriminativeGate.
      4. Localization quality branch: Predicts 1 continuous IoU quality logit.
      
    Args:
        in_channels: Input channels from neck (default: 128).
        head_dim: Intermediate feature dimension inside head branches (default: 128).
        num_classes: Number of object categories (default: 12).
        is_high_res: Whether this head processes the high-res N3 scale (default: False).
        num_layers: Number of convolutional layers per branch (default: 2).
        use_fgbr: Whether to enable Fine-Grained Boundary Refiner on regression (default: False).
        use_quality: Whether to enable Localization Quality Branch (default: True).
        use_cdg: Whether to enable Class-Discriminative Gate on classification (default: True).
    """
    def __init__(
        self,
        in_channels: int = 128,
        head_dim: int = 128,
        num_classes: int = 12,
        is_high_res: bool = False,
        num_layers: int = 2,
        use_fgbr: bool = False,
        use_quality: bool = True,
        use_cdg: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.head_dim = head_dim
        self.num_classes = num_classes
        self.is_high_res = is_high_res
        self.use_fgbr = use_fgbr
        self.use_quality = use_quality
        self.use_cdg = use_cdg
        
        # Spatial detail preservation path for small objects (active on N3)
        self.detail_path = SpatialDetailPreserver(in_channels) if is_high_res else nn.Identity()
        
        # 1. Bounding-Box Regression Branch (4 box parameters)
        reg_layers = [DecoupledConvBlock(in_channels, head_dim)]
        for _ in range(num_layers - 1):
            reg_layers.append(DecoupledConvBlock(head_dim, head_dim))
        self.reg_convs = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)
        if self.use_fgbr and self.is_high_res:
            self.fgbr = FineGrainedBoundaryRefiner(head_dim)
            
        # 1b. Localization Quality Branch (IoU-prediction)
        if self.use_quality:
            self.quality_branch = LocalizationQualityBranch(head_dim)
        
        # 2. Objectness Branch (1 confidence value)
        obj_dim = max(32, head_dim // 2)
        obj_layers = [DecoupledConvBlock(in_channels, obj_dim)]
        for _ in range(num_layers - 1):
            obj_layers.append(DecoupledConvBlock(obj_dim, obj_dim))
        self.obj_convs = nn.Sequential(*obj_layers)
        self.obj_pred = nn.Conv2d(obj_dim, 1, kernel_size=1)
        
        # 3. Classification Branch (12 classes with ClassDiscriminativeGate)
        cls_layers = [DecoupledConvBlock(in_channels, head_dim)]
        for _ in range(num_layers - 1):
            cls_layers.append(DecoupledConvBlock(head_dim, head_dim))
        self.cls_convs = nn.Sequential(*cls_layers)
        if self.use_cdg:
            self.cdg = ClassDiscriminativeGate(head_dim)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Weight initialization:
        - Conv layers: Kaiming normal.
        - BatchNorm layers: weight=1.0, bias=0.0.
        - Reg pred bias: 0.0.
        - Cls & Obj pred bias: Initialized with prior probability pi = 0.01 (-4.595)
          to prevent early training instability caused by extreme foreground-background imbalance.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # Prior probability bias initialization: -log((1 - pi) / pi) for pi = 0.01
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for a single scale.
        
        Args:
            x: Input feature tensor [B, in_channels, H, W].
            
        Returns:
            Tuple of:
              - box_preds:     [B, 4, H, W]
              - obj_preds:     [B, 1, H, W]
              - cls_preds:     [B, num_classes, H, W]
              - quality_preds: [B, 1, H, W] or None
        """
        feat = self.detail_path(x)
        
        # 1. Regression
        reg_feat = self.reg_convs(feat)
        box_preds = self.reg_pred(reg_feat)
        if hasattr(self, "fgbr"):
            box_preds = box_preds + self.fgbr(reg_feat)
            
        # 2. Objectness
        obj_preds = self.obj_pred(self.obj_convs(feat))
        
        # 3. Classification with CDG
        cls_feat = self.cls_convs(feat)
        if hasattr(self, "cdg"):
            cls_feat = cls_feat + self.cdg(cls_feat)
        cls_preds = self.cls_pred(cls_feat)
        
        # 4. Localization Quality
        qual_preds = self.quality_branch(reg_feat) if hasattr(self, "quality_branch") else None
        
        return box_preds, obj_preds, cls_preds, qual_preds


class HeadOutput(dict):
    """
    Structured container for multi-scale detection head outputs.
    
    Supports:
      1. Attribute access: output.box_preds, output.obj_preds, output.cls_preds, output.strides, output.quality_preds
      2. Dict access: output['box_preds'], output['obj_preds'], output['cls_preds'], output['strides'], output['quality_preds']
      3. Tuple unpacking: box_preds, obj_preds, cls_preds = output (backwards-compatible 3-tuple)
      4. Scale accessor: output.get_scale(0) -> {'box': ..., 'obj': ..., 'cls': ..., 'stride': 8, 'quality': ...}
    """
    def __init__(
        self,
        box_preds: List[torch.Tensor],
        obj_preds: List[torch.Tensor],
        cls_preds: List[torch.Tensor],
        strides: List[int] = [8, 16, 32],
        quality_preds: Optional[List[torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            box_preds=box_preds,
            obj_preds=obj_preds,
            cls_preds=cls_preds,
            strides=strides,
            quality_preds=quality_preds,
        )
        self.box_preds = box_preds
        self.obj_preds = obj_preds
        self.cls_preds = cls_preds
        self.strides = strides
        self.quality_preds = quality_preds

    def __iter__(self) -> Iterator[List[torch.Tensor]]:
        # Preserve backwards compatibility for 3-tuple unpacking
        return iter([self.box_preds, self.obj_preds, self.cls_preds])

    def get_scale(self, idx: int) -> Dict[str, Any]:
        """Retrieve prediction tensors and stride for a specific scale index."""
        res = {
            "box": self.box_preds[idx],
            "obj": self.obj_preds[idx],
            "cls": self.cls_preds[idx],
            "stride": self.strides[idx],
        }
        if self.quality_preds is not None:
            res["quality"] = self.quality_preds[idx]
        return res


class IndianRoadHead(nn.Module):
    """
    Custom Decoupled Detection Head for Indian Road Object Detection.
    
    Builds scale-specialized decoupled heads for N3, N4, and N5:
      - N3 (stride 8,  80x80): Small objects (traffic signs, lights, pedestrians, bicycles).
      - N4 (stride 16, 40x40): Medium objects (cars, auto-rickshaws, riders, animals).
      - N5 (stride 32, 20x20): Large objects (buses, trucks, tractors, barricades).
      
    Each head decouples:
      - Regression: 4 box parameters (with FGBR on N3)
      - Objectness: 1 presence confidence logit
      - Classification: 12 class logits (with ClassDiscriminativeGate)
      - Localization Quality: 1 continuous IoU quality logit
      
    Args:
        in_channels: Channels from neck (default: 128).
        head_dim: Intermediate feature dimension in head branches (default: 128).
        num_classes: Number of detection classes (default: 12).
        strides: Stride values for [N3, N4, N5] (default: (8, 16, 32)).
        num_layers: Number of convolutional layers per branch (default: 2).
        use_fgbr: Whether to enable Fine-Grained Boundary Refiner on N3 (default: True).
        use_quality: Whether to enable Localization Quality Branch (default: True).
        use_cdg: Whether to enable Class-Discriminative Gate (default: True).
    """
    def __init__(
        self,
        in_channels: int = 128,
        head_dim: int = 128,
        num_classes: int = 12,
        strides: Tuple[int, int, int] = (8, 16, 32),
        num_layers: int = 2,
        use_fgbr: bool = True,
        use_quality: bool = True,
        use_cdg: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.head_dim = head_dim
        self.num_classes = num_classes
        self.num_box_params = 4
        self.strides = list(strides)
        self.use_fgbr = use_fgbr
        self.use_quality = use_quality
        self.use_cdg = use_cdg
        
        # Scale-specialized decoupled heads
        # Head N3: is_high_res=True enables SpatialDetailPreserver & FGBR
        self.head_n3 = ScaleDecoupledHead(
            in_channels=in_channels,
            head_dim=head_dim,
            num_classes=num_classes,
            is_high_res=True,
            num_layers=num_layers,
            use_fgbr=use_fgbr,
            use_quality=use_quality,
            use_cdg=use_cdg,
        )
        # Head N4: Standard decoupled head with CDG and Quality
        self.head_n4 = ScaleDecoupledHead(
            in_channels=in_channels,
            head_dim=head_dim,
            num_classes=num_classes,
            is_high_res=False,
            num_layers=num_layers,
            use_fgbr=False,
            use_quality=use_quality,
            use_cdg=use_cdg,
        )
        # Head N5: Standard decoupled head with CDG and Quality
        self.head_n5 = ScaleDecoupledHead(
            in_channels=in_channels,
            head_dim=head_dim,
            num_classes=num_classes,
            is_high_res=False,
            num_layers=num_layers,
            use_fgbr=False,
            use_quality=use_quality,
            use_cdg=use_cdg,
        )

    def forward(
        self,
        n3: torch.Tensor,
        n4: torch.Tensor,
        n5: torch.Tensor,
    ) -> HeadOutput:
        """
        Forward pass through the decoupled detection head.
        
        Args:
            n3: Neck feature map at stride 8  [B, in_channels, 80, 80].
            n4: Neck feature map at stride 16 [B, in_channels, 40, 40].
            n5: Neck feature map at stride 32 [B, in_channels, 20, 20].
            
        Returns:
            HeadOutput containing:
              - box_preds:     [box_n3, box_n4, box_n5]
              - obj_preds:     [obj_n3, obj_n4, obj_n5]
              - cls_preds:     [cls_n3, cls_n4, cls_n5]
              - strides:       [8, 16, 32]
              - quality_preds: [qual_n3, qual_n4, qual_n5]
        """
        box_n3, obj_n3, cls_n3, q_n3 = self.head_n3(n3)
        box_n4, obj_n4, cls_n4, q_n4 = self.head_n4(n4)
        box_n5, obj_n5, cls_n5, q_n5 = self.head_n5(n5)
        
        quality_preds = [q_n3, q_n4, q_n5] if q_n3 is not None else None
        
        return HeadOutput(
            box_preds=[box_n3, box_n4, box_n5],
            obj_preds=[obj_n3, obj_n4, obj_n5],
            cls_preds=[cls_n3, cls_n4, cls_n5],
            strides=self.strides,
            quality_preds=quality_preds,
        )


def build_head(
    in_channels: int = 128,
    head_dim: int = 128,
    num_classes: int = 12,
    strides: Tuple[int, int, int] = (8, 16, 32),
    num_layers: int = 2,
    use_fgbr: bool = True,
    use_quality: bool = True,
    use_cdg: bool = True,
) -> IndianRoadHead:
    """Helper factory function to construct an IndianRoadHead instance."""
    return IndianRoadHead(
        in_channels=in_channels,
        head_dim=head_dim,
        num_classes=num_classes,
        strides=strides,
        num_layers=num_layers,
        use_fgbr=use_fgbr,
        use_quality=use_quality,
        use_cdg=use_cdg,
    )


if __name__ == "__main__":
    print("=" * 75)
    print(" Indian Road Custom Decoupled Head: Standalone Verification Test ")
    print("=" * 75)

    head = IndianRoadHead(
        in_channels=128,
        head_dim=128,
        num_classes=12,
        strides=(8, 16, 32),
        num_layers=2,
    )
    head.eval()

    # 1. Parameter counts
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in head.parameters())
    print(f"Total parameters:           {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters:       {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Num classes:                {head.num_classes}")
    print(f"Num box params:             {head.num_box_params}")
    print(f"Strides:                    {head.strides}")
    print("-" * 75)

    # 2. Test Batch Size 1
    print("Testing Batch Size = 1:")
    n3_b1 = torch.randn(1, 128, 80, 80)
    n4_b1 = torch.randn(1, 128, 40, 40)
    n5_b1 = torch.randn(1, 128, 20, 20)

    with torch.no_grad():
        out_b1 = head(n3_b1, n4_b1, n5_b1)

    box_b1, obj_b1, cls_b1 = out_b1
    for i, s in enumerate(head.strides):
        print(f"  Scale N{i+3} (stride {s}):")
        print(f"    Box shape: {tuple(box_b1[i].shape)}  (Expected: [1, 4, {640//s}, {640//s}])")
        print(f"    Obj shape: {tuple(obj_b1[i].shape)}  (Expected: [1, 1, {640//s}, {640//s}])")
        print(f"    Cls shape: {tuple(cls_b1[i].shape)} (Expected: [1, 12, {640//s}, {640//s}])")

    # Verify dimensions for B=1
    assert box_b1[0].shape == (1, 4, 80, 80), f"N3 box shape mismatch: {box_b1[0].shape}"
    assert obj_b1[0].shape == (1, 1, 80, 80), f"N3 obj shape mismatch: {obj_b1[0].shape}"
    assert cls_b1[0].shape == (1, 12, 80, 80), f"N3 cls shape mismatch: {cls_b1[0].shape}"

    assert box_b1[1].shape == (1, 4, 40, 40), f"N4 box shape mismatch: {box_b1[1].shape}"
    assert obj_b1[1].shape == (1, 1, 40, 40), f"N4 obj shape mismatch: {obj_b1[1].shape}"
    assert cls_b1[1].shape == (1, 12, 40, 40), f"N4 cls shape mismatch: {cls_b1[1].shape}"

    assert box_b1[2].shape == (1, 4, 20, 20), f"N5 box shape mismatch: {box_b1[2].shape}"
    assert obj_b1[2].shape == (1, 1, 20, 20), f"N5 obj shape mismatch: {obj_b1[2].shape}"
    assert cls_b1[2].shape == (1, 12, 20, 20), f"N5 cls shape mismatch: {cls_b1[2].shape}"

    # Verify NaN / Inf for B=1
    for i in range(3):
        assert not torch.isnan(box_b1[i]).any(), f"NaN in box_b1[{i}]"
        assert not torch.isinf(box_b1[i]).any(), f"Inf in box_b1[{i}]"
        assert not torch.isnan(obj_b1[i]).any(), f"NaN in obj_b1[{i}]"
        assert not torch.isinf(obj_b1[i]).any(), f"Inf in obj_b1[{i}]"
        assert not torch.isnan(cls_b1[i]).any(), f"NaN in cls_b1[{i}]"
        assert not torch.isinf(cls_b1[i]).any(), f"Inf in cls_b1[{i}]"
    print("  Spatial resolutions [80x80, 40x40, 20x20]: PASSED")
    print("  Channel dimensions [box=4, obj=1, cls=12]: PASSED")
    print("  NaN/Inf check: PASSED (All finite values)")
    print("-" * 75)

    # 3. Test Batch Size 2
    print("Testing Batch Size = 2:")
    n3_b2 = torch.randn(2, 128, 80, 80)
    n4_b2 = torch.randn(2, 128, 40, 40)
    n5_b2 = torch.randn(2, 128, 20, 20)

    with torch.no_grad():
        out_b2 = head(n3_b2, n4_b2, n5_b2)

    box_b2, obj_b2, cls_b2 = out_b2
    assert box_b2[0].shape == (2, 4, 80, 80)
    assert obj_b2[0].shape == (2, 1, 80, 80)
    assert cls_b2[0].shape == (2, 12, 80, 80)

    assert box_b2[1].shape == (2, 4, 40, 40)
    assert obj_b2[1].shape == (2, 1, 40, 40)
    assert cls_b2[1].shape == (2, 12, 40, 40)

    assert box_b2[2].shape == (2, 4, 20, 20)
    assert obj_b2[2].shape == (2, 1, 20, 20)
    assert cls_b2[2].shape == (2, 12, 20, 20)

    for i in range(3):
        assert not torch.isnan(box_b2[i]).any()
        assert not torch.isinf(box_b2[i]).any()
        assert not torch.isnan(obj_b2[i]).any()
        assert not torch.isinf(obj_b2[i]).any()
        assert not torch.isnan(cls_b2[i]).any()
        assert not torch.isinf(cls_b2[i]).any()
    print("  Batch 2 verification: PASSED")
    print("-" * 75)

    # 4. Gradient backward pass verification
    print("Testing Gradient Backward Pass:")
    head.train()
    n3_grad = torch.randn(1, 128, 80, 80, requires_grad=True)
    n4_grad = torch.randn(1, 128, 40, 40, requires_grad=True)
    n5_grad = torch.randn(1, 128, 20, 20, requires_grad=True)

    out_g = head(n3_grad, n4_grad, n5_grad)
    dummy_loss = (
        sum(b.sum() for b in out_g.box_preds) +
        sum(o.sum() for o in out_g.obj_preds) +
        sum(c.sum() for c in out_g.cls_preds)
    )
    dummy_loss.backward()

    assert n3_grad.grad is not None, "n3 gradient is None"
    assert n4_grad.grad is not None, "n4 gradient is None"
    assert n5_grad.grad is not None, "n5 gradient is None"
    assert not torch.isnan(n3_grad.grad).any(), "NaN in n3 gradient"
    assert not torch.isnan(n4_grad.grad).any(), "NaN in n4 gradient"
    assert not torch.isnan(n5_grad.grad).any(), "NaN in n5 gradient"
    print("  Input gradients (n3, n4, n5) computed successfully without NaNs: PASSED")

    # Verify parameter gradients
    for name, param in head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
    print("  All parameter gradients verified: PASSED")
    print("=" * 75)
    print("All decoupled detection head tests passed successfully!")
