"""
Custom PyTorch Backbone for Indian Road Object Detection.

Designed specifically for Indian road conditions:
- Dense traffic and heavy occlusion (vehicles overlapping, lane indiscipline).
- Wide scale variations (small objects like traffic signs, pedestrians, two-wheelers vs. large buses/trucks).
- Diverse aspect ratios (tall pedestrians/utility poles vs. elongated barricades/vehicles).

Key Architectural Components:
1. DetailPreservingStem: Dual-path stem retaining high-frequency edges and contrast.
2. DualPathDownsample: Anti-aliasing downsampling combining depthwise strided convolutions and max-pooling.
3. MultiReceptiveBlock (MRB): Custom feature extraction block with:
   - Local detail branch (depthwise 3x3)
   - Asymmetric strip convolutions (1x5 and 5x1) for tall/wide objects
   - Dilated context branch (depthwise 3x3, dilation=2)
   - Squeeze-and-Excitation (SE) channel gating for occlusion filtering
4. MultiScaleContextBlock (MSCB): Hierarchical multi-dilation depthwise context aggregator (dilations 1, 2, 4).
5. IndianRoadBackbone: Produces multi-scale feature pyramids (P3: 1/8, P4: 1/16, P5: 1/32).
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """
    Standard Convolution + BatchNorm + Activation block.
    
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


class DetailPreservingStem(nn.Module):
    """
    Dual-path detail-preserving stem.
    
    Instead of aggressive single-step downsampling, this stem uses a parallel
    convolutional path (capturing smooth textural gradients) and a max-pooling
    path (capturing peak contrast and sharp edges). This preserves fine details
    critical for small objects (e.g. traffic signs, distant pedestrians, two-wheelers).
    
    Input:  [B, 3, H, W]
    Output: [B, out_channels, H/2, W/2]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        
        # Convolution branch: progressive 3x3 convolutions with stride 2
        self.conv_branch = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=2),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, stride=1),
        )
        # Pooling branch: max-pool preserves extreme spatial cues (e.g. thin lane marks, bright lights)
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(in_channels, mid_channels, kernel_size=1, stride=1),
        )
        # Fusion of both representation paths
        self.fuse = ConvBNAct(mid_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_conv = self.conv_branch(x)
        feat_pool = self.pool_branch(x)
        return self.fuse(torch.cat([feat_conv, feat_pool], dim=1))


class DualPathDownsample(nn.Module):
    """
    Anti-aliased dual-path downsampling module.
    
    Combines depthwise strided convolution with max-pooling to downsample spatial
    dimensions by 2x while doubling or altering channel depth. Prevents information
    loss common in single strided convolution downsamplers.
    
    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H/2, W/2]
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        
        # Depthwise strided convolution path
        self.conv_path = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels),
            ConvBNAct(in_channels, mid_channels, kernel_size=1, stride=1),
        )
        # Max-pooling path
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(in_channels, mid_channels, kernel_size=1, stride=1),
        )
        # Channel fusion
        self.out_conv = ConvBNAct(mid_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_conv = self.conv_path(x)
        feat_pool = self.pool_path(x)
        return self.out_conv(torch.cat([feat_conv, feat_pool], dim=1))


class MultiReceptiveBlock(nn.Module):
    """
    Custom Feature Extraction Block tailored for dense, occluded road scenes.
    
    Rather than standard CSP/C2f bottlenecks, this block implements:
    1. Pointwise channel expansion.
    2. Multi-Branch Receptive Field Decomposition:
       - Branch 1 (Local Details): 3x3 depthwise conv for local geometry (handlebars, faces, wheels).
       - Branch 2 (Asymmetric Strip Convolutions): 1x5 and 5x1 depthwise convs designed
         specifically for tall objects (pedestrians, poles) and wide objects (cars, barricades).
       - Branch 3 (Dilated Context): 3x3 depthwise conv with dilation=2 (effective RF 5x5)
         capturing local surrounding traffic context without extra parameters.
    3. Squeeze-and-Excitation (SE) Channel Attention: Dynamically recalibrates channel
       responses to suppress background road clutter and focus on occluded targets.
    4. Pointwise projection + residual shortcut.
    """
    def __init__(self, channels: int, expansion: float = 1.0) -> None:
        super().__init__()
        hidden_dim = int(channels * expansion)
        self.expand = ConvBNAct(channels, hidden_dim, kernel_size=1)
        
        # Distribute channels across 3 specialized branches
        b1_c = hidden_dim // 3
        b2_c = hidden_dim // 3
        b3_c = hidden_dim - b1_c - b2_c
        self.branch_splits = [b1_c, b2_c, b3_c]
        
        # Branch 1: Standard 3x3 depthwise for local details
        self.b1_local = ConvBNAct(b1_c, b1_c, kernel_size=3, groups=b1_c)
        
        # Branch 2: Asymmetric strip depthwise convolutions (1x5 and 5x1)
        self.b2_strip = nn.Sequential(
            nn.Conv2d(b2_c, b2_c, kernel_size=(1, 5), padding=(0, 2), groups=b2_c, bias=False),
            nn.BatchNorm2d(b2_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(b2_c, b2_c, kernel_size=(5, 1), padding=(2, 0), groups=b2_c, bias=False),
            nn.BatchNorm2d(b2_c),
            nn.SiLU(inplace=True),
        )
        
        # Branch 3: Dilated 3x3 depthwise (dilation=2, effective RF=5x5) for local context
        self.b3_dilated = ConvBNAct(b3_c, b3_c, kernel_size=3, dilation=2, groups=b3_c)
        
        # Squeeze-and-Excitation channel attention for occlusion resilience
        reduced_dim = max(16, hidden_dim // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, reduced_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Projection back to target channels
        self.project = ConvBNAct(hidden_dim, channels, kernel_size=1, act=False)
        self.final_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.expand(x)
        x1, x2, x3 = torch.split(feat, self.branch_splits, dim=1)
        
        o1 = self.b1_local(x1)
        o2 = self.b2_strip(x2)
        o3 = self.b3_dilated(x3)
        
        fused = torch.cat([o1, o2, o3], dim=1)
        gated = fused * self.se(fused)
        projected = self.project(gated)
        
        return self.final_act(x + projected)


class MultiScaleContextBlock(nn.Module):
    """
    Custom Multi-Scale Context Mechanism using efficient hierarchical depthwise convolutions.
    
    In dense traffic scenes, context from surrounding vehicles and the road layout is
    essential to disambiguate occluded or overlapping instances.
    
    This module splits input features into 4 equal channel groups:
      - Group 0: Identity branch (preserves fine-grained details unchanged).
      - Group 1: 3x3 Depthwise Conv (dilation=1, RF=3x3).
      - Group 2: 3x3 Depthwise Conv (dilation=2, RF=5x5), cascaded with Group 1.
      - Group 3: 3x3 Depthwise Conv (dilation=4, RF=9x9), cascaded with Group 2.
      
    The outputs are concatenated and fused with a 1x1 projection, followed by a residual shortcut.
    This creates an efficient multi-receptive field context aggregation without high parameter overhead.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        g1 = channels // 4
        g2 = channels // 4
        g3 = channels // 4
        g0 = channels - g1 - g2 - g3
        self.splits = [g0, g1, g2, g3]
        
        self.dw1 = ConvBNAct(g1, g1, kernel_size=3, groups=g1, dilation=1)
        self.dw2 = ConvBNAct(g2, g2, kernel_size=3, groups=g2, dilation=2)
        self.dw3 = ConvBNAct(g3, g3, kernel_size=3, groups=g3, dilation=4)
        self.fuse = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3 = torch.split(x, self.splits, dim=1)
        
        y1 = self.dw1(x1)
        y2 = self.dw2(x2 + y1)
        y3 = self.dw3(x3 + y2)
        
        fused = torch.cat([x0, y1, y2, y3], dim=1)
        return x + self.fuse(fused)


class IndianRoadBackbone(nn.Module):
    """
    Custom PyTorch Backbone for Indian Road Object Detection.
    
    Generates three multi-scale feature maps:
      - P3: 1/8 input resolution  (e.g., 80x80 for 640x640 input)
      - P4: 1/16 input resolution (e.g., 40x40 for 640x640 input)
      - P5: 1/32 input resolution (e.g., 20x20 for 640x640 input)
      
    Architecture Overview:
      Input (3, H, W)
        |
      DetailPreservingStem (stride 2) -> (32, H/2, W/2)
        |
      Stage 1 [P2] (Downsample stride 2 + 2x MRB) -> (64, H/4, W/4)
        |
      Stage 2 [P3] (Downsample stride 2 + 3x MRB + MSCB) -> (128, H/8, W/8)  --> Output P3
        |
      Stage 3 [P4] (Downsample stride 2 + 4x MRB + MSCB) -> (256, H/16, W/16) --> Output P4
        |
      Stage 4 [P5] (Downsample stride 2 + 3x MRB + MSCB) -> (512, H/32, W/32) --> Output P5
      
    Args:
        in_channels: Number of input image channels (default: 3).
        stem_channels: Channels after stem stage (default: 32).
        stage_channels: Tuple of channels for [P2, P3, P4, P5] (default: (64, 128, 256, 512)).
        stage_depths: Tuple of block counts for [P2, P3, P4, P5] (default: (2, 3, 4, 3)).
    """
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        stage_depths: Tuple[int, int, int, int] = (2, 3, 4, 3),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        
        # Output channels and strides for P3, P4, P5 (for neck integration)
        self.out_channels: List[int] = [stage_channels[1], stage_channels[2], stage_channels[3]]
        self.out_strides: List[int] = [8, 16, 32]
        
        # Stem: High-resolution detail preservation (stride 2: 640 -> 320)
        self.stem = DetailPreservingStem(in_channels=in_channels, out_channels=stem_channels)
        
        # Stage 1 [P2]: 1/4 resolution (320 -> 160)
        # Keeps high-resolution representations intact for edge/texture features
        c_p2 = stage_channels[0]
        self.stage1_down = DualPathDownsample(stem_channels, c_p2)
        self.stage1_blocks = nn.Sequential(*[
            MultiReceptiveBlock(c_p2) for _ in range(stage_depths[0])
        ])
        
        # Stage 2 [P3]: 1/8 resolution (160 -> 80) - Small object detection level
        c_p3 = stage_channels[1]
        self.stage2_down = DualPathDownsample(c_p2, c_p3)
        self.stage2_blocks = nn.Sequential(
            *[MultiReceptiveBlock(c_p3) for _ in range(stage_depths[1])],
            MultiScaleContextBlock(c_p3)
        )
        
        # Stage 3 [P4]: 1/16 resolution (80 -> 40) - Medium object detection level
        c_p4 = stage_channels[2]
        self.stage3_down = DualPathDownsample(c_p3, c_p4)
        self.stage3_blocks = nn.Sequential(
            *[MultiReceptiveBlock(c_p4) for _ in range(stage_depths[2])],
            MultiScaleContextBlock(c_p4)
        )
        
        # Stage 4 [P5]: 1/32 resolution (40 -> 20) - Large object / global context level
        c_p5 = stage_channels[3]
        self.stage4_down = DualPathDownsample(c_p4, c_p5)
        self.stage4_blocks = nn.Sequential(
            *[MultiReceptiveBlock(c_p5) for _ in range(stage_depths[3])],
            MultiScaleContextBlock(c_p5)
        )
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal distribution for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_p2: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape [B, 3, H, W].
            return_p2: Whether to also return high-resolution P2 (1/4 scale). Default False.
            
        Returns:
            Tuple of (P3, P4, P5) feature maps:
              - P3: [B, C_p3, H/8, W/8]
              - P4: [B, C_p4, H/16, W/16]
              - P5: [B, C_p5, H/32, W/32]
            If return_p2 is True, returns (P2, P3, P4, P5).
        """
        # Stem: [B, stem_c, H/2, W/2]
        x_stem = self.stem(x)
        
        # Stage 1 [P2]: [B, c_p2, H/4, W/4]
        p2 = self.stage1_blocks(self.stage1_down(x_stem))
        
        # Stage 2 [P3]: [B, c_p3, H/8, W/8]
        p3 = self.stage2_blocks(self.stage2_down(p2))
        
        # Stage 3 [P4]: [B, c_p4, H/16, W/16]
        p4 = self.stage3_blocks(self.stage3_down(p3))
        
        # Stage 4 [P5]: [B, c_p5, H/32, W/32]
        p5 = self.stage4_blocks(self.stage4_down(p4))
        
        if return_p2:
            return p2, p3, p4, p5
        return p3, p4, p5


def build_backbone(
    in_channels: int = 3,
    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
    stage_depths: Tuple[int, int, int, int] = (2, 3, 4, 3),
) -> IndianRoadBackbone:
    """Helper factory function to construct an IndianRoadBackbone instance."""
    return IndianRoadBackbone(
        in_channels=in_channels,
        stage_channels=stage_channels,
        stage_depths=stage_depths,
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" Indian Road Custom Backbone: Standalone Verification Test ")
    print("=" * 70)

    # 1. Instantiate the backbone
    model = IndianRoadBackbone()
    model.eval()

    # 2. Create random input tensor [1, 3, 640, 640]
    input_shape = (1, 3, 640, 640)
    x = torch.randn(*input_shape)
    print(f"Input tensor shape: {tuple(x.shape)}")

    # 3. Forward pass
    with torch.no_grad():
        p3, p4, p5 = model(x)

    # 4. Print output shapes
    print(f"P3 (1/8  resolution) shape: {tuple(p3.shape)}")
    print(f"P4 (1/16 resolution) shape: {tuple(p4.shape)}")
    print(f"P5 (1/32 resolution) shape: {tuple(p5.shape)}")

    # 5. Calculate and print parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 70)
    print(f"Total parameters:             {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Total trainable parameters:   {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Out channels:                 {model.out_channels}")
    print(f"Out strides:                  {model.out_strides}")
    print("-" * 70)

    # 6. Verify resolutions
    b, c3, h3, w3 = p3.shape
    b, c4, h4, w4 = p4.shape
    b, c5, h5, w5 = p5.shape

    assert (h3, w3) == (80, 80), f"Expected P3 resolution (80, 80), got ({h3}, {w3})"
    assert (h4, w4) == (40, 40), f"Expected P4 resolution (40, 40), got ({h4}, {w4})"
    assert (h5, w5) == (20, 20), f"Expected P5 resolution (20, 20), got ({h5}, {w5})"
    assert (c3, c4, c5) == (128, 256, 512), f"Expected channels (128, 256, 512), got ({c3}, {c4}, {c5})"
    print("Resolution verification: PASSED [80x80, 40x40, 20x20]")
    print("Channel verification:    PASSED [128, 256, 512]")

    # 7. Backward pass test (Gradient check)
    model.train()
    x_grad = torch.randn(*input_shape, requires_grad=True)
    p3_g, p4_g, p5_g = model(x_grad)
    dummy_loss = p3_g.sum() + p4_g.sum() + p5_g.sum()
    dummy_loss.backward()
    assert x_grad.grad is not None, "Gradient check failed: input grad is None"
    print("Gradient backward check: PASSED")
    print("=" * 70)
    print("All backbone tests passed successfully!")
