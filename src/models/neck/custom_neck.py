"""
Custom Multi-Scale Feature-Fusion Neck for Indian Road Object Detection.

Designed specifically for Indian road conditions:
- Small, distant objects (traffic signs, lights, pedestrians, bicycles, motorcycles).
- Dense traffic with overlapping instances and heavy occlusion.
- High aspect ratio diversity (tall pedestrians/light poles vs. wide barricades/buses).
- Varied object scales simultaneously present in the scene.

Key Architectural Components:
1. AdaptiveScaleFusion (ASF):
   - Learned spatial and channel scale gating mechanism using softmax across scales.
   - Replaces static concatenation/addition with dynamic scale weighting.
2. RoadContextAggregator (RCA):
   - Combines horizontal strip depthwise convolutions (1x7 for lane/queue traffic flow)
     and vertical strip depthwise convolutions (7x1 for pedestrians and traffic poles)
     with 2D dilated depthwise convolutions (3x3, d=2).
3. HighResDetailEnhancer:
   - Preserves and sharpens high-frequency boundary details for small objects in N3.
4. RoadFusionBlock:
   - Deep multi-receptive feature refinement block combining local 3x3 depthwise,
     asymmetric strip convolutions, and Squeeze-and-Excitation channel gating.
5. NeckDownsampler:
   - Anti-aliased dual-path downsampler for bottom-up localization flow.
6. IndianRoadNeck:
   - Bidirectional (top-down semantic + bottom-up localization) feature fusion network.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AdaptiveScaleFusion(nn.Module):
    """
    Learned Adaptive Feature Fusion across scales.
    
    In traditional FPN/PAN architectures (e.g. YOLOv8), features from adjacent scales
    are simply concatenated or added together. This assumes equal importance across
    all pixels and channels. In dense Indian traffic scenes, however, fine high-resolution
    boundaries for small signs/pedestrians are easily washed out by upsampled coarse context,
    while occluded objects need strong semantic guidance.
    
    This module computes dynamic scale gating weights:
      - Channel-wise scale attention: Evaluates which channels benefit from high-level semantics
        vs. low-level details via global average pooling and 1x1 convolutions.
      - Spatial scale attention: Evaluates which pixel locations (e.g., small object regions vs.
        broad background road) require localized features vs. broad context.
      - Softmax across scales: Guarantees a learned partition of unity (weights sum to 1.0 per element).
      - Depthwise separable residual refinement: Smoothly refines the fused representation.
      
    Args:
        channels: Feature dimension for both input streams and output.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        
        # Channel-wise scale attention
        mid_c = max(16, channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, mid_c, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_c, channels * 2, kernel_size=1),
        )
        
        # Spatial scale attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid_c, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, groups=mid_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_c, 2, kernel_size=1),
        )
        
        # Depthwise-separable refinement
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, act=False),
        )
        self.final_act = nn.SiLU(inplace=True)

    def forward(self, f_primary: torch.Tensor, f_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_primary: Primary feature map at target resolution [B, C, H, W].
            f_context: Context feature map resized to target resolution [B, C, H, W].
            
        Returns:
            Adaptively fused feature map [B, C, H, W].
        """
        b, c, h, w = f_primary.shape
        cat_feat = torch.cat([f_primary, f_context], dim=1)  # [B, 2C, H, W]
        
        # 1. Channel-wise scale attention weights: [B, 2, C, 1, 1]
        c_raw = self.channel_gate(cat_feat).view(b, 2, c, 1, 1)
        c_weights = F.softmax(c_raw, dim=1)
        
        # 2. Spatial scale attention weights: [B, 2, 1, H, W]
        s_raw = self.spatial_gate(cat_feat).view(b, 2, 1, h, w)
        s_weights = F.softmax(s_raw, dim=1)
        
        # 3. Joint scale weights normalized across scale dimension
        joint_weights = c_weights * s_weights  # [B, 2, C, H, W]
        joint_weights = joint_weights / (joint_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        w_primary = joint_weights[:, 0]  # [B, C, H, W]
        w_context = joint_weights[:, 1]  # [B, C, H, W]
        
        # 4. Adaptive scale blending
        fused = w_primary * f_primary + w_context * f_context
        
        # 5. Residual refinement
        refined = self.refine(fused)
        return self.final_act(fused + refined)


class RoadContextAggregator(nn.Module):
    """
    Efficient Road Context Aggregator for dense road scenarios.
    
    Exploits structural geometric patterns prevalent in Indian road environments:
    1. Horizontal Strip Convolution (1x7): Models horizontal traffic flow, multi-lane queues,
       barricades, and vehicles cutting across paths.
    2. Vertical Strip Convolution (7x1): Models vertically oriented structures including
       pedestrians, electricity poles, traffic lights, and street signs.
    3. Dilated Depthwise Convolution (3x3, d=2): Captures 2D neighborhood context without
       reducing resolution or blurring object edges.
    4. Global Channel Excitation: Gated scene descriptor modulating contextual feature flow.
    
    Args:
        channels: Feature dimension.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        
        # Horizontal strip depthwise convolution
        self.h_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        
        # Vertical strip depthwise convolution
        self.v_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        
        # Dilated 3x3 depthwise convolution (dilation=2, receptive field 5x5)
        self.dilated = ConvBNAct(channels, channels, kernel_size=3, dilation=2, groups=channels)
        
        # Channel excitation gate
        mid_c = max(16, channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_c, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_c, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # Linear projection and fusion
        self.project = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_feat = self.h_strip(x)
        v_feat = self.v_strip(x)
        d_feat = self.dilated(x)
        
        context_combined = h_feat + v_feat + d_feat
        gated = context_combined * self.channel_gate(context_combined)
        return x + self.project(gated)


class HighResDetailEnhancer(nn.Module):
    """
    Preserves and sharpens high-frequency boundary details for small objects in N3.
    
    Extracts high-pass spatial cues (e.g. sharp edges of traffic signs, distant pedestrians,
    two-wheeler handlebars) from pristine lateral P3 features and injects them back into
    the semantically enriched N3 representation with a learnable gate.
    
    Args:
        channels: Feature dimension.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.lowpass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.detail_conv = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.refine = ConvBNAct(channels, channels, kernel_size=3, groups=channels)

    def forward(self, n3_fused: torch.Tensor, p3_lateral: torch.Tensor) -> torch.Tensor:
        # High-pass edge details: original minus local average
        edge_detail = p3_lateral - self.lowpass(p3_lateral)
        edge_feat = self.detail_conv(edge_detail)
        # Learnable residual detail injection
        enhanced = n3_fused + self.alpha * edge_feat
        return self.refine(enhanced)


class RoadFusionBlock(nn.Module):
    """
    Multi-receptive feature refinement block for neck stages.
    
    Combines local 3x3 depthwise convolution with asymmetric strip convolutions
    (1x5 and 5x1) and Squeeze-and-Excitation channel gating to refine fused features
    without the quadratic computational overhead of heavy standard convolutions.
    
    Args:
        channels: Feature dimension.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        half_c = channels // 2
        
        # Local 3x3 depthwise branch
        self.b1 = ConvBNAct(half_c, half_c, kernel_size=3, groups=half_c)
        
        # Strip asymmetric branch (1x5 and 5x1) for elongated road objects
        self.b2 = nn.Sequential(
            nn.Conv2d(half_c, half_c, kernel_size=(1, 5), padding=(0, 2), groups=half_c, bias=False),
            nn.BatchNorm2d(half_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(half_c, half_c, kernel_size=(5, 1), padding=(2, 0), groups=half_c, bias=False),
            nn.BatchNorm2d(half_c),
            nn.SiLU(inplace=True),
        )
        
        # Channel attention gating for occlusion resilience
        mid_c = max(16, channels // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_c, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_c, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.project = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        o1 = self.b1(x1)
        o2 = self.b2(x2)
        fused = torch.cat([o1, o2], dim=1)
        gated = fused * self.se(fused)
        return x + self.project(gated)


class NeckDownsampler(nn.Module):
    """
    Anti-aliased dual-path downsampler for the bottom-up localization pathway.
    
    Combines depthwise strided 3x3 convolution and 2x2 max-pooling to downsample
    feature maps by 2x while preventing high-frequency spatial aliasing.
    
    Args:
        channels: Feature dimension.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        half_c = channels // 2
        self.conv_path = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, stride=2, groups=channels),
            ConvBNAct(channels, half_c, kernel_size=1),
        )
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(channels, half_c, kernel_size=1),
        )
        self.fuse = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_conv = self.conv_path(x)
        feat_pool = self.pool_path(x)
        return self.fuse(torch.cat([feat_conv, feat_pool], dim=1))


class AnisotropicTrafficDisentangler(nn.Module):
    """
    Anisotropic Traffic Disentangler (ATD) for dense multi-object road environments.
    
    Specifically engineered for Indian road scenes to solve two major structural failure modes:
    1. Vertical elevation coupling: Resolves riders mounted on top of motorcycles and scooters
       via vertical strip depthwise convolutions (kernel 5x1) without lateral bleed.
    2. Lateral vehicle crowding: Resolves side-by-side queues of motorcycles and cars in narrow lanes
       via horizontal strip depthwise convolutions (kernel 1x5) without vertical bleed.
    3. Cross-Directional Gating:
       Computes reciprocal gating masks where horizontal features gate the vertical stream
       and vertical features gate the horizontal stream:
         F_fused = (V * G_h) + (H * G_v)
         X_out = X + gamma * Proj(F_fused)
       where gamma is a learnable parameter initialized to 0.1.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        
        # Asymmetric strip depthwise convolutions
        self.v_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.h_strip = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        
        # Lightweight cross-gating projections
        mid_dim = max(32, channels // 2)
        self.gate_v = nn.Sequential(
            nn.Conv2d(channels, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.gate_h = nn.Sequential(
            nn.Conv2d(channels, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        
        # Feature projection and residual scale
        self.proj = ConvBNAct(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_strip(x)
        h = self.h_strip(x)
        g_v = self.gate_v(v)
        g_h = self.gate_h(h)
        fused = (v * g_h) + (h * g_v)
        return x + self.gamma * self.proj(fused)


class SelectiveSpatialDetailPathway(nn.Module):
    """
    Selective Spatial Detail Pathway (SSDP) for Small-Object Representation.
    
    Extracts high-frequency spatial edge cues from high-resolution P2 (stride 4, 160x160),
    downsamples them using an anti-aliased depthwise compressor, and selectively injects
    salient high-resolution details directly into N3 (stride 8, 80x80) via a spatial gate.
    
    Prevents tiny and distant objects (pedestrians, traffic signs, distant motorcycles)
    from being erased by deep strided convolutions without the compute explosion of a full P2 head.
    
    Args:
        in_channels: Input channels from P2 (default: 64).
        out_channels: Unified neck channels for N3 (default: 128).
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 128) -> None:
        super().__init__()
        # 1. High-pass spatial edge extractor
        self.edge_filter = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # 2. Anti-aliased spatial compressor (160x160 -> 80x80)
        self.compressor = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels),
            ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1),
        )
        
        # 3. Spatial salience gate: identifies coordinates with concentrated high-frequency traffic cues
        self.salience_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        # 4. Learnable residual injection scale
        self.gamma = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        high_pass = p2 - self.edge_filter(p2)
        compressed = self.compressor(high_pass)
        salience = self.salience_gate(compressed)
        return self.gamma * (compressed * salience)


class IndianRoadNeck(nn.Module):
    """
    Custom Multi-Scale Feature-Fusion Neck for Indian Road Object Detection.
    
    Accepts backbone features:
      - P3: [B, 128, 80, 80] (1/8 input resolution)
      - P4: [B, 256, 40, 40] (1/16 input resolution)
      - P5: [B, 512, 20, 20] (1/32 input resolution)
      
    Outputs unified feature pyramid suitable for a decoupled detection head:
      - N3: [B, neck_channels, 80, 80]  (High resolution, small objects: signs, pedestrians, bikes)
      - N4: [B, neck_channels, 40, 40]  (Medium resolution: cars, auto-rickshaws, riders)
      - N5: [B, neck_channels, 20, 20]  (Low resolution, large objects: buses, trucks, road context)
      
    Information Flow:
      1. Lateral Projections: P3, P4, P5 projected to unified `neck_channels`.
      2. Top-Down Semantic Flow: P5_context -> Up -> ASF(P4, Up(P5)) -> Up -> ASF(P3, Up(P4)).
      3. High-Res Preservation: HighResDetailEnhancer injects crisp edge gradients into N3.
      4. Bottom-Up Localization Flow: N3 -> Down -> ASF(P4_td, Down(N3)) -> Down -> ASF(P5_td, Down(N4)).
      5. Anisotropic Traffic Disentangler (ATD): Cross-gates orthogonal horizontal and vertical features on N3 and N4.
      
    Args:
        in_channels: Tuple of input channel dimensions (default: (128, 256, 512)).
        neck_channels: Unified channel dimension for neck outputs (default: 128).
        num_refine_blocks: Number of RoadFusionBlocks per stage (default: 1).
        use_atd: Whether to enable Anisotropic Traffic Disentangler on N3 and N4 (default: False).
    """
    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (128, 256, 512),
        neck_channels: int = 128,
        num_refine_blocks: int = 1,
        use_atd: bool = False,
        use_ssdp: bool = False,
        p2_channels: int = 64,
    ) -> None:
        super().__init__()
        c3_in, c4_in, c5_in = in_channels
        self.in_channels = in_channels
        self.neck_channels = neck_channels
        self.use_atd = use_atd
        self.use_ssdp = use_ssdp
        self.out_channels: List[int] = [neck_channels, neck_channels, neck_channels]
        self.out_strides: List[int] = [8, 16, 32]
        
        # 1. Lateral Projections to unified neck dimension
        self.lat3 = ConvBNAct(c3_in, neck_channels, kernel_size=1)
        self.lat4 = ConvBNAct(c4_in, neck_channels, kernel_size=1)
        self.lat5 = ConvBNAct(c5_in, neck_channels, kernel_size=1)
        
        # 2. Top-Down Semantic Pathway
        self.p5_rca = RoadContextAggregator(neck_channels)
        self.up_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_asf_4 = AdaptiveScaleFusion(neck_channels)
        self.td_refine_4 = nn.Sequential(*[
            RoadFusionBlock(neck_channels) for _ in range(num_refine_blocks)
        ])
        
        self.up_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_asf_3 = AdaptiveScaleFusion(neck_channels)
        self.td_refine_3 = nn.Sequential(*[
            RoadFusionBlock(neck_channels) for _ in range(num_refine_blocks)
        ])
        
        # 3. High-Resolution Detail Preservation for N3
        self.n3_enhancer = HighResDetailEnhancer(neck_channels)
        
        # 3b. Selective Spatial Detail Pathway (SSDP) from high-res P2 (stride 4)
        if self.use_ssdp:
            self.ssdp = SelectiveSpatialDetailPathway(in_channels=p2_channels, out_channels=neck_channels)
        
        # 4. Bottom-Up Localization Pathway
        self.down_n3 = NeckDownsampler(neck_channels)
        self.bu_asf_4 = AdaptiveScaleFusion(neck_channels)
        self.bu_refine_4 = nn.Sequential(*[
            RoadFusionBlock(neck_channels) for _ in range(num_refine_blocks)
        ])
        
        self.down_n4 = NeckDownsampler(neck_channels)
        self.bu_asf_5 = AdaptiveScaleFusion(neck_channels)
        self.bu_refine_5 = nn.Sequential(*[
            RoadFusionBlock(neck_channels) for _ in range(num_refine_blocks)
        ])
        self.n5_rca = RoadContextAggregator(neck_channels)
        
        # 5. Anisotropic Traffic Disentangler for N3 (stride 8) and N4 (stride 16)
        if self.use_atd:
            self.n3_atd = AnisotropicTrafficDisentangler(neck_channels)
            self.n4_atd = AnisotropicTrafficDisentangler(neck_channels)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
        p2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the neck.
        
        Args:
            p3: Feature map at 1/8 scale [B, C3, H/8, W/8].
            p4: Feature map at 1/16 scale [B, C4, H/16, W/16].
            p5: Feature map at 1/32 scale [B, C5, H/32, W/32].
            p2: Optional feature map at 1/4 scale [B, C2, H/4, W/4] for SSDP.
            
        Returns:
            Tuple of (N3, N4, N5):
              - N3: [B, neck_channels, H/8, W/8]
              - N4: [B, neck_channels, H/16, W/16]
              - N5: [B, neck_channels, H/32, W/32]
        """
        # Step 1: Lateral projections
        lat3 = self.lat3(p3)  # [B, neck_channels, 80, 80]
        lat4 = self.lat4(p4)  # [B, neck_channels, 40, 40]
        lat5 = self.lat5(p5)  # [B, neck_channels, 20, 20]
        
        # Step 2: Top-down semantic enrichment
        td5 = self.p5_rca(lat5)                                       # [B, C, 20, 20]
        up_td5 = self.up_p5(td5)                                      # [B, C, 40, 40]
        td4 = self.td_refine_4(self.td_asf_4(lat4, up_td5))          # [B, C, 40, 40]
        
        up_td4 = self.up_p4(td4)                                      # [B, C, 80, 80]
        td3 = self.td_refine_3(self.td_asf_3(lat3, up_td4))          # [B, C, 80, 80]
        
        # Step 3: High-resolution detail enhancement for N3
        n3 = self.n3_enhancer(td3, lat3)                              # [B, C, 80, 80]
        
        # Step 3b: Selective high-resolution detail injection from P2
        if self.use_ssdp and p2 is not None:
            n3 = n3 + self.ssdp(p2)
        
        # Step 4: Bottom-up localization enrichment
        down_n3 = self.down_n3(n3)                                    # [B, C, 40, 40]
        n4 = self.bu_refine_4(self.bu_asf_4(td4, down_n3))           # [B, C, 40, 40]
        
        down_n4 = self.down_n4(n4)                                    # [B, C, 20, 20]
        n5_pre = self.bu_refine_5(self.bu_asf_5(td5, down_n4))       # [B, C, 20, 20]
        n5 = self.n5_rca(n5_pre)                                      # [B, C, 20, 20]
        
        # Step 5: Anisotropic traffic disentanglement (N3 and N4)
        if self.use_atd:
            n3 = self.n3_atd(n3)
            n4 = self.n4_atd(n4)
        
        return n3, n4, n5


def build_neck(
    in_channels: Tuple[int, int, int] = (128, 256, 512),
    neck_channels: int = 128,
    num_refine_blocks: int = 1,
    use_atd: bool = False,
    use_ssdp: bool = False,
) -> IndianRoadNeck:
    """Helper factory function to construct an IndianRoadNeck instance."""
    return IndianRoadNeck(
        in_channels=in_channels,
        neck_channels=neck_channels,
        num_refine_blocks=num_refine_blocks,
        use_atd=use_atd,
        use_ssdp=use_ssdp,
    )


if __name__ == "__main__":
    print("=" * 75)
    print(" Indian Road Custom Neck: Standalone Verification Test ")
    print("=" * 75)

    neck = IndianRoadNeck(
        in_channels=(128, 256, 512),
        neck_channels=128,
        num_refine_blocks=1
    )
    neck.eval()

    # 1. Parameter counts
    trainable_params = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in neck.parameters())
    print(f"Total parameters:           {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters:       {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Out channels:               {neck.out_channels}")
    print(f"Out strides:                {neck.out_strides}")
    print("-" * 75)

    # 2. Test Batch Size 1
    print("Testing Batch Size = 1:")
    p3_b1 = torch.randn(1, 128, 80, 80)
    p4_b1 = torch.randn(1, 256, 40, 40)
    p5_b1 = torch.randn(1, 512, 20, 20)

    with torch.no_grad():
        n3_b1, n4_b1, n5_b1 = neck(p3_b1, p4_b1, p5_b1)

    print(f"  Input P3: {tuple(p3_b1.shape)} -> Output N3: {tuple(n3_b1.shape)}")
    print(f"  Input P4: {tuple(p4_b1.shape)} -> Output N4: {tuple(n4_b1.shape)}")
    print(f"  Input P5: {tuple(p5_b1.shape)} -> Output N5: {tuple(n5_b1.shape)}")

    # Verify B=1 shapes
    assert n3_b1.shape == (1, 128, 80, 80), f"N3 shape mismatch: {n3_b1.shape}"
    assert n4_b1.shape == (1, 128, 40, 40), f"N4 shape mismatch: {n4_b1.shape}"
    assert n5_b1.shape == (1, 128, 20, 20), f"N5 shape mismatch: {n5_b1.shape}"

    # Check for NaN / Inf
    for name, tensor in [("N3", n3_b1), ("N4", n4_b1), ("N5", n5_b1)]:
        assert not torch.isnan(tensor).any(), f"NaN detected in {name} (batch=1)"
        assert not torch.isinf(tensor).any(), f"Inf detected in {name} (batch=1)"
    print("  Resolution check [80x80, 40x40, 20x20]: PASSED")
    print("  NaN/Inf check: PASSED (All finite values)")
    print("-" * 75)

    # 3. Test Batch Size 2
    print("Testing Batch Size = 2:")
    p3_b2 = torch.randn(2, 128, 80, 80)
    p4_b2 = torch.randn(2, 256, 40, 40)
    p5_b2 = torch.randn(2, 512, 20, 20)

    with torch.no_grad():
        n3_b2, n4_b2, n5_b2 = neck(p3_b2, p4_b2, p5_b2)

    print(f"  Input P3: {tuple(p3_b2.shape)} -> Output N3: {tuple(n3_b2.shape)}")
    print(f"  Input P4: {tuple(p4_b2.shape)} -> Output N4: {tuple(n4_b2.shape)}")
    print(f"  Input P5: {tuple(p5_b2.shape)} -> Output N5: {tuple(n5_b2.shape)}")

    assert n3_b2.shape == (2, 128, 80, 80), f"N3 shape mismatch: {n3_b2.shape}"
    assert n4_b2.shape == (2, 128, 40, 40), f"N4 shape mismatch: {n4_b2.shape}"
    assert n5_b2.shape == (2, 128, 20, 20), f"N5 shape mismatch: {n5_b2.shape}"

    for name, tensor in [("N3", n3_b2), ("N4", n4_b2), ("N5", n5_b2)]:
        assert not torch.isnan(tensor).any(), f"NaN detected in {name} (batch=2)"
        assert not torch.isinf(tensor).any(), f"Inf detected in {name} (batch=2)"
    print("  Batch 2 verification: PASSED")
    print("-" * 75)

    # 4. Gradient backward pass verification
    print("Testing Gradient Backward Pass:")
    neck.train()
    p3_grad = torch.randn(1, 128, 80, 80, requires_grad=True)
    p4_grad = torch.randn(1, 256, 40, 40, requires_grad=True)
    p5_grad = torch.randn(1, 512, 20, 20, requires_grad=True)

    n3_g, n4_g, n5_g = neck(p3_grad, p4_grad, p5_grad)
    dummy_loss = n3_g.sum() + n4_g.sum() + n5_g.sum()
    dummy_loss.backward()

    assert p3_grad.grad is not None, "p3 gradient is None"
    assert p4_grad.grad is not None, "p4 gradient is None"
    assert p5_grad.grad is not None, "p5 gradient is None"
    assert not torch.isnan(p3_grad.grad).any(), "NaN in p3 gradient"
    assert not torch.isnan(p4_grad.grad).any(), "NaN in p4 gradient"
    assert not torch.isnan(p5_grad.grad).any(), "NaN in p5 gradient"
    print("  Input gradients (p3, p4, p5) computed successfully without NaNs: PASSED")

    # Verify parameter gradients
    for name, param in neck.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
    print("  All parameter gradients verified: PASSED")
    print("=" * 75)
    print("All neck verification tests passed successfully!")
