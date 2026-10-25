import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from app.ml.models.common import ResidualBlock, UpsampleHead

class SpatialReliabilityGate(nn.Module):
    """
    Learned Spatial Reliability Estimator:
    Receives upsampled thermal features and high-resolution RGB features.
    Computes spatial correlation and structural divergence, producing a continuous
    reliability confidence weight R(x,y) in [0, 1].
    
    R(x,y) -> 1.0: RGB features are strongly aligned and consistent with thermal cues.
    R(x,y) -> 0.0: RGB guidance is noisy, inconsistent (e.g. shadow/texture hallucination),
                   or spatially misaligned, suppressing RGB contribution.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        # Alignment & correlation projection
        self.thermal_proj = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.rgb_proj = nn.Conv2d(channels, channels // 2, kernel_size=1)

        # Gate prediction convolutional layers
        # Input channels: 32 (thermal_proj) + 32 (rgb_proj) + 32 (absolute difference) = 96
        self.gate_conv = nn.Sequential(
            nn.Conv2d((channels // 2) * 3, 32, kernel_size=3, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Bound confidence strictly to [0.0, 1.0]
        )

    def forward(self, t_feat: torch.Tensor, rgb_feat: torch.Tensor) -> torch.Tensor:
        p_t = self.thermal_proj(t_feat)
        p_rgb = self.rgb_proj(rgb_feat)
        diff = torch.abs(p_t - p_rgb)

        corr_input = torch.cat([p_t, p_rgb, diff], dim=1)
        reliability_map = self.gate_conv(corr_input)
        return reliability_map

class ReliabilityFusionModel(nn.Module):
    """
    Model C (Proposed Contribution): Reliability-Aware RGB-Guided Thermal Super-Resolution.
    Integrates the Spatial Reliability Gate to adaptively modulate RGB feature injection,
    preventing edge hallucination and mitigating cross-modality misalignment.
    """
    def __init__(self, scale_factor: int = 8, num_channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels

        # Thermal feature branch
        self.thermal_head = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.thermal_body = nn.Sequential(*[ResidualBlock(channels=num_channels) for _ in range(num_blocks // 2)])
        self.thermal_upsample = UpsampleHead(in_channels=num_channels, scale_factor=scale_factor)

        # RGB guidance branch
        self.rgb_head = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.rgb_body = nn.Sequential(*[ResidualBlock(channels=num_channels) for _ in range(num_blocks // 2)])

        # Spatial Reliability Estimation Gate
        self.reliability_gate = SpatialReliabilityGate(channels=num_channels)

        # Cross-modality fusion trunk
        self.rgb_adapter = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.fusion_conv = nn.Conv2d(num_channels * 2, num_channels, kernel_size=1)
        self.fusion_body = nn.Sequential(*[ResidualBlock(channels=num_channels) for _ in range(num_blocks)])

        # Final reconstruction layer
        self.tail = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        lr_thermal: torch.Tensor,
        rgb_guidance: torch.Tensor,
        return_reliability: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Base bicubic thermal baseline
        base_upsampled = F.interpolate(
            lr_thermal,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Thermal feature extraction & upsampling
        t_feat = self.thermal_head(lr_thermal)
        t_feat = self.thermal_body(t_feat)
        t_feat_up = self.thermal_upsample(t_feat)  # [B, 64, sH, sW]

        # RGB guidance feature extraction
        rgb_feat = self.rgb_head(rgb_guidance)
        rgb_feat = self.rgb_body(rgb_feat)          # [B, 64, sH, sW]

        # Calculate spatial reliability confidence map R in [0, 1]
        reliability_map = self.reliability_gate(t_feat_up, rgb_feat)  # [B, 1, sH, sW]

        # Adaptively modulate RGB features using learned reliability weights
        gated_rgb = self.rgb_adapter(rgb_feat * reliability_map)

        # Fuse reliability-gated RGB with thermal backbone
        fused = torch.cat([t_feat_up, gated_rgb], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.fusion_body(fused) + t_feat_up

        residual = self.tail(fused)
        reconstructed = torch.clamp(base_upsampled + residual, 0.0, 1.0)

        if return_reliability:
            return reconstructed, reliability_map
        return reconstructed
