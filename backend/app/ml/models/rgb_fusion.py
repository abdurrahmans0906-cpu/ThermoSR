import torch
import torch.nn as nn
import torch.nn.functional as F
from app.ml.models.common import ResidualBlock, UpsampleHead

class RGBFusionModel(nn.Module):
    """
    Model B: Standard RGB-Guided Thermal Super-Resolution with Feature Fusion.
    Extracts high-frequency spatial guidance from high-resolution RGB imagery and
    fuses it with thermal features via concatenation and convolutional fusion.
    """
    def __init__(self, scale_factor: int = 8, num_channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels

        # Thermal feature branch
        self.thermal_head = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.thermal_body = nn.Sequential(*[ResidualBlock(channels=num_channels) for _ in range(num_blocks // 2)])
        self.thermal_upsample = UpsampleHead(in_channels=num_channels, scale_factor=scale_factor)

        # RGB guidance branch (already at target high-resolution)
        self.rgb_head = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.rgb_body = nn.Sequential(*[ResidualBlock(channels=num_channels) for _ in range(num_blocks // 2)])

        # Fusion trunk (concatenates 64 thermal + 64 RGB = 128 channels)
        self.fusion_conv = nn.Conv2d(num_channels * 2, num_channels, kernel_size=1)
        self.fusion_body = nn.Sequential(*[ResidualBlock(channels=num_channels) for _ in range(num_blocks)])

        # Final reconstruction layer
        self.tail = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

    def forward(self, lr_thermal: torch.Tensor, rgb_guidance: torch.Tensor) -> torch.Tensor:
        # Base bicubic thermal baseline
        base_upsampled = F.interpolate(
            lr_thermal,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        # Thermal feature extraction and upsampling to HR coordinate grid
        t_feat = self.thermal_head(lr_thermal)
        t_feat = self.thermal_body(t_feat)
        t_feat_up = self.thermal_upsample(t_feat)  # [B, 64, sH, sW]

        # RGB feature extraction at HR grid
        rgb_feat = self.rgb_head(rgb_guidance)
        rgb_feat = self.rgb_body(rgb_feat)          # [B, 64, sH, sW]

        # Standard Concatenation & Fusion
        fused = torch.cat([t_feat_up, rgb_feat], dim=1)  # [B, 128, sH, sW]
        fused = self.fusion_conv(fused)                  # [B, 64, sH, sW]
        fused = self.fusion_body(fused) + t_feat_up      # Residual addition with thermal features

        residual = self.tail(fused)
        out = base_upsampled + residual
        return torch.clamp(out, 0.0, 1.0)
