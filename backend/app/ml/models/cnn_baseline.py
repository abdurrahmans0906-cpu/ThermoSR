import torch
import torch.nn as nn
import torch.nn.functional as F
from app.ml.models.common import ResidualBlock, UpsampleHead

class ThermalCNNBaseline(nn.Module):
    """
    Model A: Thermal-Only CNN Super-Resolution Baseline.
    Processes low-resolution single-channel thermal input through deep residual feature
    extraction and reconstructs high-resolution thermal intensity with PixelShuffle.
    """
    def __init__(self, scale_factor: int = 8, num_channels: int = 64, num_blocks: int = 8):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels

        # Initial shallow feature extraction (1 -> num_channels)
        self.head = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)

        # Deep residual feature trunk
        blocks = [ResidualBlock(channels=num_channels) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.body_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Cascaded PixelShuffle upsampling
        self.upsample = UpsampleHead(in_channels=num_channels, scale_factor=scale_factor)

        # Final reconstruction layer (num_channels -> 1)
        self.tail = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

    def forward(self, lr_thermal: torch.Tensor, rgb_guidance: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass. rgb_guidance is accepted for uniform interface but unused in Model A.
        """
        # Global bicubic skip connection from LR thermal
        base_upsampled = F.interpolate(
            lr_thermal,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )

        feat_head = self.head(lr_thermal)
        feat_body = self.body_conv(self.body(feat_head)) + feat_head
        feat_up = self.upsample(feat_body)
        residual = self.tail(feat_up)

        out = base_upsampled + residual
        return torch.clamp(out, 0.0, 1.0)
