import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True)
        self.act = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.act(self.conv1(x)))
        return x + self.res_scale * res

class UpsampleHead(nn.Module):
    """Cascaded PixelShuffle upsampling head for scales 2, 4, 8, 16."""
    def __init__(self, in_channels: int = 64, scale_factor: int = 8):
        super().__init__()
        self.scale_factor = scale_factor
        layers = []
        curr_scale = 1
        
        while curr_scale < scale_factor:
            step = 2
            layers.append(nn.Conv2d(in_channels, in_channels * (step ** 2), kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(step))
            layers.append(nn.PReLU(in_channels))
            curr_scale *= step

        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)
