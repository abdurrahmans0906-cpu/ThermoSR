import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ThermalGradientLoss(nn.Module):
    """
    Measures edge gradient fidelity using Sobel filters to preserve thermal structural boundaries.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Assumes single channel [B, 1, H, W]
        grad_x_pred = F.conv2d(pred, self.sobel_x, padding=1)
        grad_y_pred = F.conv2d(pred, self.sobel_y, padding=1)
        grad_x_target = F.conv2d(target, self.sobel_x, padding=1)
        grad_y_target = F.conv2d(target, self.sobel_y, padding=1)

        loss_x = F.l1_loss(grad_x_pred, grad_x_target)
        loss_y = F.l1_loss(grad_y_pred, grad_y_target)
        return loss_x + loss_y

class ThermalConsistencyLoss(nn.Module):
    """
    Downsampling Cycle Consistency Loss:
    Enforces physical conservation of thermal energy/intensity. When the high-resolution
    prediction is downsampled back to LR resolution via area pooling, it must closely
    reproduce the original input low-resolution thermal measurement.
    """
    def __init__(self, scale_factor: int = 8):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, pred_hr: torch.Tensor, input_lr: torch.Tensor) -> torch.Tensor:
        downsampled = F.adaptive_avg_pool2d(pred_hr, input_lr.shape[-2:])
        return F.l1_loss(downsampled, input_lr)

class CompositeThermalLoss(nn.Module):
    """
    Total objective function for ThermoSR training:
    L_total = L1 + lambda_grad * L_grad + lambda_consist * L_consist
    """
    def __init__(
        self,
        scale_factor: int = 8,
        w_l1: float = 1.0,
        w_grad: float = 0.1,
        w_consist: float = 0.05,
        use_grad: bool = True,
        use_consist: bool = True,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_grad = w_grad if use_grad else 0.0
        self.w_consist = w_consist if use_consist else 0.0
        
        self.l1_loss = nn.L1Loss()
        self.grad_loss = ThermalGradientLoss() if use_grad else None
        self.consist_loss = ThermalConsistencyLoss(scale_factor=scale_factor) if use_consist else None

    def forward(
        self,
        pred_hr: torch.Tensor,
        target_hr: torch.Tensor,
        input_lr: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_l1 = self.l1_loss(pred_hr, target_hr)
        total_loss = self.w_l1 * loss_l1

        loss_dict = {"loss_l1": float(loss_l1.item())}

        if self.grad_loss is not None and self.w_grad > 0:
            loss_g = self.grad_loss(pred_hr, target_hr)
            total_loss = total_loss + self.w_grad * loss_g
            loss_dict["loss_grad"] = float(loss_g.item())
        else:
            loss_dict["loss_grad"] = 0.0

        if self.consist_loss is not None and self.w_consist > 0:
            loss_c = self.consist_loss(pred_hr, input_lr)
            total_loss = total_loss + self.w_consist * loss_c
            loss_dict["loss_consist"] = float(loss_c.item())
        else:
            loss_dict["loss_consist"] = 0.0

        loss_dict["loss_total"] = float(total_loss.item())
        return total_loss, loss_dict
