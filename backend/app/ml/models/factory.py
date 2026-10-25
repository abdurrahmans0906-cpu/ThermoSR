import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from app.ml.models.cnn_baseline import ThermalCNNBaseline
from app.ml.models.rgb_fusion import RGBFusionModel
from app.ml.models.reliability_fusion import ReliabilityFusionModel
from app.core.logging import logger

SUPPORTED_ARCHITECTURES = {
    "thermal_cnn_baseline": ThermalCNNBaseline,
    "rgb_fusion": RGBFusionModel,
    "reliability_aware_fusion": ReliabilityFusionModel,
}

SUPPORTED_SCALES = [2, 4, 8, 16]

def build_model(
    architecture: str,
    scale_factor: int = 8,
    **kwargs
) -> nn.Module:
    """
    Factory function to instantiate validated ThermoSR model architectures.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Must be one of: {list(SUPPORTED_ARCHITECTURES.keys())}"
        )
    if scale_factor not in SUPPORTED_SCALES:
        raise ValueError(
            f"Unsupported scale factor {scale_factor}. "
            f"Must be one of: {SUPPORTED_SCALES}"
        )

    model_class = SUPPORTED_ARCHITECTURES[architecture]
    model = model_class(scale_factor=scale_factor, **kwargs)
    return model

def validate_checkpoint_file(checkpoint_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates checkpoint file integrity safely without executing arbitrary code.
    Uses PyTorch 2.0+ weights_only=True.
    """
    if not os.path.isfile(checkpoint_path):
        return False, f"File does not exist: {checkpoint_path}", {}

    try:
        # Load strictly weights and metadata
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            return False, "Checkpoint root is not a dictionary", {}
        
        required_keys = ["model_state_dict"]
        for key in required_keys:
            if key not in ckpt:
                return False, f"Missing required key '{key}' in checkpoint", {}

        metadata = {
            "architecture": ckpt.get("architecture", "unknown"),
            "scale_factor": ckpt.get("scale_factor", 8),
            "epoch": ckpt.get("epoch", 0),
            "best_psnr": ckpt.get("best_psnr", None),
            "best_ssim": ckpt.get("best_ssim", None),
            "created_at": ckpt.get("created_at", None),
        }
        return True, "Checkpoint is valid and intact", metadata

    except Exception as e:
        logger.error(f"Checkpoint integrity validation failed: {e}")
        return False, f"Failed to parse checkpoint: {str(e)}", {}

def load_trained_model(
    checkpoint_path: str,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads model weights safely and returns model in eval mode on target device.
    """
    is_valid, msg, meta = validate_checkpoint_file(checkpoint_path)
    if not is_valid:
        raise ValueError(f"Invalid checkpoint: {msg}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = ckpt.get("architecture", "thermal_cnn_baseline")
    scale_factor = ckpt.get("scale_factor", 8)

    model = build_model(architecture=architecture, scale_factor=scale_factor)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
