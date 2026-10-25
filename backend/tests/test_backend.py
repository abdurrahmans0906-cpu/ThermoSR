import pytest
import os
import torch
import numpy as np
from pathlib import Path

from app.core.config import settings
from app.ml.models.factory import build_model, validate_checkpoint_file
from app.ml.models.cnn_baseline import ThermalCNNBaseline
from app.ml.models.rgb_fusion import RGBFusionModel
from app.ml.models.reliability_fusion import ReliabilityFusionModel
from app.ml.losses.thermal_loss import CompositeThermalLoss, ThermalGradientLoss, ThermalConsistencyLoss
from app.services.metrics import calculate_psnr, calculate_ssim, calculate_thermal_metrics
from app.services.preprocessing import (
    downsample_thermal,
    match_aspect_and_crop,
    render_false_color,
    render_reliability_map,
    render_error_map,
)
from app.services.artifacts import get_artifact_path

def test_config_paths():
    assert settings.PROJECT_NAME == "ThermoSR"
    assert settings.STORAGE_DIR.exists()
    assert settings.CHECKPOINTS_DIR.exists()

def test_model_a_forward():
    model = build_model("thermal_cnn_baseline", scale_factor=8)
    lr = torch.randn(2, 1, 32, 32)
    out = model(lr)
    assert out.shape == (2, 1, 256, 256)
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_model_b_forward():
    model = build_model("rgb_fusion", scale_factor=8)
    lr = torch.randn(2, 1, 32, 32)
    rgb = torch.randn(2, 3, 256, 256)
    out = model(lr, rgb)
    assert out.shape == (2, 1, 256, 256)
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_model_c_forward():
    model = build_model("reliability_aware_fusion", scale_factor=8)
    lr = torch.randn(2, 1, 32, 32)
    rgb = torch.randn(2, 3, 256, 256)
    out, rel = model(lr, rgb, return_reliability=True)
    assert out.shape == (2, 1, 256, 256)
    assert rel.shape == (2, 1, 256, 256)
    assert rel.min() >= 0.0 and rel.max() <= 1.0
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_loss_functions():
    loss_fn = CompositeThermalLoss(scale_factor=8)
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    lr = torch.rand(2, 1, 8, 8)

    loss, loss_dict = loss_fn(pred, target, lr)
    assert loss.item() > 0.0
    assert "loss_total" in loss_dict
    assert "loss_l1" in loss_dict
    assert "loss_grad" in loss_dict
    assert "loss_consist" in loss_dict

def test_metrics_calculation():
    target = np.ones((64, 64), dtype=np.float32) * 0.5
    pred = target.copy()

    psnr = calculate_psnr(pred, target)
    assert psnr == 100.0

    ssim = calculate_ssim(pred, target)
    assert ssim >= 0.99

    metrics = calculate_thermal_metrics(pred, target)
    assert metrics["psnr"] == 100.0
    assert metrics["ssim"] >= 0.99
    assert metrics["mae"] == 0.0

def test_preprocessing_aspect_and_crop():
    rgb = np.ones((1024, 1280, 3), dtype=np.float32)  # 5:4 aspect
    aligned, meta = match_aspect_and_crop(rgb, target_width=640, target_height=480)  # 4:3 aspect
    assert aligned.shape == (480, 640, 3)
    assert meta["applied_crop"] is True

def test_preprocessing_downsample():
    hr = np.ones((480, 640), dtype=np.float32)
    lr = downsample_thermal(hr, scale_factor=8)
    assert lr.shape == (60, 80)

def test_render_colormaps():
    thermal = np.random.rand(60, 80).astype(np.float32)
    false_color = render_false_color(thermal, "inferno")
    assert false_color.shape == (60, 80, 3)

    rel_map = render_reliability_map(thermal)
    assert rel_map.shape == (60, 80, 3)

    err_map = render_error_map(thermal, thermal)
    assert err_map.shape == (60, 80, 3)

def test_safe_checkpoint_validation():
    # Test nonexistent
    is_valid, msg, _ = validate_checkpoint_file("nonexistent.pth")
    assert is_valid is False

def test_path_traversal_prevention():
    # Attempt directory traversal
    p = get_artifact_path("../../windows/system32/cmd.exe")
    assert p is None
