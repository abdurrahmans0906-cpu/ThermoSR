import os
import time
import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from app.ml.models.factory import build_model, load_trained_model
from app.services.preprocessing import (
    load_thermal_image,
    load_rgb_image,
    match_aspect_and_crop,
    downsample_thermal,
    render_false_color,
    render_reliability_map,
    render_error_map,
)
from app.services.metrics import calculate_thermal_metrics
from app.services.artifacts import save_image_artifact, save_json_artifact
from app.core.logging import logger

def run_reconstruction_pipeline(
    run_id: str,
    task_mode: str,
    model_architecture: str,
    scale_factor: int,
    thermal_input_path: str,
    rgb_input_path: Optional[str] = None,
    reference_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    colormap: str = "inferno",
    progress_callback = None
) -> Dict[str, Any]:
    """
    Executes end-to-end thermal reconstruction inference with real deep learning models,
    quantitative metric calculation, and artifact generation.
    """
    if progress_callback:
        progress_callback("validating", 10)

    # Resolve compute device
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Step 1: Preprocessing & Image Loading
    if progress_callback:
        progress_callback("preprocessing", 25)

    gt_thermal = None
    if task_mode == "downsample_eval":
        # Thermal input was uploaded as the HR ground-truth reference
        gt_thermal = load_thermal_image(thermal_input_path)
        lr_thermal = downsample_thermal(gt_thermal, scale_factor=scale_factor)
    else:
        lr_thermal = load_thermal_image(thermal_input_path)
        if reference_path and os.path.exists(reference_path):
            gt_thermal = load_thermal_image(reference_path)

    # Determine target HR dimensions
    lr_h, lr_w = lr_thermal.shape[:2]
    target_hr_h = lr_h * scale_factor
    target_hr_w = lr_w * scale_factor

    # Prepare RGB guidance if required by architecture
    rgb_aligned = None
    if model_architecture in ["rgb_fusion", "reliability_aware_fusion"]:
        if not rgb_input_path or not os.path.exists(rgb_input_path):
            raise ValueError(f"RGB guidance image is required for '{model_architecture}'")
        
        rgb_raw = load_rgb_image(rgb_input_path)
        rgb_aligned, aspect_meta = match_aspect_and_crop(rgb_raw, target_width=target_hr_w, target_height=target_hr_h)

    # Step 2: Load Model or Instantiate
    if progress_callback:
        progress_callback("running_inference", 50)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model, _ = load_trained_model(checkpoint_path, device=target_device)
        is_trained_inference = True
    else:
        # Build initialized model
        model = build_model(architecture=model_architecture, scale_factor=scale_factor)
        model.to(target_device)
        model.eval()
        is_trained_inference = False

    # Convert inputs to PyTorch tensors
    t_lr_tensor = torch.from_numpy(lr_thermal).unsqueeze(0).unsqueeze(0).float().to(target_device)
    rgb_tensor = None
    if rgb_aligned is not None:
        rgb_tensor = torch.from_numpy(rgb_aligned).permute(2, 0, 1).unsqueeze(0).float().to(target_device)

    # Step 3: Run Inference
    start_time = time.perf_counter()
    reliability_np = None

    with torch.no_grad():
        if model_architecture == "thermal_cnn_baseline":
            pred_tensor = model(t_lr_tensor)
        elif model_architecture == "rgb_fusion":
            pred_tensor = model(t_lr_tensor, rgb_tensor)
        elif model_architecture == "reliability_aware_fusion":
            pred_tensor, rel_tensor = model(t_lr_tensor, rgb_tensor, return_reliability=True)
            reliability_np = rel_tensor.squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unknown architecture {model_architecture}")

    inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    pred_np = pred_tensor.squeeze().cpu().numpy()

    # Step 4: Calculate Metrics
    if progress_callback:
        progress_callback("calculating_metrics", 80)

    metrics = {
        "inference_time_ms": inference_time_ms,
        "input_dimensions": [lr_w, lr_h],
        "output_dimensions": [target_hr_w, target_hr_h],
        "is_trained_model": is_trained_inference,
        "device_used": target_device,
        "has_ground_truth": False
    }

    error_map_np = None
    if gt_thermal is not None:
        # If ground-truth dimensions don't match exactly, resize target to match scale factor expectation
        if gt_thermal.shape[:2] != (target_hr_h, target_hr_w):
            gt_thermal = cv2.resize(gt_thermal, (target_hr_w, target_hr_h), interpolation=cv2.INTER_CUBIC)
        
        computed_metrics = calculate_thermal_metrics(
            pred=pred_np,
            target=gt_thermal,
            lr_input=lr_thermal,
            scale_factor=scale_factor
        )
        metrics.update(computed_metrics)
        error_map_np = render_error_map(pred_np, gt_thermal)

    # Step 5: Render Visualizations and Save Artifacts
    reconstructed_rel = save_image_artifact(pred_np, "reconstructions", f"{run_id}_reconstructed.png")
    false_color_np = render_false_color(pred_np, colormap_name=colormap)
    false_color_rel = save_image_artifact(false_color_np, "reconstructions", f"{run_id}_false_color.png", is_rgb=True)

    reliability_rel = None
    if reliability_np is not None:
        reliability_vis = render_reliability_map(reliability_np)
        reliability_rel = save_image_artifact(reliability_vis, "reconstructions", f"{run_id}_reliability_map.png", is_rgb=True)

    error_map_rel = None
    if error_map_np is not None:
        error_map_rel = save_image_artifact(error_map_np, "reconstructions", f"{run_id}_error_map.png", is_rgb=True)

    artifacts = {
        "reconstructed": reconstructed_rel,
        "false_color": false_color_rel,
        "reliability_map": reliability_rel,
        "error_map": error_map_rel,
    }

    if progress_callback:
        progress_callback("completed", 100)

    return {
        "status": "completed",
        "artifacts": artifacts,
        "metrics": metrics,
        "inference_time_ms": inference_time_ms
    }
