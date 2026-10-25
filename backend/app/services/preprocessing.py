import os
import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, Dict, Any, Optional
from app.core.logging import logger

def load_thermal_image(file_path: str) -> np.ndarray:
    """
    Loads thermal image preserving intensity fidelity and bit depth.
    Returns 2D float32 numpy array normalized to [0.0, 1.0].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Thermal image not found: {file_path}")

    # Use cv2.IMREAD_UNCHANGED to preserve 16-bit or native channels
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode thermal image: {file_path}")

    if img.ndim == 3:
        # If saved as 3-channel (e.g. JPEG FLIR), extract single channel
        img = img[:, :, 0]

    if img.dtype == np.uint16:
        # 16-bit thermal sensor format (FLIR Tau2 / Boson)
        img_min = float(img.min())
        img_max = float(img.max())
        if img_max > img_min:
            img_norm = (img.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            img_norm = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img_norm = img.astype(np.float32) / 255.0
    else:
        img_norm = np.clip(img.astype(np.float32), 0.0, 1.0)

    return img_norm

def load_rgb_image(file_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Loads RGB guidance image as RGB float32 in [0.0, 1.0].
    target_size: (width, height) if resizing or aligning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RGB image not found: {file_path}")

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode RGB image: {file_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size is not None and (img_rgb.shape[1] != target_size[0] or img_rgb.shape[0] != target_size[1]):
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    return (img_rgb.astype(np.float32) / 255.0)

def match_aspect_and_crop(
    rgb: np.ndarray,
    target_width: int,
    target_height: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Handles aspect ratio differences (e.g., 1280x1024 [5:4] to 640x480 [4:3]).
    Crops RGB symmetrically to match target aspect ratio before resizing.
    """
    h, w = rgb.shape[:2]
    target_aspect = target_width / target_height
    curr_aspect = w / h

    meta = {
        "original_shape": [w, h],
        "target_shape": [target_width, target_height],
        "original_aspect": round(curr_aspect, 3),
        "target_aspect": round(target_aspect, 3),
        "applied_crop": False
    }

    if abs(curr_aspect - target_aspect) > 0.01:
        # Aspect mismatch: perform center-crop
        if curr_aspect > target_aspect:
            # Current is wider than target: crop width
            new_w = int(h * target_aspect)
            offset = (w - new_w) // 2
            rgb_cropped = rgb[:, offset:offset + new_w]
        else:
            # Current is taller than target: crop height
            new_h = int(w / target_aspect)
            offset = (h - new_h) // 2
            rgb_cropped = rgb[offset:offset + new_h, :]

        meta["applied_crop"] = True
        meta["cropped_shape"] = [rgb_cropped.shape[1], rgb_cropped.shape[0]]
    else:
        rgb_cropped = rgb

    # Resize to exact target dimension
    rgb_aligned = cv2.resize(rgb_cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return rgb_aligned, meta

def downsample_thermal(hr_thermal: np.ndarray, scale_factor: int = 8) -> np.ndarray:
    """
    Synthesizes low-resolution thermal input from high-resolution ground truth
    using standard bicubic downsampling.
    """
    h, w = hr_thermal.shape[:2]
    lr_w = w // scale_factor
    lr_h = h // scale_factor
    lr_img = cv2.resize(hr_thermal, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    return np.clip(lr_img, 0.0, 1.0)

def extract_paired_patch(
    hr_thermal: np.ndarray,
    rgb: np.ndarray,
    patch_size_hr: int = 256,
    scale_factor: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts aligned patches from HR thermal, HR RGB, and generates LR thermal patch.
    """
    h, w = hr_thermal.shape[:2]
    patch_size_hr = min(patch_size_hr, h, w)
    # Ensure patch size is divisible by scale factor
    patch_size_hr = (patch_size_hr // scale_factor) * scale_factor

    y0 = np.random.randint(0, h - patch_size_hr + 1)
    x0 = np.random.randint(0, w - patch_size_hr + 1)

    t_hr_patch = hr_thermal[y0:y0 + patch_size_hr, x0:x0 + patch_size_hr]
    rgb_patch = rgb[y0:y0 + patch_size_hr, x0:x0 + patch_size_hr]
    t_lr_patch = downsample_thermal(t_hr_patch, scale_factor=scale_factor)

    return t_lr_patch, rgb_patch, t_hr_patch

def apply_paired_augmentation(
    t_lr: np.ndarray,
    rgb: np.ndarray,
    t_hr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies synchronized geometric transformations to preserve paired pixel alignment.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        t_lr = np.fliplr(t_lr).copy()
        rgb = np.fliplr(rgb).copy()
        t_hr = np.fliplr(t_hr).copy()

    # Random vertical flip
    if np.random.rand() > 0.5:
        t_lr = np.flipud(t_lr).copy()
        rgb = np.flipud(rgb).copy()
        t_hr = np.flipud(t_hr).copy()

    # Random 90 deg rotation
    rot_k = np.random.randint(0, 4)
    if rot_k > 0:
        t_lr = np.rot90(t_lr, rot_k).copy()
        rgb = np.rot90(rgb, rot_k).copy()
        t_hr = np.rot90(t_hr, rot_k).copy()

    return t_lr, rgb, t_hr

def render_false_color(
    thermal_img: np.ndarray,
    colormap_name: str = "inferno"
) -> np.ndarray:
    """
    Converts 1-channel thermal intensity [0, 1] to false-color RGB for visualization.
    Uses documented OpenCV colormaps.
    """
    cmaps = {
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    cmap_id = cmaps.get(colormap_name.lower(), cv2.COLORMAP_INFERNO)
    
    t_u8 = (np.clip(thermal_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(t_u8, cmap_id)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb

def render_reliability_map(reliability: np.ndarray) -> np.ndarray:
    """
    Converts model's learned reliability gate weights [0, 1] to an interpretable heatmap.
    Uses cv2.COLORMAP_TURBO or COLORMAP_JET where red/yellow=high confidence, blue/purple=low confidence.
    """
    if reliability.ndim == 3:
        if reliability.shape[0] == 1:
            reliability = reliability[0]
        elif reliability.shape[2] == 1:
            reliability = reliability[:, :, 0]
            
    r_u8 = (np.clip(reliability, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Turbo or Viridis colormap for scientific confidence display
    color_bgr = cv2.applyColorMap(r_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

def render_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Renders absolute error heatmap between prediction and ground truth.
    """
    diff = np.abs(np.clip(pred, 0.0, 1.0) - np.clip(target, 0.0, 1.0))
    # Normalize error dynamically to emphasize subtle differences
    max_err = max(float(diff.max()), 0.01)
    diff_norm = (diff / max_err * 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
