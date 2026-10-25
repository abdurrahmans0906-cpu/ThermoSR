import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
import lpips
from typing import Dict, Any, Optional
from app.core.logging import logger

# Lazy singleton for LPIPS to avoid reloading AlexNet weights on every call
_lpips_vgg: Optional[lpips.LPIPS] = None

def get_lpips_model() -> lpips.LPIPS:
    global _lpips_vgg
    if _lpips_vgg is None:
        try:
            _lpips_vgg = lpips.LPIPS(net='alex', verbose=False).eval()
        except Exception as e:
            logger.warning(f"Could not initialize LPIPS (network download may be disabled or cached): {e}")
            _lpips_vgg = None
    return _lpips_vgg

def calculate_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is better.
    """
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))

def calculate_ssim(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes Structural Similarity Index (SSIM) in range [-1, 1].
    Higher is better.
    """
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[:, :, 0]
    if target.ndim == 3 and target.shape[2] == 1:
        target = target[:, :, 0]
        
    return float(ssim_fn(pred, target, data_range=max_val))

def calculate_lpips(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS).
    Lower is better.
    Inputs must be in range [0, 1].
    """
    model = get_lpips_model()
    if model is None:
        return None

    try:
        # Convert 1-channel or 3-channel [H, W] to [1, 3, H, W] in range [-1, 1]
        if pred.ndim == 2:
            p_3 = np.repeat(pred[np.newaxis, np.newaxis, :, :], 3, axis=1)
            t_3 = np.repeat(target[np.newaxis, np.newaxis, :, :], 3, axis=1)
        elif pred.ndim == 3:
            p_3 = np.transpose(pred, (2, 0, 1))[np.newaxis, ...]
            t_3 = np.transpose(target, (2, 0, 1))[np.newaxis, ...]
            if p_3.shape[1] == 1:
                p_3 = np.repeat(p_3, 3, axis=1)
                t_3 = np.repeat(t_3, 3, axis=1)
        else:
            return None

        # Scale [0, 1] -> [-1, 1]
        t_p = torch.from_numpy(p_3).float() * 2.0 - 1.0
        t_t = torch.from_numpy(t_3).float() * 2.0 - 1.0

        with torch.no_grad():
            score = model(t_p, t_t).item()
        return float(round(score, 4))
    except Exception as e:
        logger.warning(f"LPIPS computation skipped: {e}")
        return None

def calculate_thermal_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    lr_input: Optional[np.ndarray] = None,
    scale_factor: int = 8
) -> Dict[str, Any]:
    """
    Calculates full suite of quantitative evaluation metrics.
    Ensures scientific rigor: metrics are computed only against actual ground truth.
    """
    # Validate dimensions
    if pred.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"Shape mismatch for metric calculation: pred {pred.shape[:2]} vs target {target.shape[:2]}"
        )

    p_norm = np.clip(pred.astype(np.float32), 0.0, 1.0)
    t_norm = np.clip(target.astype(np.float32), 0.0, 1.0)

    psnr = calculate_psnr(p_norm, t_norm)
    ssim = calculate_ssim(p_norm, t_norm)
    lp = calculate_lpips(p_norm, t_norm)

    # Thermal Intensity & Gradient Fidelity
    abs_diff = np.abs(p_norm - t_norm)
    mae = float(np.mean(abs_diff))
    max_error = float(np.max(abs_diff))

    results = {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "lpips": lp,
        "mae": round(mae, 4),
        "max_error": round(max_error, 4),
        "has_ground_truth": True
    }

    if lr_input is not None:
        # Measure thermal flux conservation via downsampling consistency
        import cv2
        lr_down = cv2.resize(p_norm, (lr_input.shape[1], lr_input.shape[0]), interpolation=cv2.INTER_AREA)
        cycle_l1 = float(np.mean(np.abs(lr_down - lr_input)))
        results["thermal_consistency_l1"] = round(cycle_l1, 5)

    return results
