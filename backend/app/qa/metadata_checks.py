from typing import Tuple, List, Dict, Any, Optional
from app.qa.schemas import QAMetadataInput

def perform_deterministic_checks(meta: QAMetadataInput) -> Tuple[str, List[str], List[str], Dict[str, bool]]:
    """
    Executes pure deterministic validation on evaluation metadata.
    Returns:
        (deterministic_status, warnings, missing_evidence, checks_dict)
    
    Status can be:
        - ready_for_review
        - human_review
        - insufficient_evidence
    """
    warnings: List[str] = list(meta.deterministic_warnings or [])
    missing_evidence: List[str] = []
    checks: Dict[str, bool] = {}

    # Check 1: LR Shape validity
    lr_w, lr_h = meta.thermal_lr_shape[0], meta.thermal_lr_shape[1]
    checks["positive_dimensions"] = lr_w > 0 and lr_h > 0

    # Check 2: Scale Factor Support
    checks["supported_scale_factor"] = meta.scale_factor in [2, 4, 8, 16]

    # Check 3: LR to HR Dimension Consistency (accounting for padding/rounding +/- 2px)
    if meta.thermal_hr_shape:
        hr_w, hr_h = meta.thermal_hr_shape[0], meta.thermal_hr_shape[1]
        expected_hr_w = lr_w * meta.scale_factor
        expected_hr_h = lr_h * meta.scale_factor
        
        diff_w = abs(hr_w - expected_hr_w)
        diff_h = abs(hr_h - expected_hr_h)

        if diff_w <= 2 and diff_h <= 2:
            checks["dimension_scale_consistent"] = True
        else:
            checks["dimension_scale_consistent"] = False
            warnings.append(
                f"Dimension/scale inconsistency: LR [{lr_w}x{lr_h}] at x{meta.scale_factor} "
                f"expects ~[{expected_hr_w}x{expected_hr_h}], but declared HR is [{hr_w}x{hr_h}]."
            )
    else:
        checks["dimension_scale_consistent"] = True
        missing_evidence.append("High-resolution thermal output shape (thermal_hr_shape) not explicitly recorded.")

    # Check 4: RGB guidance shape & alignment
    if meta.rgb_shape is not None:
        checks["rgb_dimensions_available"] = True
    else:
        checks["rgb_dimensions_available"] = False
        if meta.alignment_status in ["verified", "failed"]:
            warnings.append("Alignment status specified, but RGB shape metadata is missing.")

    # Check 5: Alignment Status Verification
    # CRITICAL: Equal dimensions do NOT prove optical/spatial alignment
    if meta.alignment_status == "failed":
        checks["alignment_verified"] = False
        warnings.append("RGB/Thermal optical alignment explicitly failed. Multimodal fusion may hallucinate misaligned edges.")
    elif meta.alignment_status == "unverified":
        checks["alignment_verified"] = False
        warnings.append("RGB/Thermal spatial alignment is unverified. Note: Matching image dimensions does NOT establish optical registration.")
    elif meta.alignment_status == "verified":
        checks["alignment_verified"] = True
    else:
        checks["alignment_verified"] = False

    # Check 6: Ground Truth Reference Availability
    checks["ground_truth_available"] = meta.ground_truth_available
    if not meta.ground_truth_available:
        missing_evidence.append("High-resolution ground-truth thermal reference not provided.")
    
    # Check 7: Evaluation Metrics Completeness & Bounds
    metrics = meta.metrics or {}
    has_psnr = metrics.get("psnr") is not None
    has_ssim = metrics.get("ssim") is not None
    has_lpips = metrics.get("lpips") is not None

    checks["has_psnr"] = has_psnr
    checks["has_ssim"] = has_ssim
    checks["has_lpips"] = has_lpips

    if meta.ground_truth_available:
        if not has_psnr:
            missing_evidence.append("PSNR metric missing despite ground-truth availability.")
        if not has_ssim:
            missing_evidence.append("SSIM metric missing despite ground-truth availability.")
        if not has_lpips:
            missing_evidence.append("LPIPS perceptual metric missing.")

    # Determine Deterministic Assessment Status
    # Rule A: Critical failures mandate human_review
    if not checks["supported_scale_factor"] or meta.alignment_status == "failed":
        deterministic_status = "human_review"
    elif not checks.get("dimension_scale_consistent", True):
        deterministic_status = "human_review"
    # Rule B: Missing ground-truth or missing core metrics mandates insufficient_evidence
    elif not meta.ground_truth_available or (not has_psnr and not has_ssim):
        deterministic_status = "insufficient_evidence"
    # Rule C: If unverified alignment, flag for human_review if using RGB guidance
    elif meta.alignment_status == "unverified" and meta.rgb_shape is not None:
        deterministic_status = "human_review"
    # Rule D: Fully documented and verified record
    else:
        deterministic_status = "ready_for_review"

    return deterministic_status, warnings, missing_evidence, checks
