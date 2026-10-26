import time
from typing import Dict, Any, List, Optional
from app.core.logging import logger
from app.qa.schemas import ExperimentRecommendationRequest
from app.qa.laya_adapter import laya_adapter

def recommend_next_experiment(req: ExperimentRecommendationRequest) -> Dict[str, Any]:
    """
    Formulates advisory experiment recommendation based on available evidence,
    candidate experiments in the project, and deterministic safety rules.
    Guaranteed executed: False.
    """
    reason_codes: List[str] = []
    evidence_used: List[str] = []
    missing_evidence: List[str] = []

    # Gather evidence
    evidence_used.append(f"scale_factor={req.scale_factor}")
    evidence_used.append(f"alignment_status={req.alignment_status}")
    evidence_used.append(f"ground_truth_available={req.ground_truth_available}")
    
    if req.current_architecture:
        evidence_used.append(f"current_architecture={req.current_architecture}")

    metrics = req.metrics or {}
    psnr_raw = metrics.get("psnr")
    psnr = float(psnr_raw) if psnr_raw is not None and isinstance(psnr_raw, (int, float)) else None
    ssim_raw = metrics.get("ssim")
    ssim = float(ssim_raw) if ssim_raw is not None and isinstance(ssim_raw, (int, float)) else None
    lpips_raw = metrics.get("lpips")
    lpips = float(lpips_raw) if lpips_raw is not None and isinstance(lpips_raw, (int, float)) else None

    if psnr is not None:
        evidence_used.append(f"psnr={psnr:.2f}dB")
    else:
        missing_evidence.append("psnr")

    if ssim is not None:
        evidence_used.append(f"ssim={ssim:.4f}")
    else:
        missing_evidence.append("ssim")

    if lpips is not None:
        evidence_used.append(f"lpips={lpips:.4f}")
    else:
        missing_evidence.append("lpips")

    # Safety Rule 1: Alignment Failure
    if req.alignment_status == "failed":
        return {
            "recommendation": "human_review",
            "reason_codes": ["ALIGNMENT_FAILED", "FUSION_UNRELIABLE"],
            "explanation": "RGB and thermal sensors are misaligned. Multimodal fusion models (Standard Fusion and Reliability Fusion) may hallucinate edges. Verify optical registration before proceeding.",
            "evidence_used": evidence_used,
            "missing_evidence": missing_evidence,
            "probabilities": None,
            "model_latency_ms": 0.0,
            "executed": False,
        }

    # Safety Rule 2: Missing Ground Truth for quantitative evaluation
    if not req.ground_truth_available or (psnr is None and ssim is None):
        return {
            "recommendation": "rerun_evaluation",
            "reason_codes": ["MISSING_GROUND_TRUTH", "QUANTITATIVE_METRICS_UNAVAILABLE"],
            "explanation": "Quantitative PSNR and SSIM benchmarks require a high-resolution ground truth reference. Recommend running an evaluation task with reference ground truth.",
            "evidence_used": evidence_used,
            "missing_evidence": missing_evidence,
            "probabilities": None,
            "model_latency_ms": 0.0,
            "executed": False,
        }

    # Deterministic Recommendation Logic
    avail = set(req.available_experiments or [])
    current_arch = req.current_architecture or ""

    det_rec = "no_recommendation"
    det_reasons: List[str] = []
    det_explanation = ""

    if current_arch == "thermal_cnn_baseline":
        if "rgb_fusion" in avail:
            det_rec = "rgb_fusion"
            det_reasons = ["BASELINE_COMPLETED", "TEST_MULTIMODAL_GAIN"]
            det_explanation = "Thermal baseline evaluated. Recommend reviewing the Standard RGB Fusion experiment to test whether high-frequency RGB guidance improves thermal reconstruction."
        elif "reliability_fusion" in avail:
            det_rec = "reliability_fusion"
            det_reasons = ["BASELINE_COMPLETED", "TEST_RELIABILITY_AWARE_GAIN"]
            det_explanation = "Thermal baseline evaluated. Recommend inspecting Reliability-Aware RGB Fusion to assess gated guidance benefits."

    elif current_arch == "rgb_fusion":
        if "reliability_fusion" in avail:
            det_rec = "reliability_fusion"
            det_reasons = ["STANDARD_FUSION_REVIEWED", "INSPECT_RELIABILITY_GATING"]
            det_explanation = "Standard RGB Fusion evaluated. Recommend inspecting Reliability-Aware Fusion to observe whether the learned gate suppresses hallucinated RGB edge artifacts."
        elif "thermal_baseline" in avail:
            det_rec = "thermal_baseline"
            det_reasons = ["COMPARE_AGAINST_ABLATION"]
            det_explanation = "Recommend comparing against the Thermal-Only Baseline to verify the net quantitative gain provided by RGB guidance."

    elif current_arch == "reliability_aware_fusion" or current_arch == "reliability_fusion":
        if "rgb_fusion" in avail and "thermal_baseline" in avail:
            det_rec = "thermal_baseline"
            det_reasons = ["FULL_ABLATION_COMPARISON"]
            det_explanation = "Reliability-Aware Fusion evaluated. Recommend reviewing the Thermal Baseline and Standard Fusion ablation runs side-by-side."
        else:
            det_rec = "human_review"
            det_reasons = ["RECORD_COMPLETE"]
            det_explanation = "Proposed model evaluation is complete. Ready for manual experimental analysis and table collation."
    else:
        det_rec = "human_review"
        det_reasons = ["UNKNOWN_ARCHITECTURE"]
        det_explanation = "Architecture context unspecified. Recommend manual review of existing experiments."

    # If Laya adapter is installed and loaded, we can refine choice with Laya Router
    if laya_adapter.load_router():
        try:
            allowed_options = [
                opt for opt in [
                    "thermal_baseline",
                    "rgb_fusion",
                    "reliability_fusion",
                    "rerun_evaluation",
                    "human_review",
                    "no_recommendation"
                ] if opt in avail or opt in ["rerun_evaluation", "human_review", "no_recommendation"]
            ]

            state_text = (
                f"ThermoSR Experiment Context:\n"
                f"- Current Arch: {current_arch}\n"
                f"- Metrics: PSNR={psnr}, SSIM={ssim}, LPIPS={lpips}\n"
                f"- Available Options: {allowed_options}\n"
                f"- Baseline Recommendation: {det_rec}\n"
            )
            questions = {
                "next_experiment": {
                    "type": "choice",
                    "options": allowed_options,
                    "criteria": "Which existing experiment or review step should be inspected next based on the evaluation evidence?"
                }
            }
            start_t = time.perf_counter()
            laya_res = laya_adapter._router.predict(state_text, questions)
            latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

            ans = laya_res.get("answers", {}).get("next_experiment", {})
            choice = ans.get("choice", det_rec)
            probs = ans.get("probabilities", None)

            # Ensure choice is among allowed options
            if choice in allowed_options:
                return {
                    "recommendation": choice,
                    "reason_codes": det_reasons + ["LAYA_ADVISED"],
                    "explanation": det_explanation,
                    "evidence_used": evidence_used,
                    "missing_evidence": missing_evidence,
                    "probabilities": probs,
                    "model_latency_ms": latency_ms,
                    "executed": False,
                }
        except Exception as e:
            logger.warning(f"Laya experiment advisor inference failed: {e}")

    # Fallback / deterministic recommendation
    return {
        "recommendation": det_rec,
        "reason_codes": det_reasons,
        "explanation": det_explanation,
        "evidence_used": evidence_used,
        "missing_evidence": missing_evidence,
        "probabilities": None,
        "model_latency_ms": None,
        "executed": False,
    }
