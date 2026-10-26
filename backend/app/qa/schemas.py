from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any

class QAMetadataInput(BaseModel):
    sample_id: str = Field(..., description="Unique identifier for the sample or reconstruction run")
    experiment_id: Optional[str] = Field(None, description="Optional experiment identifier")
    experiment_type: Optional[str] = Field(None, description="Type of experiment (e.g., ablation, benchmark)")
    scale_factor: int = Field(8, description="Thermal super-resolution scale factor")
    thermal_hr_shape: Optional[List[int]] = Field(None, description="High-resolution thermal shape [width, height]")
    thermal_lr_shape: List[int] = Field(..., description="Low-resolution thermal shape [width, height]")
    rgb_shape: Optional[List[int]] = Field(None, description="RGB guidance shape [width, height]")
    alignment_status: str = Field("unverified", description="RGB/Thermal alignment: verified, unverified, failed, not_applicable")
    ground_truth_available: bool = Field(False, description="Whether ground-truth thermal reference is available")
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Calculated evaluation metrics")
    evaluation_timestamp: Optional[str] = Field(None, description="ISO timestamp of evaluation")
    deterministic_warnings: Optional[List[str]] = Field(default_factory=list, description="Pre-computed warnings")

    @field_validator("scale_factor")
    @classmethod
    def validate_scale_factor(cls, v: int) -> int:
        supported = [2, 4, 8, 16]
        if v not in supported:
            raise ValueError(f"Unsupported scale factor {v}. Must be one of {supported}.")
        return v

    @field_validator("thermal_lr_shape")
    @classmethod
    def validate_lr_shape(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("thermal_lr_shape must have exactly 2 dimensions [width, height].")
        if any(dim <= 0 for dim in v):
            raise ValueError("All dimensions in thermal_lr_shape must be positive integers.")
        return v

    @field_validator("thermal_hr_shape")
    @classmethod
    def validate_hr_shape(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None:
            if len(v) != 2:
                raise ValueError("thermal_hr_shape must have exactly 2 dimensions [width, height].")
            if any(dim <= 0 for dim in v):
                raise ValueError("All dimensions in thermal_hr_shape must be positive integers.")
        return v

    @field_validator("rgb_shape")
    @classmethod
    def validate_rgb_shape(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None:
            if len(v) not in (2, 3):
                raise ValueError("rgb_shape must have 2 or 3 dimensions [width, height, (channels)].")
            if any(dim <= 0 for dim in v):
                raise ValueError("All dimensions in rgb_shape must be positive integers.")
        return v

    @field_validator("alignment_status")
    @classmethod
    def validate_alignment(cls, v: str) -> str:
        valid_statuses = {"verified", "unverified", "failed", "not_applicable"}
        v_clean = v.lower().strip()
        if v_clean not in valid_statuses:
            raise ValueError(f"alignment_status '{v}' invalid. Must be one of {valid_statuses}.")
        return v_clean

    @field_validator("metrics")
    @classmethod
    def validate_metrics_ranges(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        ssim = v.get("ssim")
        if ssim is not None and isinstance(ssim, (int, float)) and not (-1.0 <= ssim <= 1.0):
            raise ValueError(f"SSIM must be between -1.0 and 1.0, got {ssim}")
        
        psnr = v.get("psnr")
        if psnr is not None and isinstance(psnr, (int, float)) and (psnr < 0.0 or psnr > 100.0):
            raise ValueError(f"PSNR must be between 0.0 and 100.0 dB, got {psnr}")

        lpips = v.get("lpips")
        if lpips is not None and isinstance(lpips, (int, float)) and lpips < 0.0:
            raise ValueError(f"LPIPS must be non-negative, got {lpips}")

        return v


class QAAssessmentResponse(BaseModel):
    addon: str = "ThermoSR Intelligent QA & Experiment Advisor (Laya System 1)"
    status: str = Field(..., description="success | insufficient_evidence | unavailable | invalid_request")
    assessment: str = Field(..., description="ready_for_review | human_review | insufficient_evidence | unavailable")
    warnings: List[str] = Field(default_factory=list, description="Quality & safety warnings")
    missing_evidence: List[str] = Field(default_factory=list, description="Missing evaluation fields")
    deterministic_checks: Dict[str, bool] = Field(default_factory=dict, description="Individual deterministic check outcomes")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Laya System 1 probability distribution if available")
    model_latency_ms: Optional[float] = Field(None, description="Laya model inference latency in milliseconds")
    model_version: Optional[str] = Field(None, description="Underlying Laya model identifier")
    reconstruction_modified: bool = Field(False, description="Guaranteed false: reconstruction artifacts are never altered")
    disclaimer: str = Field(
        "Advisory assessment only. Does not alter reconstruction outputs, retrain models, or guarantee physical thermal fidelity.",
        description="Scientific limitations disclaimer"
    )


class ExperimentRecommendationRequest(BaseModel):
    current_experiment_id: Optional[str] = None
    current_architecture: Optional[str] = Field("reliability_aware_fusion", description="thermal_cnn_baseline, rgb_fusion, or reliability_aware_fusion")
    available_experiments: List[str] = Field(
        default_factory=lambda: ["thermal_baseline", "rgb_fusion", "reliability_fusion"],
        description="Experiments available for inspection"
    )
    scale_factor: int = Field(8, description="Thermal SR scale factor")
    alignment_status: str = Field("unverified", description="verified, unverified, failed, not_applicable")
    ground_truth_available: bool = Field(False, description="Ground truth availability")
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Current run metrics")
    deterministic_warnings: Optional[List[str]] = Field(default_factory=list)
    comparison_results: Optional[Dict[str, Any]] = None

    @field_validator("scale_factor")
    @classmethod
    def validate_scale_factor(cls, v: int) -> int:
        if v not in [2, 4, 8, 16]:
            raise ValueError(f"Unsupported scale factor {v}.")
        return v


class ExperimentRecommendationResponse(BaseModel):
    recommendation: str = Field(
        ...,
        description="thermal_baseline | rgb_fusion | reliability_fusion | rerun_evaluation | human_review | no_recommendation"
    )
    reason_codes: List[str] = Field(default_factory=list, description="Machine-readable decision reason codes")
    explanation: str = Field(..., description="Human-readable justification based strictly on submitted metadata")
    evidence_used: List[str] = Field(default_factory=list, description="Metadata and metric fields used for decision")
    missing_evidence: List[str] = Field(default_factory=list, description="Data points absent from evaluation record")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Laya choice probabilities if available")
    model_latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    executed: bool = Field(False, description="Guaranteed false: advisory only, does not launch jobs")
    disclaimer: str = Field(
        "Recommendation only. Does not automatically launch training or reconstruction jobs.",
        description="Advisory disclaimer"
    )


class QAHealthResponse(BaseModel):
    status: str = "ok"
    laya_installed: bool = False
    laya_model_loaded: bool = False
    fallback_mode: bool = True
    supported_statuses: List[str] = ["ready_for_review", "human_review", "insufficient_evidence", "unavailable"]
    supported_recommendations: List[str] = [
        "thermal_baseline",
        "rgb_fusion",
        "reliability_fusion",
        "rerun_evaluation",
        "human_review",
        "no_recommendation"
    ]
