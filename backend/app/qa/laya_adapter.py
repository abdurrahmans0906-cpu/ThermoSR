import time
from typing import Dict, Any, Optional, Tuple, List
from app.core.logging import logger
from app.qa.schemas import QAMetadataInput
from app.qa.metadata_checks import perform_deterministic_checks

class LayaSystem1Adapter:
    """
    Isolated Adapter for Laya System 1 (non-autoregressive decision engine).
    Handles lazy loading, typed prediction, execution timing, error recovery,
    and deterministic safety overrides.
    """
    def __init__(self):
        self._router = None
        self._is_available: Optional[bool] = None
        self._load_error: Optional[str] = None
        self._mock_predict_fn = None  # Hook for isolated unit tests

    def is_installed(self) -> bool:
        try:
            import laya
            return True
        except ImportError:
            return False

    def load_router(self) -> bool:
        if self._router is not None:
            return True
        if self._load_error is not None:
            return False
        
        try:
            from laya import Router
            logger.info("Initializing Laya System 1 Router...")
            self._router = Router()
            self._is_available = True
            logger.info("Laya System 1 Router successfully initialized.")
            return True
        except ImportError:
            self._is_available = False
            self._load_error = "Laya package is not installed ('pip install laya')."
            return False
        except Exception as e:
            self._is_available = False
            self._load_error = f"Failed to initialize Laya Router: {str(e)}"
            logger.warning(f"Laya initialization failed: {e}")
            return False

    def set_mock_predict(self, mock_fn):
        """Allows injecting a mock prediction function for unit testing."""
        self._mock_predict_fn = mock_fn

    def clear_mock(self):
        self._mock_predict_fn = None

    def assess(self, meta: QAMetadataInput) -> Dict[str, Any]:
        """
        Executes deterministic checks and, if available, Laya System 1 evaluation.
        Enforces deterministic safety overrides.
        """
        # Step 1: Perform pure deterministic checks
        det_status, warnings, missing_evidence, checks = perform_deterministic_checks(meta)

        # Step 2: Check if mock is active (for unit testing)
        if self._mock_predict_fn is not None:
            start_t = time.perf_counter()
            mock_res = self._mock_predict_fn(meta)
            latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
            
            raw_assessment = mock_res.get("assessment", det_status)
            probabilities = mock_res.get("probabilities", None)
            
            # Apply deterministic override
            final_assessment = self._apply_safety_overrides(det_status, raw_assessment, checks, meta)
            
            return {
                "status": "success",
                "assessment": final_assessment,
                "warnings": warnings,
                "missing_evidence": missing_evidence,
                "deterministic_checks": checks,
                "probabilities": probabilities,
                "model_latency_ms": latency_ms,
                "model_version": "laya-system-1-mock",
                "reconstruction_modified": False,
            }

        # Step 3: Check Laya availability
        if not self.load_router():
            return {
                "status": "unavailable",
                "assessment": det_status,
                "warnings": warnings + [f"Laya System 1 model unavailable: {self._load_error}. Evaluated via deterministic rules."],
                "missing_evidence": missing_evidence,
                "deterministic_checks": checks,
                "probabilities": None,
                "model_latency_ms": None,
                "model_version": None,
                "reconstruction_modified": False,
            }

        # Step 4: Formulate structured prompt & typed question for Laya System 1
        state_text = (
            f"Thermal Super-Resolution Evaluation Record:\n"
            f"- Sample: {meta.sample_id}\n"
            f"- Scale Factor: x{meta.scale_factor}\n"
            f"- Dimensions: LR={meta.thermal_lr_shape}, HR={meta.thermal_hr_shape}\n"
            f"- Camera Alignment: {meta.alignment_status}\n"
            f"- Ground Truth Reference Available: {meta.ground_truth_available}\n"
            f"- Evaluation Metrics: {meta.metrics}\n"
            f"- Quality Warnings Detected: {warnings}\n"
            f"- Missing Evidence Detected: {missing_evidence}\n"
        )

        questions = {
            "review_readiness": {
                "type": "choice",
                "options": ["ready_for_review", "human_review", "insufficient_evidence"],
                "criteria": "Assess whether this thermal super-resolution evaluation record is sufficiently verified for academic review."
            }
        }

        # Step 5: Execute single-forward-pass prediction
        start_t = time.perf_counter()
        try:
            pred_result = self._router.predict(state_text, questions)
            latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

            answers = pred_result.get("answers", {}).get("review_readiness", {})
            raw_choice = answers.get("choice", det_status)
            probabilities = answers.get("probabilities", None)

            # Step 6: Enforce Deterministic Safety Overrides
            final_assessment = self._apply_safety_overrides(det_status, raw_choice, checks, meta)

            return {
                "status": "success",
                "assessment": final_assessment,
                "warnings": warnings,
                "missing_evidence": missing_evidence,
                "deterministic_checks": checks,
                "probabilities": probabilities,
                "model_latency_ms": latency_ms,
                "model_version": pred_result.get("routing", {}).get("model", "laya-system-1"),
                "reconstruction_modified": False,
            }

        except Exception as e:
            logger.warning(f"Laya inference error: {e}. Falling back to deterministic rules.")
            return {
                "status": "unavailable",
                "assessment": det_status,
                "warnings": warnings + [f"Laya inference failed ({str(e)}). Active status derived deterministically."],
                "missing_evidence": missing_evidence,
                "deterministic_checks": checks,
                "probabilities": None,
                "model_latency_ms": None,
                "model_version": None,
                "reconstruction_modified": False,
            }

    def _apply_safety_overrides(
        self,
        det_status: str,
        laya_status: str,
        checks: Dict[str, bool],
        meta: QAMetadataInput
    ) -> str:
        """
        Guarantees that mathematical & safety invariants strictly override model output:
        - Critical dimension mismatch cannot be marked 'ready_for_review'.
        - Explicitly failed alignment cannot be marked 'ready_for_review'.
        - Missing ground truth cannot be marked 'ready_for_review'.
        """
        valid_options = {"ready_for_review", "human_review", "insufficient_evidence"}
        if laya_status not in valid_options:
            return det_status

        # Safety Override 1: Explicit Alignment Failure
        if meta.alignment_status == "failed":
            return "human_review"

        # Safety Override 2: Dimension / Scale inconsistency
        if not checks.get("dimension_scale_consistent", True):
            return "human_review"

        # Safety Override 3: Missing Ground Truth Reference
        if not meta.ground_truth_available:
            return "insufficient_evidence"

        # Safety Override 4: If deterministic checks flagged human review, honor it
        if det_status == "human_review" and laya_status == "ready_for_review":
            return "human_review"

        return laya_status

# Global singleton
laya_adapter = LayaSystem1Adapter()
