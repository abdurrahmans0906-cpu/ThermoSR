from fastapi import APIRouter, HTTPException, status
from app.qa.schemas import (
    QAMetadataInput,
    QAAssessmentResponse,
    ExperimentRecommendationRequest,
    ExperimentRecommendationResponse,
    QAHealthResponse,
)
from app.qa.laya_adapter import laya_adapter
from app.qa.experiment_advisor import recommend_next_experiment
from app.core.logging import logger

router = APIRouter(prefix="/qa", tags=["Intelligent QA & Experiment Advisor"])

@router.get("/health", response_model=QAHealthResponse)
async def get_qa_health():
    """
    Checks the status and availability of the Intelligent QA add-on and Laya System 1.
    """
    installed = laya_adapter.is_installed()
    loaded = laya_adapter.load_router() if installed else False
    return QAHealthResponse(
        status="ok",
        laya_installed=installed,
        laya_model_loaded=loaded,
        fallback_mode=not loaded,
    )

@router.post("/laya-assess", response_model=QAAssessmentResponse)
async def assess_metadata(metadata: QAMetadataInput):
    """
    Evaluates reconstruction metadata, camera alignment, and evaluation evidence
    using deterministic checks and Laya System 1 decision engine.
    Guaranteed read-only (reconstruction_modified: false).
    """
    try:
        result = laya_adapter.assess(metadata)
        return QAAssessmentResponse(**result)
    except Exception as e:
        logger.error(f"Unexpected error in QA assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing QA assessment."
        )

@router.post("/experiment-recommendation", response_model=ExperimentRecommendationResponse)
async def get_experiment_recommendation(request: ExperimentRecommendationRequest):
    """
    Provides advisory guidance on which existing experiment to inspect next.
    Guaranteed advisory only (executed: false).
    """
    try:
        result = recommend_next_experiment(request)
        return ExperimentRecommendationResponse(**result)
    except Exception as e:
        logger.error(f"Unexpected error in experiment recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing experiment recommendation."
        )
