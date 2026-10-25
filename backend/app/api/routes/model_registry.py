import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import ModelVersion
from app.schemas.model_registry import ModelVersionResponse, ModelRegisterRequest
from app.ml.models.factory import validate_checkpoint_file
from app.core.config import settings

router = APIRouter(prefix="/model-registry", tags=["Model Registry"])

@router.get("", response_model=List[ModelVersionResponse])
async def list_model_registry(db: AsyncSession = Depends(get_db)):
    stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    models = (await db.execute(stmt)).scalars().all()

    # If registry is empty, populate the default entries for the 3 architectures
    if not models:
        defaults = [
            ModelVersion(
                name="Thermal Baseline ResCNN (Untrained)",
                architecture="thermal_cnn_baseline",
                version="v1.0",
                checkpoint_path=None,
                supported_scales=[2, 4, 8],
                configuration={"trunk_channels": 64, "residual_blocks": 8},
                validation_metrics={},
                is_available=True,
                is_trained=False,
            ),
            ModelVersion(
                name="RGB-Guided Thermal Fusion (Untrained)",
                architecture="rgb_fusion",
                version="v1.0",
                checkpoint_path=None,
                supported_scales=[2, 4, 8],
                configuration={"fusion": "concatenation", "channels": 64},
                validation_metrics={},
                is_available=True,
                is_trained=False,
            ),
            ModelVersion(
                name="Reliability-Aware RGB Guidance (Untrained)",
                architecture="reliability_aware_fusion",
                version="v1.0",
                checkpoint_path=None,
                supported_scales=[2, 4, 8],
                configuration={"gate": "spatial_reliability_gate", "channels": 64},
                validation_metrics={},
                is_available=True,
                is_trained=False,
            ),
        ]
        db.add_all(defaults)
        await db.commit()
        models = (await db.execute(stmt)).scalars().all()

    return models

@router.post("/register", response_model=ModelVersionResponse)
async def register_model_checkpoint(
    req: ModelRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    # Verify checkpoint integrity
    is_valid, msg, meta = validate_checkpoint_file(req.checkpoint_path)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint validation failed: {msg}"
        )

    model_ver = ModelVersion(
        name=req.name,
        architecture=req.architecture,
        version=req.version,
        checkpoint_path=req.checkpoint_path,
        supported_scales=req.supported_scales,
        configuration=req.configuration or meta,
        validation_metrics=req.validation_metrics or {
            "best_psnr": meta.get("best_psnr"),
            "best_ssim": meta.get("best_ssim"),
        },
        is_available=True,
        is_trained=True,
    )
    db.add(model_ver)
    await db.commit()
    await db.refresh(model_ver)
    return model_ver

@router.get("/{model_id}", response_model=ModelVersionResponse)
async def get_model_version(model_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(ModelVersion).where(ModelVersion.id == model_id)
    mv = (await db.execute(stmt)).scalars().first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    return mv
