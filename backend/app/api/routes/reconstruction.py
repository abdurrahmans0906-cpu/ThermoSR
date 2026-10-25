import os
import shutil
import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from PIL import Image

from app.db.session import get_db, SyncSessionLocal
from app.db.models import ReconstructionRun, ModelVersion
from app.schemas.reconstruction import ReconstructionRunResponse
from app.services.reconstruction import run_reconstruction_pipeline
from app.services.artifacts import get_artifact_path
from app.workers.queue import register_reconstruction_job, unregister_reconstruction_job, cancel_reconstruction_job
from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/reconstructions", tags=["Reconstruction"])

def validate_image_upload(file: UploadFile) -> str:
    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {settings.ALLOWED_EXTENSIONS}"
        )
    return ext

@router.post("", response_model=ReconstructionRunResponse)
async def create_reconstruction(
    background_tasks: BackgroundTasks,
    task_mode: str = Form("rgb_guided"),
    model_architecture: str = Form("reliability_aware_fusion"),
    scale_factor: int = Form(8),
    device: str = Form("auto"),
    colormap: str = Form("inferno"),
    model_version_id: Optional[str] = Form(None),
    thermal_file: UploadFile = File(...),
    rgb_file: Optional[UploadFile] = File(None),
    reference_file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    run_id = str(uuid.uuid4())
    upload_dir = settings.STORAGE_DIR / "uploads" / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Validate & Save Thermal File
    t_ext = validate_image_upload(thermal_file)
    thermal_path = upload_dir / f"thermal_input{t_ext}"
    with open(thermal_path, "wb") as buffer:
        shutil.copyfileobj(thermal_file.file, buffer)

    # Verify thermal image can be opened
    try:
        with Image.open(thermal_path) as img:
            img.verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Thermal file is corrupt or invalid image: {str(e)}")

    # RGB File Handling
    rgb_path = None
    if rgb_file is not None and rgb_file.filename:
        r_ext = validate_image_upload(rgb_file)
        rgb_path = upload_dir / f"rgb_input{r_ext}"
        with open(rgb_path, "wb") as buffer:
            shutil.copyfileobj(rgb_file.file, buffer)
        try:
            with Image.open(rgb_path) as img:
                img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"RGB file is corrupt or invalid image: {str(e)}")
    elif model_architecture in ["rgb_fusion", "reliability_aware_fusion"]:
        raise HTTPException(
            status_code=400,
            detail=f"Architecture '{model_architecture}' requires an RGB guidance image."
        )

    # Reference File Handling
    ref_path = None
    if reference_file is not None and reference_file.filename:
        ref_ext = validate_image_upload(reference_file)
        ref_path = upload_dir / f"reference_input{ref_ext}"
        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(reference_file.file, buffer)

    # Lookup Checkpoint if model_version_id provided
    checkpoint_path = None
    if model_version_id:
        mv_stmt = select(ModelVersion).where(ModelVersion.id == model_version_id)
        mv = (await db.execute(mv_stmt)).scalars().first()
        if mv and mv.checkpoint_path:
            checkpoint_path = mv.checkpoint_path

    # Record run in database
    run_obj = ReconstructionRun(
        id=run_id,
        task_mode=task_mode,
        model_version_id=model_version_id,
        model_architecture=model_architecture,
        scale_factor=scale_factor,
        input_artifacts={
            "thermal": str(thermal_path.relative_to(settings.STORAGE_DIR)),
            "rgb": str(rgb_path.relative_to(settings.STORAGE_DIR)) if rgb_path else None,
            "reference": str(ref_path.relative_to(settings.STORAGE_DIR)) if ref_path else None
        },
        output_artifacts={},
        configuration={
            "device": device,
            "colormap": colormap,
            "scale_factor": scale_factor,
        },
        status="running_inference",
        progress_pct=20,
        metrics={},
        inference_time_ms=0.0
    )
    db.add(run_obj)
    await db.commit()
    await db.refresh(run_obj)

    # Execute Reconstruction Pipeline
    try:
        result = run_reconstruction_pipeline(
            run_id=run_id,
            task_mode=task_mode,
            model_architecture=model_architecture,
            scale_factor=scale_factor,
            thermal_input_path=str(thermal_path),
            rgb_input_path=str(rgb_path) if rgb_path else None,
            reference_path=str(ref_path) if ref_path else None,
            checkpoint_path=checkpoint_path,
            device=device,
            colormap=colormap
        )

        run_obj.status = "completed"
        run_obj.progress_pct = 100
        run_obj.output_artifacts = result["artifacts"]
        run_obj.metrics = result["metrics"]
        run_obj.inference_time_ms = result["inference_time_ms"]
        run_obj.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(run_obj)

    except Exception as e:
        logger.error(f"Reconstruction failed for run {run_id}: {e}")
        run_obj.status = "failed"
        run_obj.error_message = str(e)
        await db.commit()
        await db.refresh(run_obj)

    return run_obj

@router.get("", response_model=List[ReconstructionRunResponse])
async def list_reconstructions(limit: int = 20, db: AsyncSession = Depends(get_db)):
    stmt = select(ReconstructionRun).order_by(ReconstructionRun.created_at.desc()).limit(limit)
    res = await db.execute(stmt)
    return res.scalars().all()

@router.get("/{run_id}", response_model=ReconstructionRunResponse)
async def get_reconstruction(run_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(ReconstructionRun).where(ReconstructionRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if not run_obj:
        raise HTTPException(status_code=404, detail="Reconstruction run not found")
    return run_obj

@router.get("/{run_id}/metrics")
async def get_reconstruction_metrics(run_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(ReconstructionRun).where(ReconstructionRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if not run_obj:
        raise HTTPException(status_code=404, detail="Reconstruction run not found")
    return run_obj.metrics

@router.get("/{run_id}/artifacts")
async def get_reconstruction_artifacts(run_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(ReconstructionRun).where(ReconstructionRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if not run_obj:
        raise HTTPException(status_code=404, detail="Reconstruction run not found")
    return {
        "inputs": run_obj.input_artifacts,
        "outputs": run_obj.output_artifacts
    }

@router.post("/{run_id}/cancel")
async def cancel_reconstruction(run_id: str, db: AsyncSession = Depends(get_db)):
    cancelled = cancel_reconstruction_job(run_id)
    stmt = select(ReconstructionRun).where(ReconstructionRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if run_obj and run_obj.status not in ["completed", "failed"]:
        run_obj.status = "cancelled"
        await db.commit()
    return {"run_id": run_id, "cancelled": cancelled}
