import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Experiment, TrainingRun, ReconstructionRun
from app.schemas.experiment import ExperimentCreateRequest, ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["Experiments"])

@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    stmt = select(Experiment).order_by(Experiment.created_at.desc())
    res = await db.execute(stmt)
    return res.scalars().all()

@router.post("", response_model=ExperimentResponse)
async def create_experiment(req: ExperimentCreateRequest, db: AsyncSession = Depends(get_db)):
    # Gather training runs
    stmt = select(TrainingRun).where(TrainingRun.id.in_(req.training_run_ids))
    runs = (await db.execute(stmt)).scalars().all()

    comparison = {
        "runs": [
            {
                "id": r.id,
                "architecture": r.model_architecture,
                "scale_factor": r.scale_factor,
                "status": r.status,
                "best_psnr": r.latest_metrics.get("psnr") if r.latest_metrics else None,
                "best_ssim": r.latest_metrics.get("ssim") if r.latest_metrics else None,
                "val_loss": r.latest_metrics.get("val_loss") if r.latest_metrics else None,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in runs
        ]
    }

    exp = Experiment(
        name=req.name,
        description=req.description,
        experiment_type=req.experiment_type,
        related_training_run_ids=req.training_run_ids,
        comparison_results=comparison
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)
    return exp

@router.get("/ablation-summary")
async def get_ablation_summary(db: AsyncSession = Depends(get_db)):
    """
    Auto-summarizes completed runs across the 3 architectures for scientific ablation analysis.
    """
    stmt = select(TrainingRun).where(TrainingRun.status == "completed").order_by(TrainingRun.created_at.desc())
    runs = (await db.execute(stmt)).scalars().all()

    # Group by architecture
    grouped = {
        "thermal_cnn_baseline": None,
        "rgb_fusion": None,
        "reliability_aware_fusion": None,
        "reliability_aware_fusion_no_consistency": None,
    }

    for r in runs:
        arch = r.model_architecture
        hp = r.hyperparameters or {}
        if arch == "reliability_aware_fusion" and not hp.get("use_thermal_consistency_loss", True):
            if not grouped["reliability_aware_fusion_no_consistency"]:
                grouped["reliability_aware_fusion_no_consistency"] = r
        elif not grouped.get(arch):
            grouped[arch] = r

    summary = []
    for key, run_obj in grouped.items():
        if run_obj:
            summary.append({
                "configuration": key,
                "run_id": run_obj.id,
                "architecture": run_obj.model_architecture,
                "scale_factor": run_obj.scale_factor,
                "metrics": run_obj.latest_metrics or {},
                "created_at": run_obj.created_at
            })

    return {"ablation_runs": summary, "total_completed_runs": len(runs)}

@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    exp = (await db.execute(stmt)).scalars().first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp
