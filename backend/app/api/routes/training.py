import uuid
import threading
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db, SyncSessionLocal
from app.db.models import TrainingRun, Dataset, DatasetPair, ModelVersion
from app.schemas.training import (
    TrainingRunCreateRequest,
    TrainingRunResponse,
)
from app.ml.training.trainer import TrainingSession
from app.workers.queue import register_training_session, unregister_training_session, cancel_training_session
from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/training-runs", tags=["Training"])

def run_training_background(run_id: str, session: TrainingSession):
    register_training_session(run_id, session)
    try:
        results = session.run()
        # Update database synchronously at the end
        with SyncSessionLocal() as db:
            run_obj = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run_obj:
                run_obj.status = results.get("status", "completed")
                run_obj.end_time = datetime.now(timezone.utc)
                run_obj.elapsed_seconds = results.get("elapsed_seconds", 0.0)
                run_obj.best_checkpoint_path = results.get("best_checkpoint")
                run_obj.metrics_history = results.get("metrics_history", [])
                
                # If completed with a checkpoint, auto-register in ModelRegistry
                if results.get("best_checkpoint") and run_obj.status == "completed":
                    mv = ModelVersion(
                        name=f"{session.architecture}_x{session.scale_factor}_{run_id[:8]}",
                        architecture=session.architecture,
                        version="v1.0",
                        checkpoint_path=results.get("best_checkpoint"),
                        training_run_id=run_id,
                        supported_scales=[session.scale_factor],
                        validation_metrics={"best_psnr": results.get("best_psnr")},
                        is_available=True,
                        is_trained=True,
                    )
                    db.add(mv)

                db.commit()
    except Exception as e:
        logger.error(f"Background training failed for {run_id}: {e}")
        with SyncSessionLocal() as db:
            run_obj = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run_obj:
                run_obj.status = "failed"
                run_obj.error_details = str(e)
                run_obj.end_time = datetime.now(timezone.utc)
                db.commit()
    finally:
        unregister_training_session(run_id)

@router.post("", response_model=TrainingRunResponse)
async def create_training_run(
    req: TrainingRunCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    # Validate dataset
    dataset_id = req.dataset_id
    if dataset_id:
        ds_stmt = select(Dataset).where(Dataset.id == dataset_id)
        dataset = (await db.execute(ds_stmt)).scalars().first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Specified dataset not found")
        dataset_root = dataset.root_dir
    else:
        # Fallback to default dataset root
        dataset_root = settings.DATASET_ROOT_DEFAULT
        ds_stmt = select(Dataset).where(Dataset.root_dir == dataset_root)
        dataset = (await db.execute(ds_stmt)).scalars().first()

    # Fetch dataset pairs
    if dataset:
        p_stmt = select(DatasetPair).where(DatasetPair.dataset_id == dataset.id)
        all_pairs = (await db.execute(p_stmt)).scalars().all()
        train_pairs = [{"thermal_path": p.thermal_path, "rgb_path": p.rgb_path} for p in all_pairs if p.split == "train"]
        val_pairs = [{"thermal_path": p.thermal_path, "rgb_path": p.rgb_path} for p in all_pairs if p.split in ["validation", "test"]]
    else:
        train_pairs = []
        val_pairs = []

    if not train_pairs:
        raise HTTPException(
            status_code=400,
            detail="No training pairs discovered. Please scan the dataset first on the Dataset page before initiating training."
        )

    run_id = str(uuid.uuid4())
    hp_dict = req.hyperparameters.model_dump() if req.hyperparameters else {
        "batch_size": 8,
        "patch_size": 64,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "total_epochs": req.total_epochs,
        "loss_reconstruction_weight": 1.0,
        "loss_gradient_weight": 0.1,
        "loss_thermal_consistency_weight": 0.05,
        "use_gradient_loss": True,
        "use_thermal_consistency_loss": True,
    }
    hp_dict["total_epochs"] = req.total_epochs

    run_obj = TrainingRun(
        id=run_id,
        model_architecture=req.model_architecture,
        dataset_id=dataset.id if dataset else None,
        scale_factor=req.scale_factor,
        hyperparameters=hp_dict,
        random_seed=req.random_seed,
        device_info=req.device or settings.DEFAULT_DEVICE,
        status="running",
        start_time=datetime.now(timezone.utc),
        total_epochs=req.total_epochs,
        current_epoch=0,
        metrics_history=[]
    )
    db.add(run_obj)
    await db.commit()
    await db.refresh(run_obj)

    # Callback to update DB after each epoch
    def on_epoch(metric_data: Dict[str, Any]):
        with SyncSessionLocal() as sdb:
            r = sdb.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if r:
                r.current_epoch = metric_data["epoch"]
                r.latest_metrics = metric_data
                history = list(r.metrics_history or [])
                history.append(metric_data)
                r.metrics_history = history
                sdb.commit()

    session = TrainingSession(
        run_id=run_id,
        architecture=req.model_architecture,
        scale_factor=req.scale_factor,
        hyperparameters=hp_dict,
        train_pairs=train_pairs,
        val_pairs=val_pairs if val_pairs else train_pairs[:10],
        dataset_root=dataset_root,
        device=req.device or "auto",
        seed=req.random_seed,
        on_epoch_complete=on_epoch
    )

    thread = threading.Thread(target=run_training_background, args=(run_id, session), daemon=True)
    thread.start()

    return run_obj

@router.get("", response_model=List[TrainingRunResponse])
async def list_training_runs(db: AsyncSession = Depends(get_db)):
    stmt = select(TrainingRun).order_by(TrainingRun.created_at.desc())
    res = await db.execute(stmt)
    return res.scalars().all()

@router.get("/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(run_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(TrainingRun).where(TrainingRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if not run_obj:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run_obj

@router.get("/{run_id}/metrics")
async def get_training_metrics(run_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(TrainingRun).where(TrainingRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if not run_obj:
        raise HTTPException(status_code=404, detail="Training run not found")
    return {
        "run_id": run_id,
        "status": run_obj.status,
        "current_epoch": run_obj.current_epoch,
        "total_epochs": run_obj.total_epochs,
        "latest": run_obj.latest_metrics,
        "history": run_obj.metrics_history,
    }

@router.post("/{run_id}/cancel")
async def cancel_training_run_endpoint(run_id: str, db: AsyncSession = Depends(get_db)):
    success = cancel_training_session(run_id)
    stmt = select(TrainingRun).where(TrainingRun.id == run_id)
    run_obj = (await db.execute(stmt)).scalars().first()
    if run_obj and run_obj.status == "running":
        run_obj.status = "cancelled"
        await db.commit()
    return {"run_id": run_id, "cancelled": success}
