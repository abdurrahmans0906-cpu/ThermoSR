import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db.session import get_db
from app.db.models import Dataset, DatasetPair
from app.schemas.dataset import (
    DatasetScanRequest,
    DatasetResponse,
    DatasetPairResponse,
    DatasetValidationReport
)
from app.services.datasets import scan_dataset
from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/scan", response_model=DatasetResponse)
async def scan_dataset_endpoint(
    req: DatasetScanRequest,
    db: AsyncSession = Depends(get_db)
):
    target_root = req.root_path or settings.DATASET_ROOT_DEFAULT
    if not os.path.exists(target_root):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset path not found: {target_root}. Please ensure the dataset exists or provide a valid path."
        )

    try:
        scan_results = scan_dataset(target_root)
    except Exception as e:
        logger.error(f"Failed to scan dataset: {e}")
        raise HTTPException(status_code=400, detail=f"Dataset scanning failed: {str(e)}")

    name = req.name or Path(target_root).name
    
    # Check if dataset already recorded
    stmt = select(Dataset).where(Dataset.root_dir == target_root)
    res = await db.execute(stmt)
    existing = res.scalars().first()

    if existing and not req.force_rescan:
        return existing

    if existing:
        dataset_obj = existing
        dataset_obj.name = name
        dataset_obj.dataset_type = scan_results["dataset_type"]
        dataset_obj.split_config = scan_results["split_counts"]
        dataset_obj.total_pairs = scan_results["total_pairs"]
        dataset_obj.valid_pairs = scan_results["valid_pairs"]
        dataset_obj.unmatched_count = scan_results["unmatched_count"]
        dataset_obj.status = "validated" if scan_results["valid_pairs"] > 0 else "empty"
    else:
        dataset_obj = Dataset(
            name=name,
            root_dir=target_root,
            dataset_type=scan_results["dataset_type"],
            split_config=scan_results["split_counts"],
            total_pairs=scan_results["total_pairs"],
            valid_pairs=scan_results["valid_pairs"],
            unmatched_count=scan_results["unmatched_count"],
            status="validated" if scan_results["valid_pairs"] > 0 else "empty"
        )
        db.add(dataset_obj)

    await db.flush()

    # Save discovered pairs
    # Clear existing pairs if rescan
    if existing:
        del_stmt = select(DatasetPair).where(DatasetPair.dataset_id == dataset_obj.id)
        existing_pairs = (await db.execute(del_stmt)).scalars().all()
        for p in existing_pairs:
            await db.delete(p)

    pairs_to_add = []
    for p in scan_results["pairs"][:1000]:  # Limit DB rows to 1000 for efficiency
        pair_row = DatasetPair(
            dataset_id=dataset_obj.id,
            split=p["split"],
            thermal_path=p["thermal_path"],
            rgb_path=p.get("rgb_path"),
            reference_path=p.get("reference_path"),
            dimensions=p["dimensions"],
            validation_status=p["validation_status"],
            alignment_status=p["alignment_status"],
            error_message=None
        )
        pairs_to_add.append(pair_row)

    db.add_all(pairs_to_add)
    await db.commit()
    await db.refresh(dataset_obj)
    return dataset_obj

@router.get("", response_model=List[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    res = await db.execute(stmt)
    return res.scalars().all()

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    res = await db.execute(stmt)
    dataset = res.scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.get("/{dataset_id}/samples", response_model=List[DatasetPairResponse])
async def get_dataset_samples(
    dataset_id: str,
    split: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(DatasetPair).where(DatasetPair.dataset_id == dataset_id)
    if split:
        stmt = stmt.where(DatasetPair.split == split)
    stmt = stmt.limit(limit)

    res = await db.execute(stmt)
    return res.scalars().all()

@router.post("/{dataset_id}/validate", response_model=DatasetValidationReport)
async def validate_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    dataset = (await db.execute(stmt)).scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    pair_stmt = select(DatasetPair).where(DatasetPair.dataset_id == dataset_id)
    pairs = (await db.execute(pair_stmt)).scalars().all()

    missing_rgb = sum(1 for p in pairs if not p.rgb_path)
    aspect_warnings = [
        {"id": p.id, "split": p.split, "dimensions": p.dimensions}
        for p in pairs if p.alignment_status == "aspect_mismatch"
    ][:10]

    return DatasetValidationReport(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        root_dir=dataset.root_dir,
        total_scanned=len(pairs),
        valid_pairs=sum(1 for p in pairs if p.validation_status == "valid"),
        missing_rgb=missing_rgb,
        missing_thermal=0,
        dimension_issues=[],
        aspect_ratio_warnings=aspect_warnings,
        split_breakdown=dataset.split_config or {},
        is_ready_for_training=dataset.valid_pairs > 0,
        summary=f"Dataset '{dataset.name}' scanned with {dataset.valid_pairs} valid pairs across {len(dataset.split_config or {})} splits."
    )
