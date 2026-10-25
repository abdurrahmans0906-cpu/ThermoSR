from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import torch
from app.core.config import settings
from app.db.session import get_db
from app.schemas.system import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def get_health(db: AsyncSession = Depends(get_db)):
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if db_status == "ok" else "degraded",
        version=settings.VERSION,
        project=settings.PROJECT_NAME,
        database=db_status,
        device=settings.DEFAULT_DEVICE,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available()
    )
