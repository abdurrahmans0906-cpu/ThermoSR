import asyncio
from app.db.session import async_engine, sync_engine, Base
from app.db.models import Dataset, DatasetPair, ModelVersion, TrainingRun, ReconstructionRun, Experiment
from app.core.logging import logger

def init_db_sync():
    """Create all tables synchronously (useful for migrations and quick starts)."""
    Base.metadata.create_all(bind=sync_engine)
    logger.info("Database tables initialized successfully via sync engine.")

async def init_db_async():
    """Create all tables asynchronously."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully via async engine.")

if __name__ == "__main__":
    init_db_sync()
