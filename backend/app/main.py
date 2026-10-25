from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.db.init_db import init_db_async
from app.services.artifacts import get_artifact_path

from app.api.routes import (
    health,
    system,
    datasets,
    reconstruction,
    training,
    experiments,
    model_registry,
    qa,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Initializing ThermoSR Application and Database...")
    await init_db_async()
    logger.info("ThermoSR Backend is ready.")
    yield
    logger.info("ThermoSR Backend shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Research-Grade Deep Learning Thermal Image Super-Resolution with RGB Guidance",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(system.router, prefix=settings.API_V1_STR)
app.include_router(datasets.router, prefix=settings.API_V1_STR)
app.include_router(reconstruction.router, prefix=settings.API_V1_STR)
app.include_router(training.router, prefix=settings.API_V1_STR)
app.include_router(experiments.router, prefix=settings.API_V1_STR)
app.include_router(model_registry.router, prefix=settings.API_V1_STR)
app.include_router(qa.router, prefix=settings.API_V1_STR)

# Secure Artifact Download Endpoint
@app.get("/api/artifacts/{file_path:path}")
async def serve_artifact(file_path: str):
    abs_path = get_artifact_path(file_path)
    if not abs_path or not abs_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found or access denied")
    return FileResponse(abs_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
