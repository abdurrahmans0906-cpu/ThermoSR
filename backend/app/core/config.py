import os
import torch
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "ThermoSR"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "artifacts"
    CHECKPOINTS_DIR: Path = BASE_DIR / "artifacts" / "checkpoints"
    DATASET_ROOT_DEFAULT: str = r"c:\Users\abdur\Downloads\challengedataset - Copy\challengedataset"
    
    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{str(BASE_DIR / 'thermosr.db').replace('\\', '/')}"
    DATABASE_SYNC_URL: str = f"sqlite:///{str(BASE_DIR / 'thermosr.db').replace('\\', '/')}"

    # File Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    # Compute Device
    DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure critical directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
(settings.STORAGE_DIR / "uploads").mkdir(parents=True, exist_ok=True)
(settings.STORAGE_DIR / "reconstructions").mkdir(parents=True, exist_ok=True)
(settings.STORAGE_DIR / "exports").mkdir(parents=True, exist_ok=True)
