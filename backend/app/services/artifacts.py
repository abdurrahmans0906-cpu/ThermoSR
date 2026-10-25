import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from app.core.config import settings

def save_image_artifact(
    image: np.ndarray,
    subfolder: str,
    filename: str,
    is_rgb: bool = False
) -> str:
    """
    Saves an image artifact to local filesystem under settings.STORAGE_DIR.
    Returns relative path from STORAGE_DIR.
    """
    target_dir = settings.STORAGE_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    full_path = target_dir / filename

    if image.dtype in [np.float32, np.float64]:
        img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_u8 = image

    if is_rgb and img_u8.ndim == 3 and img_u8.shape[2] == 3:
        # Convert RGB to BGR for cv2.imwrite
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(full_path), img_bgr)
    else:
        cv2.imwrite(str(full_path), img_u8)

    return f"{subfolder}/{filename}"

def save_json_artifact(data: Dict[str, Any], subfolder: str, filename: str) -> str:
    target_dir = settings.STORAGE_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    full_path = target_dir / filename

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return f"{subfolder}/{filename}"

def get_artifact_path(relative_path: str) -> Optional[Path]:
    """
    Validates against directory traversal and returns absolute Path.
    """
    safe_rel = os.path.normpath(relative_path).lstrip("\\/")
    abs_path = (settings.STORAGE_DIR / safe_rel).resolve()
    
    # Check that abs_path is within STORAGE_DIR or DATASET_ROOT
    if not str(abs_path).startswith(str(settings.STORAGE_DIR.resolve())):
        return None
    if not abs_path.exists():
        return None
    return abs_path
