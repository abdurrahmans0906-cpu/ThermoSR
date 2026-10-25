from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    architecture: str
    version: str
    checkpoint_path: Optional[str] = None
    configuration: Dict[str, Any]
    training_run_id: Optional[str] = None
    supported_scales: List[int]
    validation_metrics: Dict[str, Any]
    is_available: bool
    is_trained: bool
    created_at: datetime
    updated_at: datetime

class ModelRegisterRequest(BaseModel):
    name: str
    architecture: str
    version: str = "v1.0"
    checkpoint_path: str
    supported_scales: List[int] = [2, 4, 8]
    configuration: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None
