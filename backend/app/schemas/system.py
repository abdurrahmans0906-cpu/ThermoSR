from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class HealthResponse(BaseSchema):
    status: str
    version: str
    project: str
    database: str
    device: str
    torch_version: str
    cuda_available: bool

class DeviceInfo(BaseSchema):
    device: str
    device_name: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    vram_total_mb: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_free_mb: Optional[float] = None
    torch_version: str
    cpu_cores: int
    recommended_batch_size: int
    notes: List[str]
