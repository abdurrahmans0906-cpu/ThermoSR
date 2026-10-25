from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ReconstructionCreateRequest(BaseModel):
    task_mode: str = "rgb_guided"  # rgb_guided, evaluation, downsample_eval
    model_version_id: Optional[str] = None
    model_architecture: str = "reliability_aware_fusion"
    scale_factor: int = Field(default=8, ge=2, le=16)
    device: str = "auto"
    tile_size: int = 0  # 0 = full image or patch size
    tile_overlap: int = 16
    colormap: str = "inferno"  # inferno, viridis, magma, jet
    calculate_lpips: bool = True

class ReconstructionMetrics(BaseModel):
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    thermal_consistency_l1: Optional[float] = None
    gradient_similarity: Optional[float] = None
    inference_time_ms: float
    output_dimensions: List[int]
    input_dimensions: List[int]
    has_reference: bool

class ReconstructionRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    task_mode: str
    model_version_id: Optional[str] = None
    model_architecture: Optional[str] = None
    scale_factor: int
    input_artifacts: Dict[str, Any]
    output_artifacts: Dict[str, Any]
    configuration: Dict[str, Any]
    status: str
    progress_pct: int
    metrics: Dict[str, Any]
    inference_time_ms: float
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
