from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class HyperparametersConfig(BaseModel):
    batch_size: int = Field(default=8, ge=1, le=128)
    patch_size: int = Field(default=64, ge=32, le=256)
    learning_rate: float = Field(default=1e-4, gt=0)
    optimizer: str = "adamw"  # adam, adamw
    loss_reconstruction_weight: float = 1.0
    loss_gradient_weight: float = 0.1
    loss_thermal_consistency_weight: float = 0.05
    use_gradient_loss: bool = True
    use_thermal_consistency_loss: bool = True
    use_mixed_precision: bool = False
    validation_frequency_epochs: int = 1
    checkpoint_frequency_epochs: int = 1

class TrainingRunCreateRequest(BaseModel):
    model_architecture: str  # thermal_cnn_baseline, rgb_fusion, reliability_aware_fusion
    dataset_id: Optional[str] = None
    scale_factor: int = Field(default=8, ge=2, le=16)
    total_epochs: int = Field(default=10, ge=1, le=200)
    random_seed: int = 42
    device: Optional[str] = None  # auto, cuda, cpu
    hyperparameters: Optional[HyperparametersConfig] = None

class MetricPoint(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    psnr: float
    ssim: float
    lr: float

class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    model_architecture: str
    dataset_id: Optional[str] = None
    scale_factor: int
    hyperparameters: Dict[str, Any]
    random_seed: int
    device_info: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    elapsed_seconds: float
    current_epoch: int
    total_epochs: int
    best_checkpoint_path: Optional[str] = None
    latest_metrics: Dict[str, Any]
    metrics_history: List[Dict[str, Any]]
    error_details: Optional[str] = None
    created_at: datetime
