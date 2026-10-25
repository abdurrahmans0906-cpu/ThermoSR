from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class DatasetScanRequest(BaseModel):
    root_path: Optional[str] = None
    name: Optional[str] = None
    force_rescan: bool = False

class DatasetPairResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    dataset_id: str
    split: str
    thermal_path: str
    rgb_path: Optional[str] = None
    reference_path: Optional[str] = None
    dimensions: Dict[str, Any]
    validation_status: str
    alignment_status: str
    error_message: Optional[str] = None

class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    root_dir: str
    dataset_type: str
    split_config: Dict[str, Any]
    total_pairs: int
    valid_pairs: int
    unmatched_count: int
    status: str
    created_at: datetime
    updated_at: datetime

class DatasetValidationReport(BaseModel):
    dataset_id: str
    dataset_name: str
    root_dir: str
    total_scanned: int
    valid_pairs: int
    missing_rgb: int
    missing_thermal: int
    dimension_issues: List[Dict[str, Any]]
    aspect_ratio_warnings: List[Dict[str, Any]]
    split_breakdown: Dict[str, int]
    is_ready_for_training: bool
    summary: str
