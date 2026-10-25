from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class ExperimentCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    experiment_type: str = "ablation_study"
    training_run_ids: List[str]

class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: Optional[str] = None
    experiment_type: str
    related_training_run_ids: List[str]
    comparison_results: Dict[str, Any]
    created_at: datetime
