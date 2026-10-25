import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.session import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    root_dir = Column(String(500), nullable=False)
    dataset_type = Column(String(50), default="cidis")  # cidis or challenge_flir
    split_config = Column(JSON, default=dict)
    total_pairs = Column(Integer, default=0)
    valid_pairs = Column(Integer, default=0)
    unmatched_count = Column(Integer, default=0)
    status = Column(String(50), default="scanned")  # scanned, validated, error
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    pairs = relationship("DatasetPair", back_populates="dataset", cascade="all, delete-orphan")

class DatasetPair(Base):
    __tablename__ = "dataset_pairs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    split = Column(String(20), default="train")  # train, validation, test
    thermal_path = Column(String(500), nullable=False)
    rgb_path = Column(String(500), nullable=True)
    reference_path = Column(String(500), nullable=True)
    dimensions = Column(JSON, default=dict)  # {"thermal": [W, H], "rgb": [W, H]}
    validation_status = Column(String(50), default="valid")  # valid, missing_rgb, corrupted
    alignment_status = Column(String(50), default="unknown")  # aligned, aspect_mismatch, unknown
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    dataset = relationship("Dataset", back_populates="pairs")

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    architecture = Column(String(50), nullable=False)  # thermal_cnn_baseline, rgb_fusion, reliability_aware_fusion
    version = Column(String(20), default="v1.0")
    checkpoint_path = Column(String(500), nullable=True)
    configuration = Column(JSON, default=dict)
    training_run_id = Column(String(36), nullable=True)
    supported_scales = Column(JSON, default=lambda: [2, 4, 8])
    validation_metrics = Column(JSON, default=dict)  # psnr, ssim, lpips
    is_available = Column(Boolean, default=False)
    is_trained = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_architecture = Column(String(50), nullable=False)
    dataset_id = Column(String(36), nullable=True)
    scale_factor = Column(Integer, default=8)
    hyperparameters = Column(JSON, default=dict)
    random_seed = Column(Integer, default=42)
    device_info = Column(String(100), default="cpu")
    status = Column(String(30), default="pending")  # pending, running, completed, failed, cancelled
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    elapsed_seconds = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=10)
    best_checkpoint_path = Column(String(500), nullable=True)
    latest_metrics = Column(JSON, default=dict)
    metrics_history = Column(JSON, default=list)  # list of {epoch, train_loss, val_loss, psnr, ssim, lr}
    error_details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

class ReconstructionRun(Base):
    __tablename__ = "reconstruction_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    task_mode = Column(String(50), default="rgb_guided")  # rgb_guided, evaluation, downsample_eval
    model_version_id = Column(String(36), nullable=True)
    model_architecture = Column(String(50), nullable=True)
    scale_factor = Column(Integer, default=8)
    input_artifacts = Column(JSON, default=dict)  # paths to uploaded images
    output_artifacts = Column(JSON, default=dict)  # paths to reconstructed, colormap, reliability
    configuration = Column(JSON, default=dict)  # inference settings
    status = Column(String(30), default="queued")  # queued, validating, preprocessing, running_inference, calculating_metrics, completed, failed, cancelled
    progress_pct = Column(Integer, default=0)
    metrics = Column(JSON, default=dict)  # psnr, ssim, lpips, thermal_consistency
    inference_time_ms = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)
    experiment_type = Column(String(50), default="ablation_study")
    related_training_run_ids = Column(JSON, default=list)
    comparison_results = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utc_now)
