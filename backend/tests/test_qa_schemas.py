import pytest
from pydantic import ValidationError
from app.qa.schemas import (
    QAMetadataInput,
    QAAssessmentResponse,
    ExperimentRecommendationRequest,
    ExperimentRecommendationResponse,
    QAHealthResponse,
)

def test_valid_qa_metadata_input():
    data = {
        "sample_id": "test_sample_01",
        "scale_factor": 8,
        "thermal_lr_shape": [80, 60],
        "thermal_hr_shape": [640, 480],
        "rgb_shape": [640, 480, 3],
        "alignment_status": "verified",
        "ground_truth_available": True,
        "metrics": {
            "psnr": 32.5,
            "ssim": 0.895,
            "lpips": 0.12,
        }
    }
    model = QAMetadataInput(**data)
    assert model.sample_id == "test_sample_01"
    assert model.scale_factor == 8
    assert model.metrics["psnr"] == 32.5

def test_negative_dimensions_rejected():
    with pytest.raises(ValidationError) as exc:
        QAMetadataInput(
            sample_id="test_02",
            scale_factor=8,
            thermal_lr_shape=[-80, 60],
        )
    assert "positive integers" in str(exc.value)

def test_unsupported_scale_factor_rejected():
    with pytest.raises(ValidationError) as exc:
        QAMetadataInput(
            sample_id="test_03",
            scale_factor=7,  # Unsupported: must be in [2, 4, 8, 16]
            thermal_lr_shape=[80, 60],
        )
    assert "Unsupported scale factor" in str(exc.value)

def test_ssim_out_of_range_rejected():
    with pytest.raises(ValidationError) as exc:
        QAMetadataInput(
            sample_id="test_04",
            scale_factor=8,
            thermal_lr_shape=[80, 60],
            metrics={"ssim": 1.5}  # Invalid SSIM > 1.0
        )
    assert "SSIM must be between -1.0 and 1.0" in str(exc.value)

def test_psnr_negative_rejected():
    with pytest.raises(ValidationError) as exc:
        QAMetadataInput(
            sample_id="test_05",
            scale_factor=8,
            thermal_lr_shape=[80, 60],
            metrics={"psnr": -5.0}  # Invalid negative PSNR
        )
    assert "PSNR must be between" in str(exc.value)

def test_optional_metrics_missing():
    model = QAMetadataInput(
        sample_id="test_06",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        metrics={}
    )
    assert model.metrics == {}
    assert model.thermal_hr_shape is None
    assert model.ground_truth_available is False
