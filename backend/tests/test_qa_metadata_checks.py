import pytest
from app.qa.schemas import QAMetadataInput
from app.qa.metadata_checks import perform_deterministic_checks

def test_missing_ground_truth_deterministic():
    meta = QAMetadataInput(
        sample_id="run_no_gt",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[640, 480],
        ground_truth_available=False,
    )
    status, warnings, missing_evidence, checks = perform_deterministic_checks(meta)
    assert status == "insufficient_evidence"
    assert any("ground-truth" in ev.lower() for ev in missing_evidence)

def test_alignment_failed_deterministic():
    meta = QAMetadataInput(
        sample_id="run_align_fail",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[640, 480],
        rgb_shape=[640, 480, 3],
        alignment_status="failed",
        ground_truth_available=True,
        metrics={"psnr": 30.0, "ssim": 0.85}
    )
    status, warnings, missing_evidence, checks = perform_deterministic_checks(meta)
    assert status == "human_review"
    assert any("optical alignment explicitly failed" in w.lower() for w in warnings)

def test_dimension_scale_mismatch_deterministic():
    meta = QAMetadataInput(
        sample_id="run_dim_mismatch",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[500, 300],  # Mismatched! 80*8=640 != 500
        ground_truth_available=True,
        metrics={"psnr": 30.0, "ssim": 0.85}
    )
    status, warnings, missing_evidence, checks = perform_deterministic_checks(meta)
    assert status == "human_review"
    assert checks["dimension_scale_consistent"] is False
    assert any("inconsistency" in w.lower() for w in warnings)

def test_valid_verified_metadata_deterministic():
    meta = QAMetadataInput(
        sample_id="run_clean",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[640, 480],
        rgb_shape=[640, 480, 3],
        alignment_status="verified",
        ground_truth_available=True,
        metrics={"psnr": 34.2, "ssim": 0.92, "lpips": 0.11}
    )
    status, warnings, missing_evidence, checks = perform_deterministic_checks(meta)
    assert status == "ready_for_review"
    assert checks["dimension_scale_consistent"] is True
    assert checks["has_psnr"] is True
    assert checks["has_ssim"] is True
    assert len(missing_evidence) == 0
