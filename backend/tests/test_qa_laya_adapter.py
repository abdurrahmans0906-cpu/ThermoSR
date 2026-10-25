import pytest
from app.qa.schemas import QAMetadataInput
from app.qa.laya_adapter import laya_adapter

def test_laya_offline_fallback():
    # Clear mock
    laya_adapter.clear_mock()

    meta = QAMetadataInput(
        sample_id="test_fallback",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        ground_truth_available=False,
    )
    res = laya_adapter.assess(meta)
    assert res["status"] in ["unavailable", "success"]
    assert res["reconstruction_modified"] is False
    assert res["assessment"] == "insufficient_evidence"

def test_laya_mocked_success_with_probabilities():
    # Inject mock Laya decision
    mock_probabilities = {
        "ready_for_review": 0.88,
        "human_review": 0.08,
        "insufficient_evidence": 0.04
    }
    laya_adapter.set_mock_predict(lambda meta: {
        "assessment": "ready_for_review",
        "probabilities": mock_probabilities
    })

    meta = QAMetadataInput(
        sample_id="test_mock",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[640, 480],
        rgb_shape=[640, 480, 3],
        alignment_status="verified",
        ground_truth_available=True,
        metrics={"psnr": 33.1, "ssim": 0.91}
    )
    res = laya_adapter.assess(meta)
    assert res["status"] == "success"
    assert res["assessment"] == "ready_for_review"
    assert res["probabilities"] == mock_probabilities
    assert res["model_latency_ms"] is not None
    assert res["reconstruction_modified"] is False

    laya_adapter.clear_mock()

def test_laya_deterministic_safety_override_on_failed_alignment():
    # Even if Laya returns 'ready_for_review', failed alignment MUST override to 'human_review'
    laya_adapter.set_mock_predict(lambda meta: {
        "assessment": "ready_for_review",
        "probabilities": {"ready_for_review": 0.99, "human_review": 0.01}
    })

    meta = QAMetadataInput(
        sample_id="test_override",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[640, 480],
        rgb_shape=[640, 480, 3],
        alignment_status="failed",  # Optical alignment failed!
        ground_truth_available=True,
        metrics={"psnr": 33.1, "ssim": 0.91}
    )
    res = laya_adapter.assess(meta)
    assert res["assessment"] == "human_review"  # Overridden!
    assert any("alignment explicitly failed" in w.lower() for w in res["warnings"])

    laya_adapter.clear_mock()

def test_laya_deterministic_safety_override_on_missing_gt():
    # If ground truth is missing, Laya cannot mark it 'ready_for_review'
    laya_adapter.set_mock_predict(lambda meta: {
        "assessment": "ready_for_review",
        "probabilities": {"ready_for_review": 0.95}
    })

    meta = QAMetadataInput(
        sample_id="test_override_gt",
        scale_factor=8,
        thermal_lr_shape=[80, 60],
        thermal_hr_shape=[640, 480],
        ground_truth_available=False,  # No GT
    )
    res = laya_adapter.assess(meta)
    assert res["assessment"] == "insufficient_evidence"  # Overridden!

    laya_adapter.clear_mock()
