import pytest
from app.qa.schemas import ExperimentRecommendationRequest
from app.qa.experiment_advisor import recommend_next_experiment

def test_advisor_baseline_recommends_rgb_fusion():
    req = ExperimentRecommendationRequest(
        current_architecture="thermal_cnn_baseline",
        available_experiments=["thermal_baseline", "rgb_fusion", "reliability_fusion"],
        scale_factor=8,
        alignment_status="verified",
        ground_truth_available=True,
        metrics={"psnr": 28.5, "ssim": 0.82}
    )
    res = recommend_next_experiment(req)
    assert res["recommendation"] == "rgb_fusion"
    assert res["executed"] is False
    assert "BASELINE_COMPLETED" in res["reason_codes"]

def test_advisor_rgb_fusion_recommends_reliability_fusion():
    req = ExperimentRecommendationRequest(
        current_architecture="rgb_fusion",
        available_experiments=["thermal_baseline", "rgb_fusion", "reliability_fusion"],
        scale_factor=8,
        alignment_status="verified",
        ground_truth_available=True,
        metrics={"psnr": 31.2, "ssim": 0.88}
    )
    res = recommend_next_experiment(req)
    assert res["recommendation"] == "reliability_fusion"
    assert res["executed"] is False

def test_advisor_safety_override_on_failed_alignment():
    req = ExperimentRecommendationRequest(
        current_architecture="thermal_cnn_baseline",
        available_experiments=["thermal_baseline", "rgb_fusion", "reliability_fusion"],
        scale_factor=8,
        alignment_status="failed",  # Failed!
        ground_truth_available=True,
        metrics={"psnr": 28.5, "ssim": 0.82}
    )
    res = recommend_next_experiment(req)
    assert res["recommendation"] == "human_review"
    assert "ALIGNMENT_FAILED" in res["reason_codes"]
    assert res["executed"] is False

def test_advisor_safety_override_on_missing_gt():
    req = ExperimentRecommendationRequest(
        current_architecture="thermal_cnn_baseline",
        available_experiments=["thermal_baseline", "rgb_fusion", "reliability_fusion"],
        scale_factor=8,
        alignment_status="verified",
        ground_truth_available=False,  # No GT!
        metrics={}
    )
    res = recommend_next_experiment(req)
    assert res["recommendation"] == "rerun_evaluation"
    assert "MISSING_GROUND_TRUTH" in res["reason_codes"]
    assert res["executed"] is False
