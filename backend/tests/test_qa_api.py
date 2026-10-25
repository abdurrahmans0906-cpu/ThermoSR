import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_qa_health_endpoint():
    res = client.get("/api/qa/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "laya_installed" in data
    assert "fallback_mode" in data

def test_qa_assess_valid_payload():
    payload = {
        "sample_id": "test_api_sample",
        "scale_factor": 8,
        "thermal_lr_shape": [80, 60],
        "thermal_hr_shape": [640, 480],
        "rgb_shape": [640, 480, 3],
        "alignment_status": "verified",
        "ground_truth_available": True,
        "metrics": {
            "psnr": 32.4,
            "ssim": 0.89
        }
    }
    res = client.post("/api/qa/laya-assess", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["assessment"] in ["ready_for_review", "human_review", "insufficient_evidence", "unavailable"]
    assert data["reconstruction_modified"] is False
    assert "disclaimer" in data

def test_qa_assess_invalid_payload_returns_422():
    payload = {
        "sample_id": "test_invalid",
        "scale_factor": 7,  # Invalid scale factor
        "thermal_lr_shape": [-80, 60],  # Negative dimension
    }
    res = client.post("/api/qa/laya-assess", json=payload)
    assert res.status_code == 422

def test_qa_experiment_recommendation_valid():
    payload = {
        "current_architecture": "thermal_cnn_baseline",
        "available_experiments": ["thermal_baseline", "rgb_fusion", "reliability_fusion"],
        "scale_factor": 8,
        "alignment_status": "verified",
        "ground_truth_available": True,
        "metrics": {"psnr": 28.5, "ssim": 0.82}
    }
    res = client.post("/api/qa/experiment-recommendation", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["recommendation"] == "rgb_fusion"
    assert data["executed"] is False

def test_original_routes_backward_compatible():
    # Verify GET /api/health still works as before
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
