"""
Integration tests for FastAPI REST endpoints and request validation.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from src.inference.predictor import DefectPredictor
from src.models.baseline_cnn import build_baseline_cnn


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Setup a dummy model for the API test environment
    model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=3)
    model_path = tmp_path / "api_test_model.keras"
    model.save(str(model_path))

    dummy_predictor = DefectPredictor(
        model_path=model_path,
        class_names=["scratch", "inclusion", "normal"]
    )
    monkeypatch.setattr("api.main.get_predictor", lambda: dummy_predictor)
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 3


def test_predict_endpoint_valid_image(client):
    # Create test image in memory
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_defect.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "is_defective" in data
    assert "status" in data
    assert "recommendation" in data
    assert "inference_time_ms" in data


def test_predict_endpoint_invalid_file_format(client):
    # Upload unsupported text file
    fake_file = io.BytesIO(b"Hello world, not an image")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
