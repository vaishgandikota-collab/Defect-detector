"""
Unit tests for real-time predictor engine, Grad-CAM generation, and defect decision logic.
"""

import numpy as np
import pytest
from PIL import Image

from src.inference.predictor import DefectPredictor
from src.models.baseline_cnn import build_baseline_cnn


def test_predictor_lifecycle(tmp_path):
    # Create a small dummy model and save it
    model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=4)
    model_path = tmp_path / "test_model.keras"
    model.save(str(model_path))

    classes = ["crazing", "inclusion", "scratches", "normal"]
    predictor = DefectPredictor(model_path=model_path, class_names=classes)

    assert predictor.is_ready() is True

    # Test single prediction with synthetic PIL image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = predictor.predict(img, generate_explainability=True)

    assert "predicted_class" in result
    assert result["predicted_class"] in classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert "inference_time_ms" in result
    assert "is_defective" in result
    assert "recommendation" in result
    assert "top_predictions" in result
    assert len(result["top_predictions"]) <= 3


def test_batch_prediction(tmp_path):
    model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=3)
    model_path = tmp_path / "test_batch_model.keras"
    model.save(str(model_path))

    predictor = DefectPredictor(model_path=model_path, class_names=["defect_a", "defect_b", "normal"])

    imgs = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    ]
    batch_results = predictor.predict_batch(imgs)

    assert len(batch_results) == 2
    assert batch_results[0]["predicted_class"] in ["defect_a", "defect_b", "normal"]
