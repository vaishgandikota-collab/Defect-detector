"""
Unit tests for model construction, compilation, forward-pass integrity, and parameter counting.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.baseline_cnn import build_baseline_cnn
from src.models.mobilenet_model import build_mobilenet_model
from src.models.model_utils import benchmark_inference_speed, count_parameters
from src.models.transfer_learning import build_transfer_model, unfreeze_backbone_layers


def test_baseline_cnn_architecture():
    num_classes = 5
    model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=num_classes, dense_units=64)
    
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, num_classes)

    dummy_input = np.random.uniform(0.0, 1.0, (2, 224, 224, 3)).astype(np.float32)
    output = model(dummy_input, training=False)
    
    assert output.shape == (2, num_classes)
    # Check softmax sum to 1
    assert np.allclose(np.sum(output.numpy(), axis=-1), 1.0, atol=1e-5)


def test_transfer_model_build_and_unfreeze():
    model = build_transfer_model(input_shape=(224, 224, 3), num_classes=6, weights=None)
    assert model.output_shape == (None, 6)

    params_before = count_parameters(model)
    assert params_before["total_parameters"] > 0

    model = unfreeze_backbone_layers(model, num_layers_to_unfreeze=10)
    params_after = count_parameters(model)
    assert params_after["trainable_parameters"] >= params_before["trainable_parameters"]


def test_mobilenet_model():
    model = build_mobilenet_model(input_shape=(224, 224, 3), num_classes=7, weights=None)
    assert model.output_shape == (None, 7)

    dummy_input = np.random.uniform(0.0, 1.0, (1, 224, 224, 3)).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape == (1, 7)


def test_inference_benchmark():
    model = build_baseline_cnn(input_shape=(224, 224, 3), num_classes=3)
    bench = benchmark_inference_speed(model, num_warmup=2, num_runs=5)
    assert "avg_latency_ms" in bench
    assert bench["avg_latency_ms"] > 0
