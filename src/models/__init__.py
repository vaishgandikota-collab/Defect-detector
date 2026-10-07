"""
Model architecture definitions and utility functions.
"""

from src.models.baseline_cnn import build_baseline_cnn
from src.models.transfer_learning import build_transfer_model, unfreeze_backbone_layers
from src.models.mobilenet_model import build_mobilenet_model
from src.models.model_utils import (
    count_parameters,
    compute_model_size_mb,
    benchmark_inference_speed,
    save_model_artifacts
)

__all__ = [
    "build_baseline_cnn",
    "build_transfer_model",
    "unfreeze_backbone_layers",
    "build_mobilenet_model",
    "count_parameters",
    "compute_model_size_mb",
    "benchmark_inference_speed",
    "save_model_artifacts",
]
