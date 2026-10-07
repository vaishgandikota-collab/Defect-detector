"""
Automated unit tests for data validation, preprocessing, and pipeline loaders.
"""

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.data.augmentation import build_augmentation_layers, visualize_augmentation
from src.data.loader import create_datasets, discover_classes
from src.data.preprocessing import preprocess_image
from src.data.validation import DatasetValidator


def test_preprocess_image_from_array():
    # Test valid image numpy array preprocessing
    dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_img, target_size=(224, 224), normalize=True)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocess_image_grayscale():
    # Grayscale image converted to 3-channel RGB
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    processed = preprocess_image(gray_img, target_size=(224, 224), normalize=True)

    assert processed.shape == (224, 224, 3)
    assert processed.min() >= 0.0


def test_augmentation_pipeline():
    aug = build_augmentation_layers({"enabled": True, "horizontal_flip": True, "rotation_factor": 0.1})
    dummy = np.random.uniform(0.0, 1.0, (1, 224, 224, 3)).astype(np.float32)
    augmented = aug(dummy, training=True)

    assert augmented.shape == (1, 224, 224, 3)

    variations = visualize_augmentation(dummy[0], aug_pipeline=aug, num_variations=3)
    assert len(variations) == 3
    assert variations[0].shape == (224, 224, 3)


def test_dataset_validator_empty_dir(tmp_path):
    validator = DatasetValidator(tmp_path)
    report = validator.inspect_directory()
    assert report["exists"] is True
    assert report["total_images"] == 0
