"""
Data Augmentation pipeline for industrial surface defects.
Applies photometrically and geometrically plausible transforms ONLY to training data.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation_layers(aug_config: Optional[Dict[str, Any]] = None) -> keras.Sequential:
    """
    Constructs a Keras Sequential layer container for industrial image augmentation.
    Transformations are constrained to avoid physically unrealistic manufacturing distortions.

    Args:
        aug_config: Dictionary containing augmentation hyperparameter overrides.

    Returns:
        keras.Sequential: Pre-configured augmentation pipeline.
    """
    cfg = aug_config or {}
    enabled = cfg.get("enabled", True)
    
    if not enabled:
        return keras.Sequential([], name="identity_augmentation")

    pipeline_layers = []

    if cfg.get("horizontal_flip", True):
        pipeline_layers.append(layers.RandomFlip("horizontal", name="aug_hflip"))

    if cfg.get("vertical_flip", True):
        pipeline_layers.append(layers.RandomFlip("vertical", name="aug_vflip"))

    rot = cfg.get("rotation_factor", 0.15)
    if rot > 0:
        pipeline_layers.append(layers.RandomRotation(factor=rot, fill_mode="reflect", name="aug_rotation"))

    zoom = cfg.get("zoom_factor", 0.1)
    if zoom > 0:
        pipeline_layers.append(layers.RandomZoom(height_factor=(-zoom, zoom), fill_mode="reflect", name="aug_zoom"))

    trans = cfg.get("translation_factor", 0.08)
    if trans > 0:
        pipeline_layers.append(layers.RandomTranslation(
            height_factor=(-trans, trans),
            width_factor=(-trans, trans),
            fill_mode="reflect",
            name="aug_translation"
        ))

    contrast = cfg.get("contrast_factor", 0.1)
    if contrast > 0:
        pipeline_layers.append(layers.RandomContrast(factor=contrast, name="aug_contrast"))

    return keras.Sequential(pipeline_layers, name="manufacturing_augmentation")


def visualize_augmentation(
    image_np: np.ndarray,
    aug_pipeline: Optional[keras.Sequential] = None,
    num_variations: int = 5,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Generates stochastic augmented variations for an input image for UI visualization.

    Args:
        image_np: Float32 image array of shape (H, W, 3) in [0, 1].
        aug_pipeline: Augmentation sequential model.
        num_variations: Number of variations to produce.
        seed: Random seed.

    Returns:
        List[np.ndarray]: List of augmented image arrays.
    """
    if aug_pipeline is None:
        aug_pipeline = build_augmentation_layers()

    # Ensure batch dimension
    if image_np.ndim == 3:
        batch_input = np.expand_dims(image_np, axis=0)
    else:
        batch_input = image_np

    tf.random.set_seed(seed)
    variations = []
    for i in range(num_variations):
        # Forward pass with training=True enables stochastic augmentation
        augmented_tensor = aug_pipeline(batch_input, training=True)
        aug_np = np.clip(augmented_tensor.numpy()[0], 0.0, 1.0)
        variations.append(aug_np)

    return variations
