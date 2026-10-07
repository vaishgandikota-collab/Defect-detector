"""
Lightweight edge-optimized deployment architecture based on MobileNetV3Small.
Engineered for ultra-low latency real-time conveyor belt quality inspection on CPU.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_mobilenet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    weights: str = "imagenet",
    dense_units: int = 128,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    model_name: str = "mobilenet_v3_small_defect_detector"
) -> keras.Model:
    """
    Constructs a lightweight MobileNetV3-Small architecture for edge deployment.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Target defect categories.
        weights: Pretrained weights ("imagenet" or None).
        dense_units: Dense layer capacity in head.
        dropout_rate: Regularization rate.
        learning_rate: Initial optimizer learning rate.
        optimizer_name: Optimizer algorithm.
        model_name: Model identifier string.

    Returns:
        keras.Model: Compiled MobileNetV3Small model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Instantiate MobileNetV3Small backbone
    base_model = keras.applications.MobileNetV3Small(
        input_tensor=inputs,
        weights=weights,
        include_top=False,
        pooling=None,
        include_preprocessing=False
    )

    # Freeze base initially
    base_model.trainable = False

    # Compact classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="mobilenet_gap")(x)
    x = layers.BatchNormalization(name="mobilenet_bn")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="mobilenet_dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="mobilenet_dense")(x)
    
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="defect_probabilities"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    opt_lower = optimizer_name.lower()
    if opt_lower == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif opt_lower == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info("Initialized Lightweight MobileNetV3Small deployment model.")
    return model
