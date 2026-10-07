"""
Transfer learning architectures using deep ImageNet-pretrained backbones (e.g. ResNet50V2, EfficientNetB0).
Supports two-stage training workflow: feature extraction head training followed by upper layer fine-tuning.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    backbone_name: str = "ResNet50V2",
    weights: str = "imagenet",
    dense_units: int = 256,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    model_name: str = "transfer_defect_detector"
) -> keras.Model:
    """
    Constructs a transfer learning model with a frozen backbone and a custom classification head.

    Args:
        input_shape: Dimensions of input images (H, W, C).
        num_classes: Total defect categories.
        backbone_name: Backbone architecture ("ResNet50V2", "ResNet50", "EfficientNetB0").
        weights: Pretrained weights ("imagenet" or None).
        dense_units: Units in classification dense layer.
        dropout_rate: Dropout regularization fraction.
        learning_rate: Initial learning rate for Stage 1.
        optimizer_name: Optimizer algorithm.
        model_name: Model identifier string.

    Returns:
        keras.Model: Compiled transfer learning model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Select and instantiate backbone
    b_name = backbone_name.lower()
    if "resnet50v2" in b_name or "resnet50_v2" in b_name:
        base_model = keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
            input_tensor=inputs
        )
    elif "resnet50" in b_name:
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=inputs
        )
    elif "efficientnetb0" in b_name:
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_tensor=inputs
        )
    else:
        # Fallback to ResNet50V2
        base_model = keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
            input_tensor=inputs
        )

    # Freeze backbone for initial Stage 1 training
    base_model.trainable = False

    # Classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = layers.BatchNormalization(name="transfer_bn")(x)
    x = layers.Dropout(dropout_rate, name="transfer_dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="transfer_dense")(x)
    x = layers.Dropout(dropout_rate / 2, name="transfer_dropout_2")(x)
    
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

    logger.info(f"Initialized Transfer Model with {backbone_name} (Trainable base layers: {base_model.trainable})")
    return model


def unfreeze_backbone_layers(
    model: keras.Model,
    num_layers_to_unfreeze: int = 25,
    fine_tune_lr: float = 0.00005,
    optimizer_name: str = "adam"
) -> keras.Model:
    """
    Unfreezes the top N layers of the backbone for Stage 2 fine-tuning with a lower learning rate.
    """
    # Locate backbone submodel or direct layers
    # In functional models created with input_tensor, find layers before transfer_gap
    total_layers = len(model.layers)
    logger.info(f"Total model layers: {total_layers}. Unfreezing top {num_layers_to_unfreeze} layers for fine-tuning.")

    # Unfreeze only the last `num_layers_to_unfreeze` layers
    for layer in model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    for layer in model.layers[-num_layers_to_unfreeze:]:
        # Keep BatchNormalization in inference mode during fine-tuning for stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    opt_lower = optimizer_name.lower()
    if opt_lower == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=fine_tune_lr, momentum=0.9)
    elif opt_lower == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=fine_tune_lr)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=fine_tune_lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Model recompiled for fine-tuning with learning rate: {fine_tune_lr}")
    return model
