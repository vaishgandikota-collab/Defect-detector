"""
Baseline Convolutional Neural Network built strictly from first principles.
Does NOT use any pretrained weights. Configurable for regularization ablation experiments.
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    conv_filters: Optional[List[int]] = None,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    l2_reg: float = 0.0001,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    model_name: str = "baseline_manufacturing_cnn"
) -> keras.Model:
    """
    Constructs and compiles a deep convolutional network for defect classification.

    Architecture Flow:
        Input
        -> [Conv2D -> (BatchNorm) -> ReLU -> MaxPool2D] (Block 1)
        -> [Conv2D -> (BatchNorm) -> ReLU -> MaxPool2D] (Block 2)
        -> [Conv2D -> (BatchNorm) -> ReLU -> MaxPool2D] (Block 3)
        -> GlobalAveragePooling2D
        -> Dense(dense_units) -> (BatchNorm) -> ReLU
        -> (Dropout)
        -> Dense(num_classes, softmax)

    Args:
        input_shape: Dimensions of input images (H, W, C).
        num_classes: Total target defect categories.
        conv_filters: Filter depth sequence for conv blocks (default: [32, 64, 128]).
        dense_units: Neuron count in fully connected hidden layer.
        dropout_rate: Dropout fraction (0.0 to disable).
        use_batch_norm: Whether to apply batch normalization after convolutions.
        l2_reg: L2 kernel regularization coefficient.
        learning_rate: Initial optimizer learning rate.
        optimizer_name: Optimizer algorithm ("adam", "sgd", "rmsprop").
        model_name: Model identifier string.

    Returns:
        keras.Model: Compiled Keras functional model.
    """
    if conv_filters is None:
        conv_filters = [32, 64, 128]

    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    # Convolutional blocks
    for i, filters in enumerate(conv_filters, start=1):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_regularizer=regularizer,
            use_bias=not use_batch_norm,
            name=f"conv_{i}"
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
            
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{i}")(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    
    x = layers.Dense(
        dense_units,
        kernel_regularizer=regularizer,
        use_bias=not use_batch_norm,
        name="dense_feature"
    )(x)
    
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_dense")(x)
        
    x = layers.Activation("relu", name="relu_dense")(x)

    if dropout_rate > 0:
        x = layers.Dropout(rate=dropout_rate, name="dropout_head")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizer,
        name="defect_probabilities"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Optimizer selection
    opt_lower = optimizer_name.lower()
    if opt_lower == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif opt_lower == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
