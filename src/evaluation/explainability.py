"""
Explainability and Model Interpretability using Gradient-weighted Class Activation Mapping (Grad-CAM).
Localizes spatial regions in industrial surface images that heavily drive defect predictions.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def find_target_conv_layer(model: keras.Model) -> Optional[str]:
    """
    Locates the outermost 2D convolutional layer in the network graph.
    """
    # First search direct layers in reverse
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # If there's an inner functional backbone (like ResNet or MobileNet)
        if isinstance(layer, keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    return f"{layer.name}/{sub_layer.name}"
                
    # Search by layer name keyword fallback
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower() and not "dropout" in layer.name.lower():
            return layer.name

    return None


def generate_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    target_class_idx: Optional[int] = None,
    layer_name: Optional[str] = None
) -> Tuple[Optional[np.ndarray], str]:
    """
    Computes Grad-CAM 2D heatmap tensor for an input image and target class.

    Args:
        model: Keras classification model.
        img_array: Batch tensor or array of shape (1, H, W, 3) in [0, 1].
        target_class_idx: Index of class to compute gradients for (default: top-predicted).
        layer_name: Specific conv layer name (default: auto-detected last conv).

    Returns:
        Tuple[Optional[np.ndarray], str]: Heatmap array in [0, 1] of shape (H, W), status message.
    """
    if layer_name is None:
        layer_name = find_target_conv_layer(model)

    if layer_name is None:
        return None, "No suitable 2D convolutional layer found in model architecture for Grad-CAM."

    try:
        # Check if layer is in a nested model
        if "/" in layer_name:
            outer_name, inner_name = layer_name.split("/", 1)
            outer_submodel = model.get_layer(outer_name)
            target_layer = outer_submodel.get_layer(inner_name)
            grad_model = keras.models.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )
        else:
            target_layer = model.get_layer(layer_name)
            grad_model = keras.models.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )

        # Forward pass and gradient computation under GradientTape
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if target_class_idx is None:
                target_class_idx = tf.argmax(predictions[0])
            loss = predictions[:, target_class_idx]

        # Gradients of target class score with respect to conv activation map
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None, "Failed to compute gradients with respect to convolutional layer."

        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        # Weighted combination of feature maps
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU on heatmap to keep only features that positively contribute
        heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-9)
        heatmap_np = heatmap.numpy()

        return heatmap_np, "SUCCESS"

    except Exception as exc:
        logger.warning(f"Grad-CAM generation error: {exc}")
        return None, f"Grad-CAM computation encountered an issue: {str(exc)}"


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Resizes heatmap and blends it smoothly over the original RGB image.
    """
    # Ensure original is uint8 [0, 255]
    if original_img.max() <= 1.0:
        base = np.uint8(255 * original_img)
    else:
        base = np.uint8(original_img)

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap onto original image
    superimposed = cv2.addWeighted(base, 1.0 - alpha, color_heatmap, alpha, 0)
    return superimposed
