"""
Image preprocessing and tensor transformation pipelines.
Provides standalone preprocessing functions and tf.data graph-compatible operations.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

DEFAULT_IMAGE_SIZE = (224, 224)


def preprocess_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Loads, cleans, resizes, and normalizes a single image for model ingestion.

    Args:
        image_input: File path, raw bytes, numpy array, or PIL Image.
        target_size: Tuple (height, width).
        normalize: If True, rescales pixel intensities from [0, 255] to [0.0, 1.0].

    Returns:
        np.ndarray: Preprocessed float32 array of shape (height, width, 3).
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {path.resolve()}")
        # Read with PIL for robust RGB handling
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            img_array = np.array(resized_img, dtype=np.float32)

    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError("Failed to decode image from bytes. File may be corrupted.")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        img_array = resized_img.astype(np.float32)

    elif isinstance(image_input, Image.Image):
        rgb_img = image_input.convert("RGB")
        resized_img = rgb_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        img_array = np.array(resized_img, dtype=np.float32)

    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.ndim == 2:  # Grayscale
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[-1] == 4:  # RGBA
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        
        resized = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        img_array = resized.astype(np.float32)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if normalize:
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

    return img_array


def build_preprocess_fn(
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_classes: Optional[int] = None,
    normalize: bool = True
) -> Callable:
    """
    Creates a TensorFlow graph-compatible preprocessing function for tf.data pipelines.
    """
    def _parse_and_preprocess(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Read raw byte string from file
        img_bytes = tf.io.read_file(file_path)
        # Decode image to 3-channel RGB uint8 tensor
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [target_size[0], target_size[1]], method="bilinear")
        img = tf.cast(img, tf.float32)
        
        if normalize:
            img = img / 255.0

        if num_classes is not None:
            # One-hot encode label
            label = tf.one_hot(label, depth=num_classes)

        return img, label

    return _parse_and_preprocess
