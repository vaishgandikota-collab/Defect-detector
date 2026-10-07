"""
Seed utility for ensuring full experimental reproducibility across Python, NumPy, and TensorFlow.
"""

import os
import random
import numpy as np
import tensorflow as tf
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds for standard library, NumPy, and TensorFlow.

    Args:
        seed: The integer random seed (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_reproducibility_notes() -> str:
    """
    Returns technical documentation regarding reproducibility limitations on various hardware backends.
    """
    return (
        "Reproducibility Note:\n"
        "- Full bitwise reproducibility is guaranteed on CPU with single-threaded TF operations.\n"
        "- Non-deterministic CUDA kernels (such as atomicAdd in cuDNN convolutions) may cause minor "
        "floating-point divergence on GPUs.\n"
        "- Seeds configured for Python `random`, `numpy.random`, and `tf.random`."
    )
