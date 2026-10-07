"""
Model utility routines: parameter counting, disk footprint calculation,
CPU inference speed benchmarking, and artifact serialization.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.file_utils import save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def count_parameters(model: keras.Model) -> Dict[str, int]:
    """
    Computes total, trainable, and non-trainable parameter counts.
    """
    trainable_count = int(np.sum([keras.ops.size(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([keras.ops.size(p) for p in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count

    return {
        "total_parameters": total_count,
        "trainable_parameters": trainable_count,
        "non_trainable_parameters": non_trainable_count,
    }


def compute_model_size_mb(model_path: Union[str, Path]) -> float:
    """
    Measures the disk size of a saved model in Megabytes.
    """
    p = Path(model_path)
    if not p.exists():
        return 0.0
    
    if p.is_file():
        return round(p.stat().st_size / (1024 * 1024), 2)
    
    total_bytes = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def benchmark_inference_speed(
    model: keras.Model,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_warmup: int = 10,
    num_runs: int = 50,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Measures realistic CPU forward-pass latency, standard deviation, and throughput (FPS).
    """
    dummy_input = np.random.uniform(0.0, 1.0, size=(batch_size, *input_shape)).astype(np.float32)

    # Warmup runs to compile tf.function graph
    for _ in range(num_warmup):
        _ = model(dummy_input, training=False)

    latencies_ms: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies_ms))
    std_latency = float(np.std(latencies_ms))
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    fps = round(1000.0 / avg_latency * batch_size, 2) if avg_latency > 0 else 0.0

    return {
        "avg_latency_ms": round(avg_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "throughput_fps": fps,
    }


def save_model_artifacts(
    model: keras.Model,
    save_path: Union[str, Path],
    class_names: List[str],
    metadata_path: Optional[Union[str, Path]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves the serialized Keras model (.keras format) alongside associated metadata JSON.
    """
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Save standard Keras format
    model.save(str(target))
    logger.info(f"Model saved successfully to {target.resolve()}")

    params = count_parameters(model)
    model_size_mb = compute_model_size_mb(target)

    meta = {
        "model_name": getattr(model, "name", target.stem),
        "save_path": str(target),
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_shape": list(model.input_shape[1:] if model.input_shape else [224, 224, 3]),
        "parameters": params,
        "model_size_mb": model_size_mb,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": f"TensorFlow {tf.__version__} / Keras {keras.__version__}",
    }

    if extra_metadata:
        meta.update(extra_metadata)

    if metadata_path:
        save_json(meta, metadata_path)
        logger.info(f"Model metadata saved to {metadata_path}")
