"""
Training callbacks and real-time gradient/optimization diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class GradientDiagnosticsCallback(keras.callbacks.Callback):
    """
    Monitors layer weight updates and assesses potential vanishing or exploding gradients during training.
    """

    def __init__(self, check_frequency: int = 1):
        super().__init__()
        self.check_frequency = check_frequency
        self.gradient_history: List[Dict[str, Any]] = []
        self.prev_weights: Dict[str, np.ndarray] = {}

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        # Store current trainable weights to compute delta post-epoch
        self.prev_weights = {
            layer.name: layer.get_weights()[0]
            for layer in self.model.layers
            if hasattr(layer, "get_weights") and len(layer.get_weights()) > 0
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if epoch % self.check_frequency != 0:
            return

        weight_deltas = []
        for layer in self.model.layers:
            if layer.name in self.prev_weights and len(layer.get_weights()) > 0:
                curr_w = layer.get_weights()[0]
                prev_w = self.prev_weights[layer.name]
                delta_norm = float(np.linalg.norm(curr_w - prev_w))
                weight_deltas.append(delta_norm)

        avg_delta = float(np.mean(weight_deltas)) if weight_deltas else 0.0
        max_delta = float(np.max(weight_deltas)) if weight_deltas else 0.0
        min_delta = float(np.min(weight_deltas)) if weight_deltas else 0.0

        status = "STABLE"
        if avg_delta < 1e-6:
            status = "POTENTIAL_VANISHING_GRADIENTS"
        elif avg_delta > 10.0:
            status = "POTENTIAL_EXPLODING_GRADIENTS"

        entry = {
            "epoch": epoch + 1,
            "avg_weight_delta": avg_delta,
            "min_weight_delta": min_delta,
            "max_weight_delta": max_delta,
            "optimization_status": status,
        }
        self.gradient_history.append(entry)


def build_callbacks(
    checkpoint_path: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    csv_log_path: Optional[Union[str, Path]] = None,
    patience_early_stopping: int = 8,
    patience_reduce_lr: int = 3,
    min_lr: float = 1e-6,
    monitor: str = "val_loss",
    enable_tensorboard: bool = True
) -> List[keras.callbacks.Callback]:
    """
    Constructs an enterprise-grade callback suite for training execution.
    """
    callbacks: List[keras.callbacks.Callback] = []

    # 1. Early Stopping with best weights restoration
    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience_early_stopping,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
        mode="min" if "loss" in monitor else "max"
    )
    callbacks.append(early_stop)

    # 2. Learning Rate Scheduler on Plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience_reduce_lr,
        min_lr=min_lr,
        verbose=1,
        mode="min" if "loss" in monitor else "max"
    )
    callbacks.append(reduce_lr)

    # 3. Model Checkpointing
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode="min" if "loss" in monitor else "max"
        )
        callbacks.append(checkpoint)

    # 4. CSV Metrics Logger
    if csv_log_path:
        csv_path = Path(csv_log_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger = keras.callbacks.CSVLogger(str(csv_path), separator=",", append=False)
        callbacks.append(csv_logger)

    # 5. TensorBoard Logger
    if enable_tensorboard and log_dir:
        tb_dir = Path(log_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(tb_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch"
        )
        callbacks.append(tensorboard)

    # 6. Gradient & Optimization Diagnostics
    diagnostics = GradientDiagnosticsCallback()
    callbacks.append(diagnostics)

    return callbacks
