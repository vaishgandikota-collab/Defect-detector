"""
Unified model training routines for Baseline CNN, Transfer Learning, and MobileNet.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.baseline_cnn import build_baseline_cnn
from src.models.mobilenet_model import build_mobilenet_model
from src.models.model_utils import (
    benchmark_inference_speed,
    count_parameters,
    save_model_artifacts
)
from src.models.transfer_learning import build_transfer_model, unfreeze_backbone_layers
from src.training.callbacks import build_callbacks
from utils.file_utils import save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 25,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    csv_log_path: Optional[Union[str, Path]] = None,
    patience: int = 7
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Executes a standard single-stage model training loop with full metrics recording.
    """
    if callbacks is None:
        callbacks = build_callbacks(
            checkpoint_path=checkpoint_path,
            csv_log_path=csv_log_path,
            patience_early_stopping=patience
        )

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_time_sec = round(time.time() - t0, 2)

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_val_acc = max(hist_dict.get("val_accuracy", [0.0]))
    best_val_loss = min(hist_dict.get("val_loss", [float("inf")]))
    best_epoch = int(np.argmin(hist_dict.get("val_loss", [0])) + 1)

    train_summary = {
        "training_time_sec": training_time_sec,
        "epochs_trained": len(hist_dict.get("loss", [])),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(float(best_val_acc), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "history": hist_dict,
    }

    logger.info(f"Training completed in {training_time_sec}s. Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")
    return model, train_summary


def run_two_stage_transfer_training(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    stage1_epochs: int = 10,
    stage2_epochs: int = 15,
    num_unfreeze_layers: int = 25,
    fine_tune_lr: float = 0.00005,
    checkpoint_path: Optional[Union[str, Path]] = None,
    csv_log_path: Optional[Union[str, Path]] = None
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Orchestrates two-stage transfer learning:
    Stage 1: Frozen backbone (train classifier head only)
    Stage 2: Unfrozen upper layers (fine-tune with low learning rate)
    """
    logger.info("=== Starting Stage 1: Transfer Learning Feature Extraction ===")
    callbacks_s1 = build_callbacks(
        checkpoint_path=None,
        csv_log_path=csv_log_path,
        patience_early_stopping=6
    )

    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks_s1,
        verbose=1
    )

    logger.info("=== Starting Stage 2: Transfer Learning Fine-Tuning ===")
    model = unfreeze_backbone_layers(
        model,
        num_layers_to_unfreeze=num_unfreeze_layers,
        fine_tune_lr=fine_tune_lr
    )

    callbacks_s2 = build_callbacks(
        checkpoint_path=checkpoint_path,
        csv_log_path=csv_log_path,
        patience_early_stopping=8
    )

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs + stage2_epochs,
        initial_epoch=len(hist1.history["loss"]),
        callbacks=callbacks_s2,
        verbose=1
    )
    total_time_sec = round(time.time() - t0, 2)

    # Combine stage histories
    combined_history = {}
    for k in hist1.history.keys():
        combined_history[k] = [float(v) for v in hist1.history[k]] + [float(v) for v in hist2.history[k]]

    best_val_acc = max(combined_history.get("val_accuracy", [0.0]))
    best_val_loss = min(combined_history.get("val_loss", [float("inf")]))
    best_epoch = int(np.argmin(combined_history.get("val_loss", [0])) + 1)

    train_summary = {
        "training_time_sec": total_time_sec,
        "epochs_trained": len(combined_history.get("loss", [])),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(float(best_val_acc), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "history": combined_history,
        "stage1_epochs": len(hist1.history["loss"]),
        "stage2_epochs": len(hist2.history["loss"]),
    }

    return model, train_summary
