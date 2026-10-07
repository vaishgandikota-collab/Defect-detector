"""
Hyperparameter optimization engine using Optuna for architectural and optimization tuning.
Evaluates candidates strictly on the validation set without test-set data leakage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.models.baseline_cnn import build_baseline_cnn
from src.training.callbacks import build_callbacks
from utils.file_utils import ensure_dir, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_hyperparameter_tuning(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    n_trials: int = 10,
    epochs_per_trial: int = 8,
    output_dir: Union[str, Path] = "experiments/hyperparameter"
) -> Dict[str, Any]:
    """
    Executes automated Bayesian / TPE hyperparameter search over learning rate,
    dropout, dense capacity, L2 regularization, and optimizer family.
    """
    out_path = Path(output_dir)
    ensure_dir(out_path)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed. Executing grid-fallback search.")
        return _run_grid_fallback(train_ds, val_ds, num_classes, image_size, epochs_per_trial, out_path)

    trial_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
        l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])

        model = build_baseline_cnn(
            input_shape=(image_size[0], image_size[1], 3),
            num_classes=num_classes,
            dense_units=dense_units,
            dropout_rate=dropout,
            l2_reg=l2_reg,
            learning_rate=lr,
            optimizer_name=optimizer_name
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_per_trial,
            callbacks=callbacks,
            verbose=0
        )

        val_accs = history.history.get("val_accuracy", [0.0])
        val_losses = history.history.get("val_loss", [float("inf")])
        best_acc = float(max(val_accs))
        best_loss = float(min(val_losses))

        trial_records.append({
            "trial_number": trial.number,
            "learning_rate": lr,
            "dropout_rate": dropout,
            "dense_units": dense_units,
            "l2_reg": l2_reg,
            "optimizer": optimizer_name,
            "best_val_accuracy": round(best_acc, 4),
            "best_val_loss": round(best_loss, 4),
            "epochs_completed": len(val_accs)
        })

        return best_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    df_results = pd.DataFrame(trial_records)
    csv_file = out_path / "tuning_results.csv"
    df_results.to_csv(csv_file, index=False)

    best_config = {
        "best_trial_number": study.best_trial.number,
        "best_value": round(float(study.best_value), 4),
        "best_params": study.best_params,
        "total_trials": len(trial_records)
    }

    save_json(best_config, out_path / "best_hyperparameters.json")
    logger.info(f"Tuning finished. Best Val Accuracy: {study.best_value:.4f} with params: {study.best_params}")

    return {
        "best_config": best_config,
        "results_table": df_results.to_dict(orient="records")
    }


def _run_grid_fallback(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    image_size: Tuple[int, int],
    epochs: int,
    out_path: Path
) -> Dict[str, Any]:
    """Deterministic fallback grid when optuna is absent."""
    candidate_lrs = [0.01, 0.001, 0.0001]
    candidate_opts = ["adam", "sgd", "rmsprop"]
    records = []

    trial_idx = 0
    for lr in candidate_lrs:
        for opt in candidate_opts:
            trial_idx += 1
            model = build_baseline_cnn(
                input_shape=(image_size[0], image_size[1], 3),
                num_classes=num_classes,
                learning_rate=lr,
                optimizer_name=opt
            )
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
                verbose=0
            )
            acc = float(max(history.history.get("val_accuracy", [0.0])))
            loss = float(min(history.history.get("val_loss", [float("inf")])))
            records.append({
                "trial_number": trial_idx,
                "learning_rate": lr,
                "optimizer": opt,
                "best_val_accuracy": round(acc, 4),
                "best_val_loss": round(loss, 4),
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path / "tuning_results.csv", index=False)
    best_row = df.loc[df["best_val_accuracy"].idxmax()].to_dict()
    save_json(best_row, out_path / "best_hyperparameters.json")

    return {
        "best_config": best_row,
        "results_table": records
    }
