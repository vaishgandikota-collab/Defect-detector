"""
Experimental suite orchestration: Regularization ablation studies and learning-rate grid benchmarks.
Executes real empirical evaluations and records comprehensive diagnostic metrics.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data.loader import create_datasets
from src.evaluation.metrics import evaluate_model_performance
from src.models.baseline_cnn import build_baseline_cnn
from src.models.model_utils import count_parameters
from utils.file_utils import ensure_dir, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_ablation_study(
    train_dir: Union[str, Path],
    val_dir: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 12,
    output_dir: Union[str, Path] = "experiments/ablation"
) -> pd.DataFrame:
    """
    Executes the formal 6-experiment Regularization Ablation Study:
    - Exp A: No Regularization
    - Exp B: Dropout Only
    - Exp C: BatchNorm Only
    - Exp D: L2 Weight Decay Only
    - Exp E: Data Augmentation Only
    - Exp F: Combined Regularization (All)
    """
    out_path = ensure_dir(output_dir)

    ablation_specs = [
        {"name": "A_No_Regularization", "dropout": 0.0, "batch_norm": False, "l2": 0.0, "aug": False},
        {"name": "B_Dropout_Only", "dropout": 0.4, "batch_norm": False, "l2": 0.0, "aug": False},
        {"name": "C_BatchNorm_Only", "dropout": 0.0, "batch_norm": True, "l2": 0.0, "aug": False},
        {"name": "D_L2_Only", "dropout": 0.0, "batch_norm": False, "l2": 0.001, "aug": False},
        {"name": "E_Augmentation_Only", "dropout": 0.0, "batch_norm": False, "l2": 0.0, "aug": True},
        {"name": "F_Combined_All", "dropout": 0.3, "batch_norm": True, "l2": 0.0001, "aug": True},
    ]

    records: List[Dict[str, Any]] = []

    for spec in ablation_specs:
        exp_name = spec["name"]
        logger.info(f"--- Running Ablation Experiment: {exp_name} ---")

        # Create dataset for this experiment spec (enabling or disabling augmentation)
        ds_train, ds_val, ds_test, meta = create_datasets(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            image_size=image_size,
            batch_size=batch_size,
            use_augmentation=spec["aug"],
            seed=42
        )

        model = build_baseline_cnn(
            input_shape=(image_size[0], image_size[1], 3),
            num_classes=meta["num_classes"],
            dropout_rate=spec["dropout"],
            use_batch_norm=spec["batch_norm"],
            l2_reg=spec["l2"],
            model_name=exp_name
        )

        params = count_parameters(model)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]

        t0 = time.time()
        hist = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        train_time = round(time.time() - t0, 2)

        # Collect train/val metrics
        train_acc = float(hist.history["accuracy"][-1])
        val_acc = float(hist.history["val_accuracy"][-1])
        train_loss = float(hist.history["loss"][-1])
        val_loss = float(hist.history["val_loss"][-1])

        # Evaluate on untouched test set
        test_eval = evaluate_model_performance(model, ds_test, class_names=meta["classes"])

        record = {
            "Experiment": exp_name,
            "Dropout": "Yes" if spec["dropout"] > 0 else "No",
            "BatchNorm": "Yes" if spec["batch_norm"] else "No",
            "L2_Reg": "Yes" if spec["l2"] > 0 else "No",
            "Augmentation": "Yes" if spec["aug"] else "No",
            "Train_Accuracy": round(train_acc, 4),
            "Val_Accuracy": round(val_acc, 4),
            "Train_Loss": round(train_loss, 4),
            "Val_Loss": round(val_loss, 4),
            "Test_Accuracy": round(test_eval["accuracy"], 4),
            "Macro_Precision": round(test_eval["macro_precision"], 4),
            "Macro_Recall": round(test_eval["macro_recall"], 4),
            "Macro_F1": round(test_eval["macro_f1"], 4),
            "Training_Time_Sec": train_time,
            "Total_Parameters": params["total_parameters"],
        }
        records.append(record)

    df_ablation = pd.DataFrame(records)
    csv_file = out_path / "ablation_results.csv"
    df_ablation.to_csv(csv_file, index=False)
    logger.info(f"Ablation study saved to {csv_file}")

    return df_ablation


def run_learning_rate_experiments(
    train_dir: Union[str, Path],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 8,
    output_dir: Union[str, Path] = "experiments/hyperparameter"
) -> pd.DataFrame:
    """
    Evaluates learning rates (0.1, 0.01, 0.001, 0.0001) across Adam, SGD, and RMSprop.
    """
    out_path = ensure_dir(output_dir)
    ds_train, ds_val, ds_test, meta = create_datasets(
        train_dir=train_dir,
        image_size=image_size,
        batch_size=batch_size,
        use_augmentation=True
    )

    configs = [
        {"lr": 0.01, "opt": "adam"},
        {"lr": 0.001, "opt": "adam"},
        {"lr": 0.0001, "opt": "adam"},
        {"lr": 0.01, "opt": "sgd"},
        {"lr": 0.001, "opt": "sgd"},
        {"lr": 0.001, "opt": "rmsprop"},
    ]

    records = []
    for cfg in configs:
        lr = cfg["lr"]
        opt = cfg["opt"]
        logger.info(f"Testing Optimizer={opt}, LearningRate={lr}")

        model = build_baseline_cnn(
            input_shape=(image_size[0], image_size[1], 3),
            num_classes=meta["num_classes"],
            learning_rate=lr,
            optimizer_name=opt
        )

        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        t0 = time.time()
        hist = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, verbose=0)
        t_sec = round(time.time() - t0, 2)

        best_val_acc = float(max(hist.history.get("val_accuracy", [0.0])))
        best_val_loss = float(min(hist.history.get("val_loss", [float("inf")])))
        best_epoch = int(np.argmin(hist.history.get("val_loss", [0])) + 1)

        records.append({
            "Optimizer": opt.upper(),
            "Learning_Rate": lr,
            "Batch_Size": batch_size,
            "Best_Val_Accuracy": round(best_val_acc, 4),
            "Best_Val_Loss": round(best_val_loss, 4),
            "Best_Epoch": best_epoch,
            "Training_Duration_Sec": t_sec,
        })

    df = pd.DataFrame(records)
    csv_path = out_path / "lr_optimizer_experiments.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"LR optimizer experiments saved to {csv_path}")

    return df
