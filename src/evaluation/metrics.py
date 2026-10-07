"""
Multi-class statistical evaluation metrics and classification reporting.
Computes accuracy, macro/weighted/per-class precision, recall, F1, and OvR ROC/PR AUC.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score
)
from utils.file_utils import ensure_dir, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_model_performance(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: Optional[List[str]] = None,
    output_report_csv: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Computes rigorous classification performance metrics on an evaluation dataset.
    """
    y_true_list = []
    y_pred_probs_list = []

    for x_batch, y_batch in test_ds:
        preds = model(x_batch, training=False).numpy()
        y_pred_probs_list.append(preds)
        
        # Check if y_batch is one-hot or integer labels
        y_true_np = y_batch.numpy()
        if y_true_np.ndim > 1 and y_true_np.shape[-1] > 1:
            y_true_list.append(np.argmax(y_true_np, axis=-1))
        else:
            y_true_list.append(y_true_np.ravel())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    num_classes = y_pred_probs.shape[-1]
    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class_metrics = {}
    for i, c_name in enumerate(class_names):
        per_class_metrics[c_name] = {
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1": round(float(f1s[i]), 4),
            "support": int(supports[i]),
        }

    # Multi-class One-vs-Rest ROC-AUC
    roc_auc = None
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(
                y_true,
                y_pred_probs,
                multi_class="ovr",
                average="macro"
            ))
    except Exception as exc:
        logger.debug(f"Multi-class ROC-AUC calculation skipped: {exc}")

    results = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_ovr": round(roc_auc, 4) if roc_auc is not None else None,
        "per_class": per_class_metrics,
        "total_evaluated_samples": len(y_true),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_probs": y_pred_probs.tolist(),
    }

    if output_report_csv:
        generate_classification_report(y_true, y_pred, class_names, output_report_csv)

    return results


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_csv_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Exports a tabular classification report (Precision, Recall, F1, Support) to CSV.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()
    df = df.round(4)
    
    out_path = Path(output_csv_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=True)
    logger.info(f"Saved classification report CSV to: {out_path}")
    return df
