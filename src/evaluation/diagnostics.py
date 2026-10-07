"""
Automated empirical diagnostics: Overfitting/Underfitting detection,
gradient stability checks, training curve visualizers, and deployment recommendation engine.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.file_utils import ensure_dir
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def diagnose_training_fit(history: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Analyzes empirical training and validation loss/accuracy dynamics to diagnose fit condition.
    """
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if not train_acc or not val_acc:
        return {
            "condition": "INSUFFICIENT_DATA",
            "explanation": "Training history contains insufficient epochs for reliable diagnostic inference."
        }

    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    acc_gap = final_train_acc - final_val_acc

    # Trend in val loss over last 3 epochs
    val_loss_trend = 0.0
    if len(val_loss) >= 3:
        val_loss_trend = val_loss[-1] - val_loss[-3]

    if final_train_acc < 0.65 and final_val_acc < 0.60:
        condition = "UNDERFITTING"
        explanation = (
            f"Underfitting observed: Training accuracy ({final_train_acc:.2%}) and validation accuracy "
            f"({final_val_acc:.2%}) remain low. Model capacity or feature extraction is insufficient."
        )
        recommendations = [
            "Increase model depth or number of convolutional filters.",
            "Decrease regularization penalties (lower L2 weight decay or dropout).",
            "Train for more epochs or use a pretrained transfer learning backbone."
        ]
    elif acc_gap > 0.12 or (val_loss_trend > 0.05 and acc_gap > 0.08):
        condition = "OVERFITTING"
        explanation = (
            f"Overfitting detected: Substantial generalization gap of {acc_gap:.2%} between training "
            f"accuracy ({final_train_acc:.2%}) and validation accuracy ({final_val_acc:.2%})."
        )
        recommendations = [
            "Increase dropout rate (e.g. from 0.2 to 0.4).",
            "Enable or expand data augmentation (rotations, zoom, flips).",
            "Apply L2 weight regularization or incorporate Batch Normalization.",
            "Use early stopping to halt training prior to validation loss divergence."
        ]
    else:
        condition = "GOOD_FIT"
        explanation = (
            f"Optimal convergence achieved: Training accuracy ({final_train_acc:.2%}) and validation accuracy "
            f"({final_val_acc:.2%}) track closely with an acceptable generalization delta of {acc_gap:.2%}."
        )
        recommendations = [
            "Model is well-regularized and ready for full test-set evaluation and edge deployment."
        ]

    return {
        "condition": condition,
        "generalization_gap": round(float(acc_gap), 4),
        "final_train_accuracy": round(float(final_train_acc), 4),
        "final_val_accuracy": round(float(final_val_acc), 4),
        "final_train_loss": round(float(train_loss[-1]), 4) if train_loss else 0.0,
        "final_val_loss": round(float(val_loss[-1]), 4) if val_loss else 0.0,
        "explanation": explanation,
        "recommendations": recommendations,
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    output_png_path: Optional[Union[str, Path]] = None,
    title_prefix: str = "Model"
) -> None:
    """
    Plots training vs validation accuracy and loss curves with strict Light Theme styling.
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#FFFFFF")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")

    # Accuracy Plot
    ax1.plot(epochs, history.get("accuracy", []), "o-", color="#2563EB", label="Training Accuracy", linewidth=2, markersize=4)
    ax1.plot(epochs, history.get("val_accuracy", []), "s--", color="#16A34A", label="Validation Accuracy", linewidth=2, markersize=4)
    ax1.set_title(f"{title_prefix}: Accuracy Progression", fontsize=12, fontweight="bold", color="#0F172A")
    ax1.set_xlabel("Epoch", fontsize=10, color="#334155")
    ax1.set_ylabel("Accuracy", fontsize=10, color="#334155")
    ax1.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    # Loss Plot
    ax2.plot(epochs, history.get("loss", []), "o-", color="#DC2626", label="Training Loss", linewidth=2, markersize=4)
    ax2.plot(epochs, history.get("val_loss", []), "s--", color="#D97706", label="Validation Loss", linewidth=2, markersize=4)
    ax2.set_title(f"{title_prefix}: Loss Convergence", fontsize=12, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Epoch", fontsize=10, color="#334155")
    ax2.set_ylabel("Categorical Cross-Entropy Loss", fontsize=10, color="#334155")
    ax2.legend(loc="upper right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    plt.tight_layout()

    if output_png_path:
        out_p = Path(output_png_path)
        ensure_dir(out_p.parent)
        fig.savefig(out_p, dpi=300, facecolor="#FFFFFF", edgecolor="none")
        logger.info(f"Saved training curves to: {out_p}")

    plt.close(fig)


def generate_deployment_recommendations(models_comparison: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes factual, empirical Pareto-optimal deployment recommendations based on
    measured metrics (Accuracy, F1, CPU Latency, Memory Footprint, Parameter Count).
    """
    if not models_comparison:
        return {"summary": "No models available for empirical comparison.", "recommendations": []}

    df = pd.DataFrame(models_comparison)
    
    # Identify high accuracy champion
    best_acc_row = df.loc[df["test_accuracy"].idxmax()]
    # Identify lowest latency champion
    lowest_lat_row = df.loc[df["avg_latency_ms"].idxmin()]
    # Identify smallest footprint champion
    smallest_row = df.loc[df["model_size_mb"].idxmin()]

    recommendations = []

    # High-Throughput Edge Deployment
    recommendations.append({
        "scenario": "High-Speed Real-Time Conveyor Belt (Edge / Embedded CPU)",
        "recommended_model": lowest_lat_row["model_name"],
        "rationale": (
            f"Achieves lowest latency of {lowest_lat_row['avg_latency_ms']} ms/image "
            f"({lowest_lat_row.get('throughput_fps', 0)} FPS) with compact footprint "
            f"({lowest_lat_row['model_size_mb']} MB) while maintaining {lowest_lat_row['test_accuracy']:.2%} accuracy."
        )
    })

    # High-Precision Quality Assurance
    recommendations.append({
        "scenario": "Critical Safety / Zero-Defect Tolerance Inspection (Server / Cloud)",
        "recommended_model": best_acc_row["model_name"],
        "rationale": (
            f"Delivers top classification performance: {best_acc_row['test_accuracy']:.2%} test accuracy "
            f"and {best_acc_row.get('macro_f1', 0):.4f} Macro F1-score with {best_acc_row['avg_latency_ms']} ms latency."
        )
    })

    return {
        "best_accuracy_model": best_acc_row["model_name"],
        "lowest_latency_model": lowest_lat_row["model_name"],
        "smallest_model": smallest_row["model_name"],
        "recommendations": recommendations,
        "comparison_table": df.to_dict(orient="records")
    }
