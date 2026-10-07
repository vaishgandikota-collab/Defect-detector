"""
Comprehensive evaluation, benchmark comparison, and champion model selection script.
Evaluates models strictly against the untouched test split, generating all publication-grade figures and reports.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data.loader import create_datasets
from src.evaluation.confusion_matrix import compute_and_plot_confusion_matrix
from src.evaluation.diagnostics import generate_deployment_recommendations
from src.evaluation.metrics import evaluate_model_performance
from src.models.baseline_cnn import build_baseline_cnn
from src.models.mobilenet_model import build_mobilenet_model
from src.models.model_utils import benchmark_inference_speed, compute_model_size_mb, count_parameters, save_model_artifacts
from src.models.transfer_learning import build_transfer_model
from src.training.train import train_model
from utils.file_utils import ensure_dir, load_yaml_config, save_json
from utils.logging_utils import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def generate_comparison_plots(df_comp: pd.DataFrame, output_dir: Path) -> None:
    """
    Generates multi-metric comparison bar charts with strict Light Theme styling.
    """
    ensure_dir(output_dir)

    # 1. Accuracy vs Latency Trade-off
    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor="#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    ax1.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")

    colors = ["#2563EB", "#16A34A", "#D97706"]
    bars = ax1.bar(df_comp["model_name"], df_comp["test_accuracy"] * 100, color=colors, width=0.45, edgecolor="#0F172A", linewidth=1.2)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1E293B")

    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="semibold", color="#1E293B")
    ax1.set_title("Defect Detection Accuracy across Architectures", fontsize=13, fontweight="bold", pad=14, color="#0F172A")
    plt.xticks(fontsize=11, fontweight="semibold", color="#334155")
    plt.tight_layout()
    fig.savefig(output_dir / "model_accuracy_comparison.png", dpi=300, facecolor="#FFFFFF")
    plt.close(fig)

    # 2. CPU Inference Latency Comparison
    fig, ax2 = plt.subplots(figsize=(8, 5), facecolor="#FFFFFF")
    ax2.set_facecolor("#FFFFFF")
    ax2.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")

    bars2 = ax2.barh(df_comp["model_name"], df_comp["avg_latency_ms"], color=["#3B82F6", "#10B981", "#F59E0B"], height=0.45, edgecolor="#0F172A", linewidth=1.2)

    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.5, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f} ms", ha="left", va="center", fontsize=11, fontweight="bold", color="#1E293B")

    ax2.set_xlabel("Average CPU Inference Latency (ms)", fontsize=12, fontweight="semibold", color="#1E293B")
    ax2.set_title("Inference Speed Benchmark (Lower is Faster)", fontsize=13, fontweight="bold", pad=14, color="#0F172A")
    plt.yticks(fontsize=11, fontweight="semibold", color="#334155")
    plt.tight_layout()
    fig.savefig(output_dir / "model_latency_comparison.png", dpi=300, facecolor="#FFFFFF")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Models & Generate Reports")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    img_size = tuple(cfg["dataset"]["image_size"])
    figures_dir = ensure_dir(Path(cfg["paths"]["figures_dir"]))

    # 1. Load Data
    ds_train, ds_val, ds_test, meta = create_datasets(
        train_dir=cfg["dataset"]["train_dir"],
        val_dir=cfg["dataset"].get("val_dir"),
        test_dir=cfg["dataset"].get("test_dir"),
        image_size=img_size,
        batch_size=cfg["training"]["batch_size"],
        use_augmentation=True,
        class_names_output_path=cfg["paths"]["class_names_path"]
    )

    models_info = []

    # --- 1. Baseline Model ---
    baseline_path = Path("models/baseline/baseline_model.keras")
    if not baseline_path.exists():
        logger.info("Training baseline model for evaluation...")
        baseline_m = build_baseline_cnn(input_shape=(img_size[0], img_size[1], 3), num_classes=meta["num_classes"])
        baseline_m, _ = train_model(baseline_m, ds_train, ds_val, epochs=8, checkpoint_path=baseline_path)
    else:
        baseline_m = keras.models.load_model(str(baseline_path), compile=False)

    eval_base = evaluate_model_performance(baseline_m, ds_test, class_names=meta["classes"])
    lat_base = benchmark_inference_speed(baseline_m, input_shape=(img_size[0], img_size[1], 3))
    params_base = count_parameters(baseline_m)
    models_info.append({
        "model_name": "Baseline CNN",
        "model_path": str(baseline_path),
        "test_accuracy": eval_base["accuracy"],
        "macro_f1": eval_base["macro_f1"],
        "macro_precision": eval_base["macro_precision"],
        "macro_recall": eval_base["macro_recall"],
        "avg_latency_ms": lat_base["avg_latency_ms"],
        "throughput_fps": lat_base["throughput_fps"],
        "total_parameters": params_base["total_parameters"],
        "model_size_mb": compute_model_size_mb(baseline_path),
        "eval_details": eval_base
    })

    # --- 2. MobileNet Model (Lightweight Deployment) ---
    mobilenet_path = Path("models/deployment/mobilenet_model.keras")
    if not mobilenet_path.exists():
        logger.info("Training MobileNetV3 deployment model for evaluation...")
        mobilenet_m = build_mobilenet_model(input_shape=(img_size[0], img_size[1], 3), num_classes=meta["num_classes"])
        mobilenet_m, _ = train_model(mobilenet_m, ds_train, ds_val, epochs=8, checkpoint_path=mobilenet_path)
    else:
        mobilenet_m = keras.models.load_model(str(mobilenet_path), compile=False)

    eval_mobile = evaluate_model_performance(mobilenet_m, ds_test, class_names=meta["classes"])
    lat_mobile = benchmark_inference_speed(mobilenet_m, input_shape=(img_size[0], img_size[1], 3))
    params_mobile = count_parameters(mobilenet_m)
    models_info.append({
        "model_name": "MobileNetV3 (Edge)",
        "model_path": str(mobilenet_path),
        "test_accuracy": eval_mobile["accuracy"],
        "macro_f1": eval_mobile["macro_f1"],
        "macro_precision": eval_mobile["macro_precision"],
        "macro_recall": eval_mobile["macro_recall"],
        "avg_latency_ms": lat_mobile["avg_latency_ms"],
        "throughput_fps": lat_mobile["throughput_fps"],
        "total_parameters": params_mobile["total_parameters"],
        "model_size_mb": compute_model_size_mb(mobilenet_path),
        "eval_details": eval_mobile
    })

    # --- 3. Transfer Learning Model ---
    transfer_path = Path("models/transfer/transfer_model.keras")
    if not transfer_path.exists():
        logger.info("Training Transfer Learning model for evaluation...")
        transfer_m = build_transfer_model(input_shape=(img_size[0], img_size[1], 3), num_classes=meta["num_classes"])
        transfer_m, _ = train_model(transfer_m, ds_train, ds_val, epochs=8, checkpoint_path=transfer_path)
    else:
        transfer_m = keras.models.load_model(str(transfer_path), compile=False)

    eval_trans = evaluate_model_performance(transfer_m, ds_test, class_names=meta["classes"])
    lat_trans = benchmark_inference_speed(transfer_m, input_shape=(img_size[0], img_size[1], 3))
    params_trans = count_parameters(transfer_m)
    models_info.append({
        "model_name": "ResNet50V2 (Transfer)",
        "model_path": str(transfer_path),
        "test_accuracy": eval_trans["accuracy"],
        "macro_f1": eval_trans["macro_f1"],
        "macro_precision": eval_trans["macro_precision"],
        "macro_recall": eval_trans["macro_recall"],
        "avg_latency_ms": lat_trans["avg_latency_ms"],
        "throughput_fps": lat_trans["throughput_fps"],
        "total_parameters": params_trans["total_parameters"],
        "model_size_mb": compute_model_size_mb(transfer_path),
        "eval_details": eval_trans
    })

    # Build Comparison DataFrame
    df_comparison = pd.DataFrame([
        {k: v for k, v in m.items() if k != "eval_details"}
        for m in models_info
    ])
    df_comparison.to_csv(cfg["paths"]["experiment_results"], index=False)
    logger.info(f"Saved experiment comparison results to: {cfg['paths']['experiment_results']}")

    # Champion Model Selection
    champion_idx = int(df_comparison["test_accuracy"].idxmax())
    champion_info = models_info[champion_idx]
    logger.info(f"Champion Model: {champion_info['model_name']} with Test Accuracy: {champion_info['test_accuracy']:.4f}")

    # Copy Champion to best_model.keras
    shutil.copy(champion_info["model_path"], cfg["paths"]["best_model_path"])
    logger.info(f"Deployed champion to: {cfg['paths']['best_model_path']}")

    # Generate Confusion Matrix for Champion
    cm_plot_path = figures_dir / "confusion_matrix.png"
    compute_and_plot_confusion_matrix(
        y_true=champion_info["eval_details"]["y_true"],
        y_pred=champion_info["eval_details"]["y_pred"],
        class_names=meta["classes"],
        output_png_path=cm_plot_path,
        title=f"Confusion Matrix — {champion_info['model_name']}"
    )

    # Classification Report CSV
    clf_report_path = cfg["paths"]["classification_report"]
    df_clf = pd.DataFrame(champion_info["eval_details"]["per_class"]).transpose()
    df_clf.to_csv(clf_report_path, index=True)

    # Generate Comparison Figures
    generate_comparison_plots(df_comparison, figures_dir)

    # Save Model Metadata
    save_json({
        "champion_model": champion_info["model_name"],
        "test_accuracy": champion_info["test_accuracy"],
        "macro_f1": champion_info["macro_f1"],
        "latency_ms": champion_info["avg_latency_ms"],
        "classes": meta["classes"],
        "num_classes": meta["num_classes"],
        "image_size": list(img_size)
    }, cfg["paths"]["metadata_path"])

    # Deployment Recommendations
    recs = generate_deployment_recommendations(models_info)
    save_json(recs, Path("reports") / "deployment_recommendations.json")

    logger.info("=== Full Pipeline Evaluation & Artifact Generation Complete ===")


if __name__ == "__main__":
    main()
