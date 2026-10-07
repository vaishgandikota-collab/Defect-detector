"""
Training script for the First-Principles Baseline CNN Model.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import create_datasets
from src.evaluation.diagnostics import plot_training_curves
from src.evaluation.metrics import evaluate_model_performance
from src.models.baseline_cnn import build_baseline_cnn
from src.models.model_utils import benchmark_inference_speed, save_model_artifacts
from src.training.train import train_model
from utils.file_utils import ensure_dir, load_yaml_config
from utils.logging_utils import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline CNN")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    train_dir = cfg["dataset"]["train_dir"]
    val_dir = cfg["dataset"].get("val_dir")
    test_dir = cfg["dataset"].get("test_dir")
    img_size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["training"]["batch_size"]
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]

    # 1. Load Data
    ds_train, ds_val, ds_test, meta = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=img_size,
        batch_size=batch_size,
        use_augmentation=cfg["augmentation"]["enabled"],
        aug_config=cfg["augmentation"],
        class_names_output_path=cfg["paths"]["class_names_path"]
    )

    # 2. Build Model
    m_cfg = cfg["models"]["baseline"]
    model = build_baseline_cnn(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=meta["num_classes"],
        conv_filters=m_cfg.get("conv_filters", [32, 64, 128]),
        dense_units=m_cfg.get("dense_units", 128),
        dropout_rate=m_cfg.get("dropout_rate", 0.3),
        use_batch_norm=m_cfg.get("use_batch_norm", True),
        l2_reg=m_cfg.get("l2_reg", 0.0001),
        learning_rate=cfg["training"]["learning_rate"],
        optimizer_name=cfg["training"]["optimizer"],
        model_name="baseline_cnn"
    )

    # 3. Train
    model_save_path = Path(cfg["paths"]["models_dir"]) / "baseline" / "baseline_model.keras"
    csv_log_path = Path("experiments") / "model_comparison" / "baseline_training_log.csv"

    model, train_summary = train_model(
        model=model,
        train_ds=ds_train,
        val_ds=ds_val,
        epochs=epochs,
        checkpoint_path=model_save_path,
        csv_log_path=csv_log_path,
        patience=cfg["training"]["early_stopping"]["patience"]
    )

    # 4. Evaluate on untouched test set
    test_results = evaluate_model_performance(model, ds_test, class_names=meta["classes"])
    latency_bench = benchmark_inference_speed(model, input_shape=(img_size[0], img_size[1], 3))

    logger.info(f"Baseline Test Accuracy: {test_results['accuracy']:.4f} | Macro F1: {test_results['macro_f1']:.4f} | Latency: {latency_bench['avg_latency_ms']} ms")

    # 5. Save Artifacts & Figures
    figures_dir = ensure_dir(cfg["paths"]["figures_dir"])
    plot_training_curves(
        train_summary["history"],
        output_png_path=figures_dir / "baseline_training_curves.png",
        title_prefix="Baseline CNN"
    )

    save_model_artifacts(
        model=model,
        save_path=model_save_path,
        class_names=meta["classes"],
        metadata_path=Path(cfg["paths"]["models_dir"]) / "baseline" / "baseline_metadata.json",
        extra_metadata={
            "test_metrics": test_results,
            "latency": latency_bench,
            "training_summary": train_summary
        }
    )


if __name__ == "__main__":
    main()
