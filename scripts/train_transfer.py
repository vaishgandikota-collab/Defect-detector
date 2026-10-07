"""
Training script for the Transfer Learning Model (2-Stage Training).
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import create_datasets
from src.evaluation.diagnostics import plot_training_curves
from src.evaluation.metrics import evaluate_model_performance
from src.models.model_utils import benchmark_inference_speed, save_model_artifacts
from src.models.transfer_learning import build_transfer_model
from src.training.train import run_two_stage_transfer_training
from utils.file_utils import ensure_dir, load_yaml_config
from utils.logging_utils import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Transfer Learning CNN")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    train_dir = cfg["dataset"]["train_dir"]
    val_dir = cfg["dataset"].get("val_dir")
    test_dir = cfg["dataset"].get("test_dir")
    img_size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["training"]["batch_size"]

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

    # 2. Build Transfer Model
    t_cfg = cfg["models"]["transfer"]
    model = build_transfer_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=meta["num_classes"],
        backbone_name=t_cfg.get("backbone", "ResNet50V2"),
        weights=t_cfg.get("weights", "imagenet"),
        dense_units=t_cfg.get("dense_units", 256),
        dropout_rate=t_cfg.get("dropout_rate", 0.4),
        learning_rate=cfg["training"]["learning_rate"],
        optimizer_name=cfg["training"]["optimizer"],
        model_name="transfer_resnet50v2"
    )

    # 3. Two-Stage Training
    model_save_path = Path(cfg["paths"]["models_dir"]) / "transfer" / "transfer_model.keras"
    csv_log_path = Path("experiments") / "model_comparison" / "transfer_training_log.csv"

    model, train_summary = run_two_stage_transfer_training(
        model=model,
        train_ds=ds_train,
        val_ds=ds_val,
        stage1_epochs=8,
        stage2_epochs=12,
        num_unfreeze_layers=t_cfg.get("fine_tune_layers", 25),
        fine_tune_lr=t_cfg.get("fine_tune_lr", 0.00005),
        checkpoint_path=model_save_path,
        csv_log_path=csv_log_path
    )

    # 4. Evaluate on untouched test set
    test_results = evaluate_model_performance(model, ds_test, class_names=meta["classes"])
    latency_bench = benchmark_inference_speed(model, input_shape=(img_size[0], img_size[1], 3))

    logger.info(f"Transfer Test Accuracy: {test_results['accuracy']:.4f} | Macro F1: {test_results['macro_f1']:.4f} | Latency: {latency_bench['avg_latency_ms']} ms")

    # 5. Save Artifacts & Figures
    figures_dir = ensure_dir(cfg["paths"]["figures_dir"])
    plot_training_curves(
        train_summary["history"],
        output_png_path=figures_dir / "transfer_training_curves.png",
        title_prefix="Transfer Learning (ResNet50V2)"
    )

    save_model_artifacts(
        model=model,
        save_path=model_save_path,
        class_names=meta["classes"],
        metadata_path=Path(cfg["paths"]["models_dir"]) / "transfer" / "transfer_metadata.json",
        extra_metadata={
            "test_metrics": test_results,
            "latency": latency_bench,
            "training_summary": train_summary
        }
    )


if __name__ == "__main__":
    main()
