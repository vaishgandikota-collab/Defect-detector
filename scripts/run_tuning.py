"""
Hyperparameter tuning runner script using Optuna/KerasTuner.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import create_datasets
from src.training.tuning import run_hyperparameter_tuning
from utils.file_utils import load_yaml_config
from utils.logging_utils import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--trials", type=int, default=10, help="Number of tuning trials")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per trial")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    # 1. Load Data
    ds_train, ds_val, ds_test, meta = create_datasets(
        train_dir=cfg["dataset"]["train_dir"],
        val_dir=cfg["dataset"].get("val_dir"),
        test_dir=cfg["dataset"].get("test_dir"),
        image_size=tuple(cfg["dataset"]["image_size"]),
        batch_size=cfg["training"]["batch_size"],
        use_augmentation=True
    )

    logger.info("=== Starting Hyperparameter Optimization ===")
    results = run_hyperparameter_tuning(
        train_ds=ds_train,
        val_ds=ds_val,
        num_classes=meta["num_classes"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        n_trials=args.trials,
        epochs_per_trial=args.epochs,
        output_dir="experiments/hyperparameter"
    )

    logger.info("=== Hyperparameter Optimization Completed ===")
    logger.info(f"Best Config: {results['best_config']}")


if __name__ == "__main__":
    main()
