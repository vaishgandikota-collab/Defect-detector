"""
Runner script for the 6-experiment Regularization Ablation Study.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.experiments import run_ablation_study
from utils.file_utils import load_yaml_config
from utils.logging_utils import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Regularization Ablation Study")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per ablation experiment")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["training"].get("seed", 42))

    logger.info("=== Starting Formal Regularization Ablation Study ===")
    df_ablation = run_ablation_study(
        train_dir=cfg["dataset"]["train_dir"],
        val_dir=cfg["dataset"].get("val_dir"),
        test_dir=cfg["dataset"].get("test_dir"),
        image_size=tuple(cfg["dataset"]["image_size"]),
        batch_size=cfg["training"]["batch_size"],
        epochs=args.epochs,
        output_dir="experiments/ablation"
    )

    logger.info("=== Ablation Study Completed Successfully ===")
    print(df_ablation.to_string(index=False))


if __name__ == "__main__":
    main()
