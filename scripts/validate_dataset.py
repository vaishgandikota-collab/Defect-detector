"""
Automated dataset validation script. Scans dataset health, detects duplicates, formats,
and class distributions, exporting reports/dataset_report.json and reports/dataset_summary.csv.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validation import validate_dataset_directory
from utils.file_utils import load_yaml_config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate Dataset Quality & Integrity")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_dir = Path("data")
    report_json = Path(cfg["paths"]["dataset_report"])
    summary_csv = Path(cfg["paths"]["reports_dir"]) / "dataset_summary.csv"

    logger.info("Executing automated dataset health and validation audit...")
    report = validate_dataset_directory(
        data_dir=data_dir,
        output_report_path=report_json,
        output_summary_csv=summary_csv
    )

    logger.info("=== Dataset Health Audit Complete ===")
    logger.info(f"Is Dataset Healthy: {report.get('is_healthy', True)}")
    logger.info(f"Report JSON: {report_json.resolve()}")
    logger.info(f"Summary CSV: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
