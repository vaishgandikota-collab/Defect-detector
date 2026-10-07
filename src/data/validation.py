"""
Dataset validation and automated health inspection module.
Checks integrity, format conformity, duplicates, dimensional anomalies, and class imbalance.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.file_utils import ensure_dir, save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_IMAGE_DIM = 32


class DatasetValidator:
    """
    Performs comprehensive dataset health and integrity checks across raw/split data folders.
    """

    def __init__(self, data_root: Union[str, Path], min_dimension: int = MIN_IMAGE_DIM):
        self.data_root = Path(data_root)
        self.min_dimension = min_dimension
        self.hashes: Set[str] = set()

    def _compute_hash(self, file_path: Path) -> str:
        """Computes MD5 hash for duplicate detection."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def inspect_directory(self, target_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Scans a directory containing class subfolders and validates every image file.
        """
        scan_dir = Path(target_dir) if target_dir else self.data_root
        if not scan_dir.exists():
            return {
                "exists": False,
                "error": f"Directory not found: {scan_dir.resolve()}",
                "total_images": 0,
                "classes": [],
                "class_counts": {},
            }

        sub_splits = [d for d in scan_dir.iterdir() if d.is_dir() and d.name in {"train", "validation", "test"}]
        if sub_splits and target_dir is None:
            # Aggregate across splits
            all_classes: Set[str] = set()
            total_valid = 0
            class_counts: Dict[str, int] = {}
            corrupted_files: List[str] = []
            unsupported_files: List[str] = []
            duplicate_files: List[str] = []
            undersized_files: List[str] = []
            image_records: List[Dict[str, Any]] = []

            for split in sub_splits:
                split_report = self.inspect_directory(split)
                if split_report.get("exists"):
                    for c in split_report.get("classes", []):
                        all_classes.add(c)
                        class_counts[c] = class_counts.get(c, 0) + split_report.get("class_counts", {}).get(c, 0)
                    total_valid += split_report.get("total_images", 0)
                    corrupted_files.extend(split_report.get("corrupted_files", []))
                    unsupported_files.extend(split_report.get("unsupported_files", []))
                    duplicate_files.extend(split_report.get("duplicate_files", []))
                    undersized_files.extend(split_report.get("undersized_files", []))
                    if "records" in split_report:
                        for r in split_report["records"]:
                            r["split"] = split.name
                        image_records.extend(split_report["records"])

            sorted_classes = sorted(list(all_classes))
            imb_ratio = 1.0
            if class_counts and min(class_counts.values()) > 0:
                imb_ratio = max(class_counts.values()) / min(class_counts.values())

            return {
                "exists": True,
                "directory": str(scan_dir),
                "total_images": total_valid,
                "num_classes": len(sorted_classes),
                "classes": sorted_classes,
                "class_counts": class_counts,
                "imbalance_ratio": round(imb_ratio, 2),
                "is_imbalanced": imb_ratio > 3.0,
                "corrupted_count": len(corrupted_files),
                "corrupted_files": corrupted_files[:20],
                "unsupported_count": len(unsupported_files),
                "unsupported_files": unsupported_files[:20],
                "duplicate_count": len(duplicate_files),
                "duplicate_files": duplicate_files[:20],
                "undersized_count": len(undersized_files),
                "undersized_files": undersized_files[:20],
                "is_healthy": (
                    len(corrupted_files) == 0 and
                    total_valid > 0 and
                    len(sorted_classes) >= 2
                ),
                "records": image_records,
            }

        class_subdirs = [d for d in scan_dir.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name not in {"raw", "processed"}]
        class_names = sorted([d.name for d in class_subdirs])
        total_valid = 0
        corrupted_files: List[str] = []
        unsupported_files: List[str] = []
        duplicate_files: List[str] = []
        undersized_files: List[str] = []
        class_counts: Dict[str, int] = {}
        image_records: List[Dict[str, Any]] = []

        for class_dir in class_subdirs:
            class_name = class_dir.name
            class_counts[class_name] = 0
            
            for file_path in class_dir.iterdir():
                if file_path.is_dir():
                    continue

                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    unsupported_files.append(str(file_path))
                    continue

                # Check duplication
                file_hash = self._compute_hash(file_path)
                if file_hash in self.hashes:
                    duplicate_files.append(str(file_path))
                else:
                    self.hashes.add(file_hash)

                # Check corruption & dimensions
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Fast structural verification
                    
                    # Reopen to read dimensions & channel depth
                    with Image.open(file_path) as img:
                        w, h = img.size
                        mode = img.mode

                    if w < self.min_dimension or h < self.min_dimension:
                        undersized_files.append(str(file_path))
                        continue

                    total_valid += 1
                    class_counts[class_name] += 1
                    image_records.append({
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "class_name": class_name,
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / max(1, h), 4),
                        "mode": mode,
                        "file_size_kb": round(file_path.stat().st_size / 1024, 2)
                    })

                except Exception as exc:
                    logger.warning(f"Corrupted or unreadable image: {file_path} - Reason: {exc}")
                    corrupted_files.append(str(file_path))

        # Check class imbalance
        imbalance_ratio = 1.0
        if class_counts and min(class_counts.values()) > 0:
            imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        
        is_imbalanced = imbalance_ratio > 3.0

        report = {
            "exists": True,
            "directory": str(scan_dir),
            "total_images": total_valid,
            "num_classes": len(class_names),
            "classes": class_names,
            "class_counts": class_counts,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "is_imbalanced": is_imbalanced,
            "corrupted_count": len(corrupted_files),
            "corrupted_files": corrupted_files[:20],
            "unsupported_count": len(unsupported_files),
            "unsupported_files": unsupported_files[:20],
            "duplicate_count": len(duplicate_files),
            "duplicate_files": duplicate_files[:20],
            "undersized_count": len(undersized_files),
            "undersized_files": undersized_files[:20],
            "is_healthy": (
                len(corrupted_files) == 0 and
                total_valid > 0 and
                len(class_names) >= 2
            ),
            "records": image_records,
        }

        return report


def validate_dataset_directory(
    data_dir: Union[str, Path],
    output_report_path: Optional[Union[str, Path]] = None,
    output_summary_csv: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Executes dataset validation, exports JSON and CSV reports, and returns structured summary.
    """
    validator = DatasetValidator(data_dir)
    report = validator.inspect_directory()

    # Extract records for CSV summary if available
    records = []
    if "records" in report:
        records = report.pop("records")
    elif "splits" in report:
        for split_name, split_report in report["splits"].items():
            if isinstance(split_report, dict) and "records" in split_report:
                split_records = split_report.pop("records")
                for r in split_records:
                    r["split"] = split_name
                records.extend(split_records)

    if output_report_path:
        save_json(report, output_report_path)
        logger.info(f"Dataset validation report saved to: {output_report_path}")

    if output_summary_csv and records:
        df = pd.DataFrame(records)
        df_path = Path(output_summary_csv)
        df_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(df_path, index=False)
        logger.info(f"Dataset summary CSV saved to: {output_summary_csv}")

    return report
