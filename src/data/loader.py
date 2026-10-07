"""
Dataset loader and tf.data pipeline generator.
Builds zero-leakage, high-throughput training, validation, and testing pipelines.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data.augmentation import build_augmentation_layers
from src.data.preprocessing import build_preprocess_fn
from utils.file_utils import save_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_classes(data_dir: Union[str, Path]) -> List[str]:
    """
    Scans a directory and identifies class subfolders, sorted alphabetically.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root.resolve()}")
    
    classes = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not classes:
        # Check if train subfolder exists
        train_dir = root / "train"
        if train_dir.exists():
            classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
            
    return classes


def gather_image_paths_and_labels(
    data_dir: Union[str, Path],
    classes: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[str]]:
    """
    Collects all valid image file paths and corresponding integer labels from class subfolders.
    """
    root = Path(data_dir)
    if classes is None:
        classes = discover_classes(root)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    file_paths: List[str] = []
    labels: List[int] = []

    for class_name in classes:
        cls_dir = root / class_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                file_paths.append(str(p))
                labels.append(class_to_idx[class_name])

    return file_paths, labels, classes


def create_datasets(
    train_dir: Union[str, Path],
    val_dir: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    use_augmentation: bool = True,
    aug_config: Optional[Dict[str, Any]] = None,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    class_names_output_path: Optional[Union[str, Path]] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """
    Builds training, validation, and test tf.data.Dataset streams with zero data leakage.

    If val_dir and test_dir are not provided, splits the train_dir data deterministically.
    """
    train_path = Path(train_dir)
    classes = discover_classes(train_path)
    
    if not classes:
        raise ValueError(f"No class folders found in {train_path.resolve()}")

    num_classes = len(classes)
    logger.info(f"Discovered {num_classes} defect classes: {classes}")

    if class_names_output_path:
        save_json(classes, class_names_output_path)

    # Check if explicit train, val, test folders exist
    val_path = Path(val_dir) if val_dir else None
    test_path = Path(test_dir) if test_dir else None

    if val_path and val_path.exists() and test_path and test_path.exists():
        train_files, train_labels, _ = gather_image_paths_and_labels(train_path, classes)
        val_files, val_labels, _ = gather_image_paths_and_labels(val_path, classes)
        test_files, test_labels, _ = gather_image_paths_and_labels(test_path, classes)
    else:
        # Perform deterministic split
        all_files, all_labels, _ = gather_image_paths_and_labels(train_path, classes)
        if len(all_files) == 0:
            raise ValueError(f"No image files found in {train_path.resolve()}")

        # Stratified train / val / test split
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            all_files,
            all_labels,
            test_size=test_split,
            random_state=seed,
            stratify=all_labels if len(set(all_labels)) > 1 and min(np.bincount(all_labels)) > 1 else None
        )

        relative_val_split = validation_split / (1.0 - test_split)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=relative_val_split,
            random_state=seed,
            stratify=train_val_labels if len(set(train_val_labels)) > 1 and min(np.bincount(train_val_labels)) > 1 else None
        )

    logger.info(f"Dataset split counts: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    preprocess_fn = build_preprocess_fn(target_size=image_size, num_classes=num_classes)
    aug_layers = build_augmentation_layers(aug_config) if use_augmentation else None

    # 1. Training Dataset Pipeline
    ds_train = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    ds_train = ds_train.shuffle(buffer_size=max(100, len(train_files)), seed=seed)
    ds_train = ds_train.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if use_augmentation and aug_layers is not None:
        ds_train = ds_train.map(
            lambda x, y: (aug_layers(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Validation Dataset Pipeline (NO augmentation)
    ds_val = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    ds_val = ds_val.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 3. Test Dataset Pipeline (NO augmentation)
    ds_test = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    ds_test = ds_test.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(buffer_size=tf.data.AUTOTUNE)

    metadata = {
        "classes": classes,
        "num_classes": num_classes,
        "image_size": list(image_size),
        "batch_size": batch_size,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "test_samples": len(test_files),
        "total_samples": len(train_files) + len(val_files) + len(test_files),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }

    return ds_train, ds_val, ds_test, metadata
