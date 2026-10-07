"""
Dataset preparation, organization, and synthetic manufacturing defect generator.
Ensures the pipeline is immediately executable with real or synthesized industrial surface datasets.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from utils.file_utils import ensure_dir, load_yaml_config, save_json
from utils.logging_utils import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)

DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    "normal"
]


def generate_synthetic_surface_sample(defect_type: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Synthesizes physically realistic industrial steel/metal surface textures with specific defect patterns.
    """
    w, h = size
    # Base metallic background with noise
    base_gray = random.randint(140, 180)
    img = np.full((h, w, 3), base_gray, dtype=np.uint8)
    
    # Add subtle metallic rolling texture
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    if defect_type == "scratches":
        # Sharp high-contrast directional lines
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(10, w - 10)
            y1 = random.randint(10, h - 10)
            x2 = x1 + random.randint(-80, 80)
            y2 = y1 + random.randint(-80, 80)
            color = (random.randint(40, 70), random.randint(40, 70), random.randint(40, 70))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 3))

    elif defect_type == "crazing":
        # Web-like micro-crack network
        for _ in range(random.randint(8, 16)):
            x1 = random.randint(20, w - 20)
            y1 = random.randint(20, h - 20)
            x2 = x1 + random.randint(-30, 30)
            y2 = y1 + random.randint(-30, 30)
            draw.line([(x1, y1), (x2, y2)], fill=(50, 50, 50), width=1)

    elif defect_type == "inclusion":
        # Dark non-metallic foreign particle inclusions
        for _ in range(random.randint(2, 6)):
            cx = random.randint(30, w - 30)
            cy = random.randint(30, h - 30)
            r = random.randint(4, 12)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 30, 30))

    elif defect_type == "patches":
        # Localized oxidized / discoloration patches
        for _ in range(random.randint(1, 3)):
            cx = random.randint(40, w - 40)
            cy = random.randint(40, h - 40)
            r = random.randint(20, 45)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(random.randint(80, 110), random.randint(80, 110), random.randint(80, 110)))

    elif defect_type == "pitted_surface":
        # Micro-cavities / small pinholes
        for _ in range(random.randint(15, 30)):
            cx = random.randint(10, w - 10)
            cy = random.randint(10, h - 10)
            r = random.randint(1, 4)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 40, 40))

    elif defect_type == "rolled-in_scale":
        # Elongated pressed scale ridges
        for _ in range(random.randint(2, 4)):
            x = random.randint(20, w - 60)
            y = random.randint(20, h - 60)
            draw.rectangle([x, y, x + random.randint(30, 80), y + random.randint(6, 18)], fill=(60, 60, 60))

    elif defect_type == "normal":
        # Smooth surface with minor uniform texture
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return np.array(pil_img)


def setup_sample_dataset(
    base_dir: Path,
    classes: List[str],
    samples_per_class: int = 40,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> None:
    """
    Generates structured train, validation, and test datasets.
    """
    set_seed(42)
    train_dir = base_dir / "train"
    val_dir = base_dir / "validation"
    test_dir = base_dir / "test"

    for c in classes:
        ensure_dir(train_dir / c)
        ensure_dir(val_dir / c)
        ensure_dir(test_dir / c)

    num_test = max(2, int(samples_per_class * test_split))
    num_val = max(2, int(samples_per_class * val_split))
    num_train = samples_per_class - num_test - num_val

    logger.info(f"Populating sample dataset: {num_train} train, {num_val} val, {num_test} test per class across {len(classes)} classes.")

    for c in classes:
        for i in range(num_train):
            img = generate_synthetic_surface_sample(c)
            Image.fromarray(img).save(train_dir / c / f"{c}_train_{i:03d}.png")
        for i in range(num_val):
            img = generate_synthetic_surface_sample(c)
            Image.fromarray(img).save(val_dir / c / f"{c}_val_{i:03d}.png")
        for i in range(num_test):
            img = generate_synthetic_surface_sample(c)
            Image.fromarray(img).save(test_dir / c / f"{c}_test_{i:03d}.png")

    logger.info(f"Dataset preparation complete in: {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and Organize Dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--samples-per-class", type=int, default=35, help="Number of samples to generate if synthetic")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_root = Path("data")
    classes = cfg["dataset"].get("classes", DEFECT_CLASSES)

    train_dir = Path(cfg["dataset"]["train_dir"])
    if not train_dir.exists() or not any(train_dir.iterdir()):
        logger.info("No existing dataset found. Generating demonstration manufacturing dataset.")
        setup_sample_dataset(
            base_dir=data_root,
            classes=classes,
            samples_per_class=args.samples_per_class,
            val_split=cfg["dataset"].get("validation_split", 0.15),
            test_split=cfg["dataset"].get("test_split", 0.15)
        )
    else:
        logger.info(f"Existing dataset discovered at: {train_dir.resolve()}")


if __name__ == "__main__":
    main()
