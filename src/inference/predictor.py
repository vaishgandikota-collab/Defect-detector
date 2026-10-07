"""
Real-time manufacturing defect inference engine.
Handles single and batch predictions, Grad-CAM overlays, latency benchmarking, and quality decisions.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from src.data.preprocessing import preprocess_image
from src.evaluation.explainability import generate_gradcam_heatmap, overlay_gradcam
from utils.file_utils import load_json
from utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    "normal"
]


class DefectPredictor:
    """
    Production-grade inference engine for real-time defect classification and quality disposition.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.image_size = image_size
        self.model: Optional[keras.Model] = None
        self.model_path = Path(model_path) if model_path else None

        # Load class names
        if class_names is not None:
            self.class_names = class_names
        elif class_names_path and Path(class_names_path).exists():
            self.class_names = load_json(class_names_path)
        else:
            self.class_names = DEFAULT_CLASSES

        if self.model_path and self.model_path.exists():
            self.load_model(self.model_path)

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Loads the compiled .keras model into memory.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found at: {p.resolve()}")
        
        logger.info(f"Loading defect detection model from: {p}")
        self.model = keras.models.load_model(str(p), compile=False)
        self.model_path = p

    def is_ready(self) -> bool:
        """Checks if the underlying neural network model is loaded."""
        return self.model is not None

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        generate_explainability: bool = True,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes single-image inference and generates quality disposition.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please provide a valid model path.")

        # Preprocess input image to (224, 224, 3) normalized float array
        processed_np = preprocess_image(image_input, target_size=self.image_size, normalize=True)
        batch_tensor = np.expand_dims(processed_np, axis=0)

        # Measure inference latency
        t0 = time.perf_counter()
        probs = self.model(batch_tensor, training=False).numpy()[0]
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"

        # Determine Defect Status
        is_defective = ("normal" not in pred_class.lower()) and ("good" not in pred_class.lower())

        # Top-K predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "class_name": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                "confidence": round(float(probs[i]), 4),
                "confidence_percent": f"{float(probs[i]) * 100:.2f}%"
            }
            for i in top_indices
        ]

        # Industrial Decision Engine
        if not is_defective:
            recommendation = "PASS: Surface conforms to manufacturing specifications."
            severity = "None"
        elif confidence >= 0.80:
            recommendation = f"REJECT: High-confidence defect detected ({pred_class.upper()}). Route to scrap or rework."
            severity = "Major Defect" if ("crazing" in pred_class or "scale" in pred_class or "inclusion" in pred_class) else "Defect"
        else:
            recommendation = f"RE-INSPECT: Low-confidence anomaly detected ({pred_class.upper()}). Secondary manual inspection required."
            severity = "Minor Defect / Borderline"

        # Explainability via Grad-CAM
        gradcam_overlay = None
        gradcam_msg = "Disabled"
        if generate_explainability:
            heatmap, status = generate_gradcam_heatmap(self.model, batch_tensor, target_class_idx=pred_idx)
            gradcam_msg = status
            if heatmap is not None:
                gradcam_overlay = overlay_gradcam(processed_np, heatmap)

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.2f}%",
            "is_defective": is_defective,
            "status": "DEFECTIVE" if is_defective else "NON-DEFECTIVE",
            "defect_severity": severity,
            "recommendation": recommendation,
            "inference_time_ms": latency_ms,
            "top_predictions": top_predictions,
            "all_probabilities": {
                (self.class_names[i] if i < len(self.class_names) else f"Class_{i}"): round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "gradcam_status": gradcam_msg,
            "gradcam_overlay": gradcam_overlay,
            "preprocessed_image": processed_np,
        }

    def predict_batch(
        self,
        images: List[Union[str, Path, bytes, np.ndarray, Image.Image]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executes batch inference across a sequence of images.
        """
        results = []
        for img in images:
            res = self.predict(img, generate_explainability=False, top_k=top_k)
            results.append(res)
        return results
