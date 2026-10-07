"""
Comprehensive evaluation, metrics, confusion matrices, diagnostics, and explainability modules.
"""

from src.evaluation.metrics import evaluate_model_performance, generate_classification_report
from src.evaluation.confusion_matrix import compute_and_plot_confusion_matrix
from src.evaluation.diagnostics import diagnose_training_fit, generate_deployment_recommendations
from src.evaluation.explainability import generate_gradcam_heatmap, overlay_gradcam

__all__ = [
    "evaluate_model_performance",
    "generate_classification_report",
    "compute_and_plot_confusion_matrix",
    "diagnose_training_fit",
    "generate_deployment_recommendations",
    "generate_gradcam_heatmap",
    "overlay_gradcam",
]
