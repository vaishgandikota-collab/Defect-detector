"""
Confusion Matrix computation, normalization, and light-themed visualization.
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from utils.file_utils import ensure_dir
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_and_plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str],
    output_png_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    title: str = "Manufacturing Defect Confusion Matrix"
) -> np.ndarray:
    """
    Computes and plots a clean, high-resolution confusion matrix with a strict LIGHT THEME.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    cm = confusion_matrix(y_t, y_p, labels=list(range(len(class_names))))
    
    if normalize:
        cm_display = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # Setup figure with strict light theme
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    if HAS_SEABORN:
        cmap = sns.light_palette("#2563EB", as_cmap=True)
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            linewidths=1.0,
            linecolor="#E5E7EB",
            ax=ax,
            annot_kws={"size": 11, "weight": "bold", "color": "#1E293B"}
        )
    else:
        im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                val_str = f"{cm_display[i, j]:.2f}" if normalize else f"{int(cm_display[i, j])}"
                ax.text(j, i, val_str,
                        ha="center", va="center",
                        color="#FFFFFF" if cm_display[i, j] > thresh else "#1E293B",
                        fontweight="bold", fontsize=11)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, color="#0F172A")
    ax.set_xlabel("Predicted Defect Class", fontsize=12, fontweight="semibold", labelpad=10, color="#1E293B")
    ax.set_ylabel("True Ground-Truth Class", fontsize=12, fontweight="semibold", labelpad=10, color="#1E293B")
    
    plt.xticks(rotation=35, ha="right", fontsize=10, color="#334155")
    plt.yticks(rotation=0, fontsize=10, color="#334155")
    plt.tight_layout()

    if output_png_path:
        out_p = Path(output_png_path)
        ensure_dir(out_p.parent)
        fig.savefig(out_p, dpi=300, facecolor="#FFFFFF", edgecolor="none")
        logger.info(f"Saved confusion matrix figure to: {out_p}")

    plt.close(fig)
    return cm
