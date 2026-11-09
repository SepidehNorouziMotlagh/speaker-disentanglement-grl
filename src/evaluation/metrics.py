"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Equal Error Rate: point where FPR == FNR."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fnr = 1 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[eer_index] + fnr[eer_index]) / 2)