# pybrain/core/metrics.py
"""
Common segmentation metrics (Dice, HD95, Volume).
"""

import numpy as np


def compute_dice(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes the Dice score between two binary masks."""
    y_pred, y_true = y_pred > 0, y_true > 0
    intersection = np.logical_and(y_pred, y_true).sum()
    return 2.0 * intersection / (y_pred.sum() + y_true.sum() + 1e-8)


def compute_volume_cc(mask: np.ndarray, vox_vol_cc: float) -> float:
    """Computes volume in cubic centimeters from a binary mask."""
    return float(mask.sum()) * vox_vol_cc
