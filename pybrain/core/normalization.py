# pybrain/core/normalization.py
"""
Signal normalization functions for medical imaging.
"""

import numpy as np


def norm01(arr: np.ndarray) -> np.ndarray:
    """
    Normalizes array to [0, 1] range using robust 2-98% percentiles
    to avoid outlier-induced contrast squash.
    """
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    if p98 == p2:
        # Fallback for binary or uniform masks
        denom = arr.max() - arr.min()
        return (arr - arr.min()) / (denom + 1e-8)
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)


def zscore_robust(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Standardized BraTS normalisation: z-score within brain,
    preserving high-intensity outliers via robust statistics.
    """
    # 1. Check if mask is robust enough for stats
    # 100k voxels (at 1mm is ~100cc). If smaller, the mask is likely a failure
    use_mask = mask.sum() > 100000

    if use_mask:
        vals = arr[mask.astype(bool)]
        # Robust stats: clip 0.5-99.5% for T1c/Enhancing contrast
        p_lo, p_hi = np.percentile(vals, 0.5), np.percentile(vals, 99.5)
        clipped_vals = np.clip(vals, p_lo, p_hi)
        mu, sigma = clipped_vals.mean(), clipped_vals.std()
    else:
        # Fallback: use robust 1st/99th trimmed stats of the whole image
        # (Standard BraTS pre-processing for near-empty/noisy masks)
        p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        vals = arr[(arr >= p1) & (arr <= p99)]
        if len(vals) < 1000:
            vals = arr.flatten()
        mu, sigma = vals.mean(), vals.std()

    out = (arr - mu) / (sigma + 1e-6)

    # Only mask the output if the mask was deemed reliable
    if use_mask:
        out[~mask.astype(bool)] = 0.0

    return out
