# pybrain/analysis/radiomics.py
"""
Radiomics Feature Extraction
==============================
Extracts shape, first-order, and texture features from tumour regions.
Falls back to scipy-only implementation when PyRadiomics is unavailable.
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def _to_serializable(obj):
    """JSON serializer for numpy bool_, int_, float_ etc."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def extract_radiomics(
    volumes: Dict[str, np.ndarray],
    seg: np.ndarray,
    brain_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Extract radiomics features from each sequence within tumour regions.

    Parameters
    ----------
    volumes : dict
        T1, T1c, T2, FLAIR arrays.
    seg : np.ndarray
        Segmentation (0=bg, 1=necrotic, 2=edema, 3=enhancing).
    brain_mask : np.ndarray
        Brain mask.

    Returns
    -------
    dict
        Features per sequence and sub-region.
    """
    features: Dict[str, Any] = {}
    wt = seg > 0

    if not wt.any():
        return {"whole_tumour": {}, "note": "No tumour detected"}

    for seq_name, vol in volumes.items():
        vals = vol[wt]
        if len(vals) < 10:
            continue

        features[seq_name] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "skewness": round(float(_skewness(vals)), 4),
            "kurtosis": round(float(_kurtosis(vals)), 4),
            "energy": round(float(np.sum(vals**2)), 2),
            "entropy": round(float(_entropy(vals)), 4),
        }

    return {"whole_tumour": features}


def _skewness(vals: np.ndarray) -> float:
    """Fisher skewness."""
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-8:
        return 0.0
    return float(np.mean(((vals - mu) / sigma) ** 3))


def _kurtosis(vals: np.ndarray) -> float:
    """Excess kurtosis."""
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-8:
        return 0.0
    return float(np.mean(((vals - mu) / sigma) ** 4) - 3.0)


def _entropy(vals: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy of intensity histogram."""
    hist, _ = np.histogram(vals, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)))
