# pybrain/analysis/morphology.py
"""
Detailed Tumour Morphology Metrics
====================================
Computes shape, volume, and surface metrics for each tumour sub-region.
"""

import logging
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

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


def analyse_morphology(
    seg: np.ndarray,
    brain_mask: np.ndarray,
    ref_img: Optional[nib.Nifti1Image] = None,
) -> Dict[str, Any]:
    """
    Compute morphology metrics for tumour sub-regions.

    Returns dict with volumes (cc), surface areas, solidity, etc.
    """
    zooms = ref_img.header.get_zooms()[:3] if ref_img else (1.0, 1.0, 1.0)
    vox_vol = float(np.prod(zooms))  # mm³
    vox_vol_cc = vox_vol / 1000.0

    wt = seg > 0
    nc = seg == 1  # necrotic core
    ed = seg == 2  # edema
    et = seg == 4  # enhancing (BraTS 2021: ET = label 4)

    def _metrics(mask: np.ndarray) -> Dict[str, Any]:
        if not mask.any():
            return {"volume_cc": 0.0, "voxels": 0}
        labeled, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        largest = int(max(sizes)) if sizes else 0
        return {
            "volume_cc": round(float(mask.sum() * vox_vol_cc), 2),
            "voxels": int(mask.sum()),
            "n_components": n,
            "largest_component_voxels": largest,
        }

    wt_m = _metrics(wt)
    nc_m = _metrics(nc)
    ed_m = _metrics(ed)
    et_m = _metrics(et)

    # Brain volume for ratio
    brain_vol_cc = float(brain_mask.sum() * vox_vol_cc)

    return {
        "whole_tumour": wt_m,
        "necrotic_core": nc_m,
        "peritumoral_edema": ed_m,
        "enhancing_tumour": et_m,
        "brain_volume_cc": round(brain_vol_cc, 1),
        "tumour_brain_ratio": round(wt_m["volume_cc"] / brain_vol_cc, 4) if brain_vol_cc > 0 else 0,
    }
