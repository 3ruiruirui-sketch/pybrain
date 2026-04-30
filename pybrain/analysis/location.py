# pybrain/analysis/location.py
"""
Automated Tumour Location Analysis
====================================
Maps tumour segmentation to anatomical brain regions using
atlas-based labelling and hemisphere analysis.
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


def analyse_location(
    seg: np.ndarray,
    brain_mask: np.ndarray,
    ref_img: Optional[nib.Nifti1Image] = None,
) -> Dict[str, Any]:
    """
    Analyse tumour location in standard anatomical space.

    Returns a dict with hemisphere, lobe, and region information.
    """
    shape = seg.shape
    midline = shape[2] // 2  # X axis = left/right

    wt_mask = seg > 0
    if not wt_mask.any():
        return {"hemisphere": "none", "regions": [], "multi_focal": False}

    # Centre of mass
    com = ndimage.center_of_mass(wt_mask)
    com_x = com[2]  # X in RAS

    if com_x < midline - 10:
        hemisphere = "left"
    elif com_x > midline + 10:
        hemisphere = "right"
    else:
        hemisphere = "bilateral"

    # Extent across hemispheres
    left_vox = int(wt_mask[:, :, :midline].sum())
    right_vox = int(wt_mask[:, :, midline:].sum())
    total_vox = left_vox + right_vox
    if total_vox > 0:
        left_pct = left_vox / total_vox * 100
        right_pct = right_vox / total_vox * 100
    else:
        left_pct = right_pct = 50.0

    # Lobe estimation (crude z-slice based)
    d = shape[0]
    frontal = wt_mask[: d // 3, :, :].sum()
    parietal = wt_mask[d // 3 : 2 * d // 3, :, :].sum()
    occipital = wt_mask[2 * d // 3 :, :, :].sum()

    lobe_counts = {
        "frontal": int(frontal),
        "parietal": int(parietal),
        "occipital": int(occipital),
    }
    primary_lobe = max(lobe_counts, key=lobe_counts.get)

    # Multi-focal detection
    labeled, n_foci = ndimage.label(wt_mask)
    multi_focal = n_foci > 1

    return {
        "hemisphere": hemisphere,
        "hemisphere_pct": {"left": round(left_pct, 1), "right": round(right_pct, 1)},
        "primary_lobe": primary_lobe,
        "lobe_voxels": lobe_counts,
        "n_foci": n_foci,
        "multi_focal": multi_focal,
        "centre_of_mass": [round(c, 1) for c in com],
    }
