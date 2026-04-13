# pybrain/core/brainmask.py
"""
Robust brain masking logic — morphological skull-stripping.
Targets ~1200–1600 cc for adult brains.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import ball, closing, erosion, opening


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.float32)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return (labeled == best).astype(np.float32)


def robust_brain_mask(volumes: dict, vox_vol_cc: float = 0.001) -> np.ndarray:
    """
    Revised brain masking logic (v2.1):
    - T1/T1c biased for solid anatomical shell.
    - Aggressive morphological closing (ball(6)) to bridge pathological gaps.
    - Solid-core approach to prevent "Swiss cheese" fragmentation.
    """
    from skimage.filters import threshold_otsu

    # 1. Prepare Multi-Contrast combined volume (T1 Biased)
    ref_vols = []
    # Weighted contribution: T1/T1c are more structurally stable than FLAIR for masking
    weights = {"T1": 1.2, "T1c": 1.2, "T2": 0.8, "FLAIR": 0.8}
    
    for seq, w in weights.items():
        if seq in volumes and volumes[seq] is not None:
            v = volumes[seq]
            p2, p98 = np.percentile(v, 2), np.percentile(v, 98)
            norm = np.clip((v - p2) / (p98 - p2 + 1e-8), 0, 1)
            ref_vols.append((norm * w, v.shape))

    if not ref_vols:
        return np.zeros_like(next(iter(volumes.values()))).astype(np.float32)

    # Validate all arrays have the same shape before stacking
    shapes = {shape for _, shape in ref_vols}
    if len(shapes) > 1:
        raise ValueError(
            f"Brain mask: all input volumes must have the same shape, got {shapes}"
        )

    # Combined using sum (Weighted) instead of Max for smoother boundaries
    combined = np.sum(np.stack([v for v, _ in ref_vols], axis=0), axis=0)
    combined = combined / np.max(combined)

    # 2. Otsu on central 60% crop
    d, h, w = combined.shape
    center = combined[d // 4 : 3 * d // 4, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    try:
        thresh = threshold_otsu(center)
    except Exception:
        thresh = 0.20

    # Stronger baseline to avoid "halo" noise from skull
    thresh = max(thresh * 1.05, 0.18)
    mask = combined > thresh

    # 3. Aggressive Solidification
    #    a. Sever bridges (small opening)
    mask = opening(mask, footprint=ball(2))
    #    b. Keep largest component (The Brain)
    brain = _largest_component(mask)
    #    c. STRENGTHEN SHELL: Aggressive closing to seal tumor gaps
    brain = closing(brain.astype(bool), footprint=ball(6))
    #    d. Fill internal holes
    brain = ndimage.binary_fill_holes(brain).astype(np.float32)

    # 4. Final Polish & Volume Guard
    vol_cc = float(brain.sum() * vox_vol_cc)
    
    # If still too large (> 1650cc), trim the edges slightly
    if vol_cc > 1650.0:
        brain = erosion(brain.astype(bool), footprint=ball(2))
        brain = _largest_component(brain.astype(np.float32))

    # Safety: ensure it is filled and clean
    brain = ndimage.binary_fill_holes(brain).astype(np.float32)
    return brain.astype(np.float32)
