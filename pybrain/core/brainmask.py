# pybrain/core/brainmask.py
"""
Robust brain masking with HD-BET integration and morphological fallback.

Priority order:
  1. HD-BET (if installed) — deep-learning skull stripper, best accuracy.
  2. Morphological mask — classical Otsu + morphological ops, no GPU needed.
  3. Nonzero union — last resort for pre-processed / BraTS-style inputs.

All paths accept a ref_nifti_path (the reference sequence NIfTI) so the
caller controls which volume drives the mask.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ─── HD-BET availability ─────────────────────────────────────────────────────

_hdbet_checked = False
_hdbet_ok = False


def _hdbet_available() -> bool:
    """Check whether HD-BET is importable (cached after first call)."""
    global _hdbet_checked, _hdbet_ok
    if not _hdbet_checked:
        _hdbet_checked = True
        try:
            import HD_BET  # noqa: F401

            _hdbet_ok = True
            logger.info("HD-BET available for skull stripping")
        except ImportError:
            _hdbet_ok = False
            logger.info("HD-BET not installed — using morphological fallback")
    return _hdbet_ok


# ─── BraTS pre-processed detection ───────────────────────────────────────────


def _is_already_skull_stripped(volumes: Dict[str, np.ndarray]) -> bool:
    """
    Detect whether volumes are already skull-stripped (BraTS convention).

    BraTS images have zero background and nonzero only inside the brain.
    A robust heuristic: if the standard deviation across sequences is
    very low in the outer 20 % of the FOV the data is likely skull-stripped.
    """
    for vol in volumes.values():
        if vol is None:
            continue
        d, h, w = vol.shape
        # outer rim
        rim = np.concatenate(
            [
                vol[: d // 5, :, :].ravel(),
                vol[-d // 5 :, :, :].ravel(),
                vol[:, : h // 5, :].ravel(),
                vol[:, -h // 5 :, :].ravel(),
            ]
        )
        if np.std(rim) > 1.0:
            return False
    return True


# ─── HD-BET skull stripping ──────────────────────────────────────────────────


def _strip_with_hdbet(
    nifti_path: Path,
    out_mask_path: Path,
    work_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Run HD-BET on a single NIfTI file and return the binary mask.

    Parameters
    ----------
    nifti_path : Path
        Input NIfTI file (e.g. T1).
    out_mask_path : Path
        Where to save the resulting mask NIfTI.
    work_dir : Path, optional
        Temporary directory for HD-BET intermediate files.

    Returns
    -------
    mask : np.ndarray or None
        Binary float32 mask, or None on failure.
    """
    if not _hdbet_available():
        return None

    try:
        from HD_BET.run import run_hd_bet  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("HD-BET import failed at runtime")
        return None

    tmp_dir = work_dir or Path(tempfile.mkdtemp(prefix="hdbet_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Running HD-BET on {nifti_path.name}...")
        run_hd_bet(
            str(nifti_path),
            str(out_mask_path),
            mode="fast",
            device=0,  # GPU 0 — falls back to CPU internally if unavailable
            bet=True,
        )
        if out_mask_path.exists():
            import nibabel as nib

            mask = nib.load(str(out_mask_path)).get_fdata().astype(np.float32)
            mask = (mask > 0.5).astype(np.float32)
            return mask
    except Exception as exc:
        logger.warning(f"HD-BET failed ({exc}) — falling back to morphological mask")

    return None


# ─── Morphological fallback ──────────────────────────────────────────────────


def _morphological_mask(
    volumes: Dict[str, np.ndarray],
    vox_vol_cc: float,
) -> np.ndarray:
    """
    Classical Otsu + morphological skull stripping.
    Weighted combination of T1/T1c (×1.2) and T2/FLAIR (×0.8).
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import ball, closing, erosion, opening

    ref_vols = []
    weights = {"T1": 1.2, "T1c": 1.2, "T2": 0.8, "FLAIR": 0.8}

    for seq, w in weights.items():
        if seq in volumes and volumes[seq] is not None:
            v = volumes[seq]
            p2, p98 = np.percentile(v, 2), np.percentile(v, 98)
            norm = np.clip((v - p2) / (p98 - p2 + 1e-8), 0, 1)
            ref_vols.append((norm * w, v.shape))

    if not ref_vols:
        return np.zeros_like(next(iter(volumes.values()))).astype(np.float32)

    shapes = {shape for _, shape in ref_vols}
    if len(shapes) > 1:
        raise ValueError(f"All volumes must have the same shape, got {shapes}")

    combined = np.sum(np.stack([v for v, _ in ref_vols], axis=0), axis=0)
    combined = combined / np.max(combined)

    d, h, w = combined.shape
    center = combined[d // 4 : 3 * d // 4, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    try:
        thresh = threshold_otsu(center)
    except Exception:
        thresh = 0.20

    thresh = max(thresh * 1.05, 0.18)
    mask = combined > thresh

    mask = opening(mask, footprint=ball(2))
    brain = _largest_component(mask)
    brain = closing(brain.astype(bool), footprint=ball(6))
    brain = ndimage.binary_fill_holes(brain).astype(np.float32)

    vol_cc = float(brain.sum() * vox_vol_cc)
    if vol_cc > 1650.0:
        brain = erosion(brain.astype(bool), footprint=ball(2))
        brain = _largest_component(brain.astype(np.float32))

    brain = ndimage.binary_fill_holes(brain).astype(np.float32)
    return brain.astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.float32)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return (labeled == best).astype(np.float32)


# ─── Nonzero union fallback ──────────────────────────────────────────────────


def _mask_from_nonzero(volumes: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Last-resort mask: union of nonzero voxels across all sequences.
    Useful when data is already skull-stripped (BraTS convention).
    """
    mask = None
    for vol in volumes.values():
        if vol is None:
            continue
        nz = (vol > 0).astype(np.float32)
        mask = nz if mask is None else np.maximum(mask, nz)
    if mask is None:
        first = next(iter(volumes.values()))
        mask = np.ones_like(first, dtype=np.float32)
    return mask


# ─── Public API ───────────────────────────────────────────────────────────────


def robust_brain_mask(
    volumes: Dict[str, np.ndarray],
    vox_vol_cc: float = 0.001,
    ref_nifti_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute a brain mask using the best available method.

    Parameters
    ----------
    volumes : dict
        Sequence name -> 3-D numpy array (T1, T1c, T2, FLAIR).
    vox_vol_cc : float
        Single-voxel volume in cubic centimetres.
    ref_nifti_path : Path, optional
        Path to the reference NIfTI file (e.g. T1).  When HD-BET is
        available this file is passed to the deep-learning stripper.
    work_dir : Path, optional
        Temp directory for HD-BET intermediate outputs.

    Returns
    -------
    mask : np.ndarray
        Binary float32 mask, same shape as input volumes.
    """
    # Quick BraTS detection
    if _is_already_skull_stripped(volumes):
        logger.info("Volumes appear skull-stripped — using nonzero union mask")
        return _mask_from_nonzero(volumes)

    # Try HD-BET first
    if _hdbet_available() and ref_nifti_path is not None:
        out_mask = (work_dir or Path(tempfile.gettempdir())) / "hdbet_mask.nii.gz"
        mask = _strip_with_hdbet(ref_nifti_path, out_mask, work_dir=work_dir)
        if mask is not None:
            vol_cc = float(mask.sum() * vox_vol_cc)
            if 800 <= vol_cc <= 1700:
                logger.info(f"HD-BET mask: {vol_cc:.0f} cc")
                return mask
            logger.warning(f"HD-BET mask {vol_cc:.0f} cc outside 800-1700 cc — falling back")

    # Morphological fallback
    logger.info("Using morphological skull stripping")
    mask = _morphological_mask(volumes, vox_vol_cc)

    vol_cc = float(mask.sum() * vox_vol_cc)
    if vol_cc < 800 or vol_cc > 1700:
        logger.warning(f"Morphological mask {vol_cc:.0f} cc outside expected range (800-1700 cc)")

    return mask.astype(np.float32)
