"""Post-processing for tumor segmentation.

Extracted from scripts/3_brain_tumor_analysis.py to provide a single source
of truth for segmentation post-processing logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi

from pybrain.io.logging_utils import get_logger


logger = get_logger("core.postprocessing")


@dataclass
class PostprocessingConfig:
    """Configuration for post-processing steps."""
    
    # Shape-based component filtering
    shape_filtering: bool = True
    max_eccentricity: float = 0.98
    min_solidity: float = 0.55
    
    # CRF refinement
    crf_refinement: bool = False
    
    # Anatomical constraints
    anatomical_constraints: bool = True
    surface_threshold_mm: float = 8.0
    
    # Edema pruning
    prune_isolated_edema: bool = True
    edema_core_gap_mm: float = 3.0
    edema_max_distance_mm: float = 40.0
    
    # FLAIR intensity filtering
    edema_intensity_filter: bool = True
    edema_min_flair_zscore: float = 1.0
    edema_zscore_distance_tiers: Optional[list] = None
    
    # Minimum component size
    min_component_cc: float = 0.5


def postprocess_segmentation(
    ensemble_prob: np.ndarray,
    brain_mask: np.ndarray,
    vox_vol_cc: float,
    volumes: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    config: PostprocessingConfig,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tumor_type: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Convert ensemble probability maps to a binary BraTS segmentation.

    Steps:
      1. Threshold → binary masks with hierarchical consistency (ET⊂TC⊂WT).
      2. Shape-based component filtering.
      3. Anatomical surface-distance constraints.
      4. BraTS label derivation (necrosis, edema, enhancing).
      5. Prune isolated edema components.
      6. FLAIR-intensity edema filtering.
      7. Small isolated component removal.

    Returns
    -------
    seg_full, necrotic, edema, enhancing, final_thresholds
    """
    tc_prob = ensemble_prob[0]
    wt_prob = ensemble_prob[1]
    et_prob = ensemble_prob[2]

    wt_thresh = thresholds.get("wt", 0.50)
    tc_thresh = thresholds.get("tc", 0.40)
    et_thresh = thresholds.get("et", 0.45)
    
    logger.info(f"Thresholds applied: WT={wt_thresh}  TC={tc_thresh}  ET={et_thresh}")

    # ── 1. Threshold + hierarchical consistency ───────────────────────────────
    _max_prob = float(wt_prob.max())
    _model_non_activation = _max_prob < 0.10
    if _model_non_activation:
        logger.warning(
            f"⚠️  MODEL NON-ACTIVATION DETECTED — max_prob={_max_prob:.3f} < 0.10 "
            "(BraTS model sees no tumour signal; likely non-GBM pathology)"
        )

    wt_bin = (wt_prob > wt_thresh).astype(np.float32) * brain_mask
    tc_bin = (tc_prob > tc_thresh).astype(np.float32) * brain_mask * wt_bin
    et_bin = (et_prob > et_thresh).astype(np.float32) * brain_mask * tc_bin

    # ── 2. Shape-based component filtering ────────────────────────────────────
    if config.shape_filtering:
        from skimage.measure import label, regionprops

        labels = label(wt_bin > 0)
        props = regionprops(labels)
        keep_mask = np.zeros_like(wt_bin, dtype=bool)
        for p in props:
            # eccentricity not available for 3D images, skip that check
            is_good = (
                p.solidity >= config.min_solidity
                and getattr(p, "extent", 1.0) >= 0.005
            )
            if is_good or (p.area * vox_vol_cc > 1.0):
                keep_mask[labels == p.label] = True
    else:
        keep_mask = np.ones_like(wt_bin, dtype=bool)

    wt_bin = wt_bin * keep_mask
    tc_bin = tc_bin * keep_mask
    et_bin = et_bin * keep_mask

    # ── 3. Anatomical constraints ─────────────────────────────────────────────
    if config.anatomical_constraints:
        wt_bin = apply_anatomical_constraints(wt_bin, brain_mask, voxel_spacing, config.surface_threshold_mm)
        tc_bin = tc_bin * (wt_bin > 0)
        et_bin = et_bin * (wt_bin > 0)

    # ── 4. BraTS label derivation ─────────────────────────────────────────────
    enhancing = et_bin
    necrotic = np.clip(tc_bin - enhancing, 0, 1)
    edema = np.clip(wt_bin - tc_bin, 0, 1)

    seg_full = np.zeros_like(necrotic, dtype=np.uint8)
    seg_full[edema > 0] = 2
    seg_full[necrotic > 0] = 1
    seg_full[enhancing > 0] = 4  # BraTS convention: 4 = enhancing tumor

    # ── 5. Prune edema unlikely to be tumour-related ─────────────────────────
    dist_to_core_map: Optional[np.ndarray] = None
    core_mask_global = (necrotic > 0) | (enhancing > 0)

    if config.prune_isolated_edema:
        core_mask = core_mask_global
        if core_mask.any():
            gap_mm = config.edema_core_gap_mm
            min_vs = float(min(voxel_spacing)) or 1.0
            iters = max(1, int(round(gap_mm / min_vs)))
            core_expanded = ndi.binary_dilation(core_mask, iterations=iters)

            # (i) Connected-component pruning
            wt_mask_full = seg_full > 0
            labeled_wt, n_wt = ndi.label(wt_mask_full)  # type: ignore[arg-type]
            n_wt = int(n_wt)
            keep_wt = np.zeros_like(wt_mask_full, dtype=bool)
            kept_comps = 0
            for cid in range(1, n_wt + 1):
                comp = labeled_wt == cid
                if bool((core_expanded & comp).any()):
                    keep_wt |= comp
                    kept_comps += 1

            removed_vox_comp = int((wt_mask_full & ~keep_wt).sum())
            if removed_vox_comp > 0:
                logger.info(
                    f"Pruned {n_wt - kept_comps} isolated edema component(s): "
                    f"{removed_vox_comp * vox_vol_cc:.1f} cc removed (kept {kept_comps} of {n_wt} WT components)"
                )

            # (ii) Distance-based pruning
            max_dist_mm = config.edema_max_distance_mm
            if max_dist_mm > 0 or config.edema_zscore_distance_tiers:
                dist_to_core = ndi.distance_transform_edt(~core_mask, sampling=voxel_spacing)
                if isinstance(dist_to_core, tuple):
                    dist_to_core = dist_to_core[0]
                dist_to_core_map = dist_to_core
            
            if max_dist_mm > 0 and dist_to_core_map is not None:
                edema_mask_current = (seg_full == 2) & keep_wt
                edema_too_far = edema_mask_current & (dist_to_core_map > max_dist_mm)  # type: ignore[operator]
                removed_vox_dist = int(edema_too_far.sum())
                if removed_vox_dist > 0:
                    logger.info(
                        f"Pruned {removed_vox_dist * vox_vol_cc:.1f} cc of edema beyond {max_dist_mm:.0f} mm from core"
                    )
                    keep_wt = keep_wt & ~edema_too_far

            # Apply final mask
            total_removed = int((wt_mask_full & ~keep_wt).sum())
            if total_removed > 0:
                seg_full[~keep_wt] = 0
                edema = edema * keep_wt.astype(edema.dtype)
                necrotic = necrotic * keep_wt.astype(necrotic.dtype)
                enhancing = enhancing * keep_wt.astype(enhancing.dtype)

    # ── 6. FLAIR-intensity edema filter ──────────────────────────────────────
    if config.edema_intensity_filter:
        flair_vol = volumes.get("FLAIR")
        if flair_vol is not None and brain_mask is not None:
            non_tumour_brain = (brain_mask > 0) & (seg_full == 0)
            n_ref = int(non_tumour_brain.sum())
            if n_ref > 1000:
                flair_ref = flair_vol[non_tumour_brain]
                flair_med = float(np.median(flair_ref))
                flair_mad = float(np.median(np.abs(flair_ref - flair_med)))
                flair_sd = max(flair_mad * 1.4826, 1e-6)

                edema_mask_curr = seg_full == 2
                flair_z = (flair_vol - flair_med) / flair_sd

                tiers = config.edema_zscore_distance_tiers
                if tiers and dist_to_core_map is not None:
                    tiers_norm = [(float(d), float(z)) for d, z in tiers]
                    z_threshold = np.full_like(flair_z, tiers_norm[-1][1], dtype=np.float32)
                    prev_d = -1.0
                    for d_max, z_min in tiers_norm:
                        in_band = (dist_to_core_map > prev_d) & (dist_to_core_map <= d_max)
                        z_threshold = np.where(in_band, z_min, z_threshold)
                        prev_d = d_max
                    edema_too_dim = edema_mask_curr & (flair_z < z_threshold)
                    rule_desc = "tiered " + ", ".join(f"d≤{d:.0f}mm:z≥{z:.2f}" for d, z in tiers_norm)
                else:
                    min_z = config.edema_min_flair_zscore
                    edema_too_dim = edema_mask_curr & (flair_z < min_z)
                    rule_desc = f"uniform z≥{min_z:.2f}"

                n_dim = int(edema_too_dim.sum())
                if n_dim > 0:
                    logger.info(
                        f"FLAIR-intensity filter ({rule_desc}): removed {n_dim * vox_vol_cc:.1f} cc dim edema"
                    )
                    seg_full[edema_too_dim] = 0
                    edema = edema * (1 - edema_too_dim.astype(edema.dtype))

    # ── 7. Small component removal ────────────────────────────────────────────
    min_voxels = int(config.min_component_cc / vox_vol_cc)
    labeled, n_comps = ndi.label(seg_full > 0)  # type: ignore[arg-type]
    if n_comps > 0:
        sizes = ndi.sum(seg_full > 0, labeled, range(1, n_comps + 1))  # type: ignore[arg-type]
        keep = np.zeros_like(seg_full, dtype=bool)
        largest = int(np.argmax(sizes)) + 1
        for i, sz in enumerate(sizes, start=1):
            if sz >= min_voxels or i == largest:
                keep[labeled == i] = True
        seg_full[~keep] = 0

    final_thresholds = {
        "wt": wt_thresh,
        "tc": tc_thresh,
        "et": et_thresh,
        "_model_non_activation": _model_non_activation,
    }

    return seg_full, necrotic, edema, enhancing, final_thresholds


def apply_anatomical_constraints(
    mask: np.ndarray,
    brain_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    surface_threshold_mm: float = 8.0,
) -> np.ndarray:
    """
    Suppress tumor voxels too close to the brain surface (likely skull stripping artifacts).
    """
    from scipy.ndimage import distance_transform_edt

    # Distance from brain surface (distance to background outside brain)
    # Invert brain_mask: 1 outside brain, 0 inside brain
    brain_mask_bool = brain_mask.astype(bool)
    surface_dist = distance_transform_edt(~brain_mask_bool, sampling=voxel_spacing)
    if isinstance(surface_dist, tuple):
        surface_dist = surface_dist[0]

    # Suppress voxels within threshold mm of surface
    surface_suppression = mask * (surface_dist >= surface_threshold_mm).astype(mask.dtype)
    return surface_suppression
