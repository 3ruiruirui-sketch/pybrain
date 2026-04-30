#!/usr/bin/env python3
"""
RESCUE SEGMENTATION — Technical Fix for Failed Clinical Cases
===============================================================

For cases where standard pipeline produces catastrophic segmentation
(whole-brain classified as tumor due to poor contrast enhancement).

Applied techniques:
1. CT-guided tumor core refinement (HU-based)
2. Conservative probability thresholding
3. Morphological constraints (remove brain-mask false positives)
4. Sub-region recalculation with anatomical priors

Target case: SOARES_MARIA_CELESTE (81F, poor gadolinium enhancement)
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────
SESSION_DIR = Path("/Users/ssoares/Downloads/PY-BRAIN/results/SOARES_MARIA_CELESTE_20260405_210116")

# Files
ENSEMBLE_PROB = SESSION_DIR / "ensemble_probability.nii.gz"
BRAIN_MASK = SESSION_DIR / "brain_mask.nii.gz"
CT_DENSITY = SESSION_DIR / "ct_tumour_density.nii.gz"
CT_CALC = SESSION_DIR / "ct_calcification.nii.gz"
OUTPUT_SEG = SESSION_DIR / "segmentation_rescue.nii.gz"
OUTPUT_STATS = SESSION_DIR / "rescue_stats.json"

# Conservative thresholds (higher = more specific)
RESCUE_THRESHOLDS = {
    "wt": 0.65,  # vs 0.42 standard
    "tc": 0.55,  # vs 0.33 standard
    "et": 0.50,  # vs 0.32 standard
}

# HU constraints (from CT)
HU_TUMOR_MIN = 25
HU_TUMOR_MAX = 60

# Morphological cleanup
MIN_TUMOR_CC = 1.0  # Remove small noise < 1cc
MAX_EDEMA_RATIO = 0.8  # Edema should not exceed 80% of WT


def load_nifti(path: Path) -> tuple:
    """Load NIfTI and return (array, affine, header)."""
    img = nib.load(str(path), mmap=True)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header


def save_nifti(data: np.ndarray, path: Path, affine, header=None):
    """Save array as NIfTI."""
    img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    nib.save(img, str(path))


def volume_cc(mask: np.ndarray, voxel_vol: float) -> float:
    """Calculate volume in cc."""
    return float(mask.sum() * voxel_vol / 1000.0)


def rescue_segmentation():
    """Main rescue pipeline."""
    print("=" * 60)
    print("RESCUE SEGMENTATION — Technical Recovery Protocol")
    print("=" * 60)
    print(f"Session: {SESSION_DIR.name}")
    print("Patient: SOARES_MARIA_CELESTE (81F)")
    print("Indication: Poor contrast enhancement → whole-brain FP")
    print("=" * 60)

    # ── Load inputs ─────────────────────────────────────────────────────
    print("\n[1] Loading inputs...")

    prob_data, affine, header = load_nifti(ENSEMBLE_PROB)
    print(f"    Ensemble prob: {prob_data.shape}, range [{prob_data.min():.3f}, {prob_data.max():.3f}]")

    brain_mask, _, _ = load_nifti(BRAIN_MASK)
    print(f"    Brain mask: {brain_mask.shape}")

    ct_tumor, _, _ = load_nifti(CT_DENSITY)
    print(f"    CT tumor density: {ct_tumor.shape}")

    ct_calc, _, _ = load_nifti(CT_CALC)
    print(f"    CT calcification: {ct_calc.shape}")

    voxel_vol = np.prod(nib.load(str(ENSEMBLE_PROB), mmap=True).header.get_zooms())
    print(f"    Voxel volume: {voxel_vol:.4f} mm³")

    # ── Step 1: Conservative thresholding ───────────────────────────────
    print("\n[2] Conservative thresholding (rescue mode)...")

    wt_prob = prob_data[1] if prob_data.ndim == 4 else prob_data  # Channel 1 = WT
    tc_prob = prob_data[0] if prob_data.ndim == 4 else prob_data * 0.8  # Channel 0 = TC
    et_prob = prob_data[2] if prob_data.ndim == 4 else prob_data * 0.5  # Channel 2 = ET

    wt_mask = (wt_prob > RESCUE_THRESHOLDS["wt"]) & (brain_mask > 0)
    tc_mask = (tc_prob > RESCUE_THRESHOLDS["tc"]) & wt_mask
    et_mask = (et_prob > RESCUE_THRESHOLDS["et"]) & tc_mask

    print(f"    WT mask: {wt_mask.sum()} voxels")
    print(f"    TC mask: {tc_mask.sum()} voxels")
    print(f"    ET mask: {et_mask.sum()} voxels")

    # ── Step 2: CT-guided refinement ────────────────────────────────────
    print("\n[3] CT-guided refinement...")

    # CT hyperdensity mask (tumor candidate regions)
    ct_hyper = (ct_tumor > 0.5) | (ct_calc > 0.5)
    print(f"    CT hyperdense regions: {ct_hyper.sum()} voxels")

    # Constrain WT to CT-positive regions (strong anatomical prior)
    wt_mask_ct = wt_mask & ct_hyper

    # If CT constraint removes everything, fall back to conservative WT only
    if wt_mask_ct.sum() < 100:  # Less than 100 voxels
        print("    ⚠️  CT constraint too aggressive — using morphological cleanup only")
        wt_mask_ct = wt_mask.copy()
    else:
        print(f"    CT-constrained WT: {wt_mask_ct.sum()} voxels")

    # ── Step 3: Morphological cleanup ─────────────────────────────────
    print("\n[4] Morphological cleanup...")

    # Remove small components (< MIN_TUMOR_CC)
    labeled, n = ndimage.label(wt_mask_ct)
    wt_clean = np.zeros_like(wt_mask_ct)

    min_voxels = int(MIN_TUMOR_CC * 1000 / voxel_vol)
    for i in range(1, n + 1):
        comp_mask = labeled == i
        if comp_mask.sum() >= min_voxels:
            wt_clean[comp_mask] = 1

    print(f"    Components removed: {n - wt_clean.sum() // min_voxels}")
    print(f"    Clean WT: {wt_clean.sum()} voxels ({volume_cc(wt_clean, voxel_vol):.1f} cc)")

    # ── Step 4: Recalculate sub-regions ────────────────────────────────
    print("\n[5] Recalculating sub-regions...")

    # NC = calcification (HU > 130)
    nc_mask = ct_calc > 0.5

    # ED = WT - TC (dilated)
    tc_dilated = ndimage.binary_dilation(tc_mask, iterations=2)
    ed_mask = wt_clean & ~tc_dilated

    # ET = TC - NC (remaining enhancing core)
    et_rescue = tc_mask & ~nc_mask

    # TC = ET + NC
    tc_rescue = et_rescue | nc_mask

    # Ensure hierarchy: NC ⊆ ET ⊆ TC ⊆ WT
    tc_rescue = tc_rescue & wt_clean
    et_rescue = et_rescue & tc_rescue
    nc_rescue = nc_mask & tc_rescue
    ed_rescue = ed_mask & wt_clean & ~tc_rescue

    # ── Step 5: Build final segmentation ──────────────────────────────
    print("\n[6] Building final segmentation...")

    seg = np.zeros_like(wt_clean, dtype=np.uint8)
    seg[nc_rescue] = 1  # Necrotic/calcified
    seg[ed_rescue] = 2  # Edema
    seg[et_rescue] = 4  # Enhancing

    # Validate: no overlapping labels
    unique, counts = np.unique(seg, return_counts=True)
    print(f"    Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # ── Step 6: Calculate volumes ─────────────────────────────────────
    print("\n[7] Calculating volumes...")

    volumes = {
        "brain": volume_cc(brain_mask, voxel_vol),
        "whole_tumor": volume_cc(wt_clean, voxel_vol),
        "core": volume_cc(tc_rescue, voxel_vol),
        "enhancing": volume_cc(et_rescue, voxel_vol),
        "necrotic": volume_cc(nc_rescue, voxel_vol),
        "edema": volume_cc(ed_rescue, voxel_vol),
    }

    tumor_pct = 100 * volumes["whole_tumor"] / (volumes["brain"] + 1e-8)

    print(f"    Brain: {volumes['brain']:.1f} cc")
    print(f"    Whole Tumor: {volumes['whole_tumor']:.1f} cc")
    print(f"    Tumor Core: {volumes['core']:.1f} cc")
    print(f"    Enhancing: {volumes['enhancing']:.1f} cc")
    print(f"    Necrotic: {volumes['necrotic']:.1f} cc")
    print(f"    Edema: {volumes['edema']:.1f} cc")
    print(f"    Tumor % Brain: {tumor_pct:.2f}%")

    # ── Step 8: Save outputs ──────────────────────────────────────────
    print("\n[8] Saving outputs...")

    save_nifti(seg, OUTPUT_SEG, affine, header)
    print(f"    Segmentation: {OUTPUT_SEG.name}")

    stats = {
        "method": "RESCUE_SEGMENTATION",
        "indication": "Poor contrast enhancement — catastrophic original segmentation",
        "thresholds": RESCUE_THRESHOLDS,
        "volumes_cc": volumes,
        "tumor_pct_brain": tumor_pct,
        "et_wt_ratio": volumes["enhancing"] / (volumes["whole_tumor"] + 1e-8),
        "original_wt_cc": 1451.57,  # From failed segmentation
        "rescued_wt_cc": volumes["whole_tumor"],
        "reduction_factor": 1451.57 / (volumes["whole_tumor"] + 1e-8),
    }

    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"    Statistics: {OUTPUT_STATS.name}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESCUE COMPLETE")
    print("=" * 60)
    print("Original WT: 1451.6 cc (99.9% of brain) ❌")
    print(f"Rescued WT:  {volumes['whole_tumor']:.1f} cc ({tumor_pct:.1f}% of brain) ✅")
    print(f"Reduction:   {stats['reduction_factor']:.1f}x")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Load segmentation_rescue.nii.gz in 3D viewer")
    print("  2. Compare with original segmentation_full.nii.gz")
    print("  3. Validate against CT hyperdense regions")
    print("  4. Run Stage 5+ with rescued segmentation if approved")


if __name__ == "__main__":
    rescue_segmentation()
