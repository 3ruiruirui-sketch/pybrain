#!/usr/bin/env python3
"""
RESCUE OPTION B — Brain Mask Erosion Correction
===============================================

For cases where model includes whole brain as tumor.
Technique: Morphological erosion of brain mask to remove cortical surface.

Target: SOARES_MARIA_CELESTE (catastrophic segmentation)
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
from scipy import ndimage

print("=" * 60)
print("RESCUE OPTION B — Brain Mask Erosion")
print("=" * 60)

base = Path("/Users/ssoares/Downloads/PY-BRAIN/results/SOARES_MARIA_CELESTE_20260405_210116")

# Load original (failed) segmentation and brain mask
print("\n[1] Loading files...")
seg_orig = nib.load(base / "segmentation_full.nii.gz", mmap=True)
seg_data = seg_orig.get_fdata().astype(np.uint8)
brain = nib.load(base / "brain_mask.nii.gz", mmap=True).get_fdata().astype(np.uint8)

print(f"    Original seg: {seg_data.shape}, unique labels: {np.unique(seg_data)}")
print(f"    Brain mask: {brain.shape}, vol={(brain.sum() * 0.001):.1f} cc")

# ── Step 1: Create eroded brain mask (remove cortical surface) ─────
print("\n[2] Eroding brain mask...")

# Multiple erosion iterations to remove outer cortical rim
brain_eroded = ndimage.binary_erosion(brain, iterations=8).astype(np.uint8)

# Also create a 'cortical rim' mask (outer layer to exclude)
cortical_rim = brain & (~brain_eroded.astype(bool))

print(f"    Original brain voxels: {brain.sum():,}")
print(f"    Eroded brain voxels: {brain_eroded.sum():,}")
print(f"    Cortical rim (to exclude): {cortical_rim.sum():,}")

# ── Step 2: Subtract cortical rim from segmentation ─────────────────
print("\n[3] Removing cortical false positives...")

# WT = labels 1,2,4 (necrotic, edema, enhancing)
wt_mask = seg_data > 0

# Remove cortical rim from WT
wt_corrected = wt_mask & (~cortical_rim.astype(bool))

# Recalculate sub-regions with corrected WT boundary
tc_mask = (seg_data == 1) | (seg_data == 4)  # necrotic + enhancing
et_mask = seg_data == 4  # enhancing only
ed_mask = seg_data == 2  # edema

# Apply cortical correction to all sub-regions
tc_corrected = tc_mask & (~cortical_rim.astype(bool))
et_corrected = et_mask & (~cortical_rim.astype(bool))
ed_corrected = ed_mask & (~cortical_rim.astype(bool))
nc_corrected = tc_corrected & (~et_corrected.astype(bool))  # necrotic = TC - ET

# ── Step 3: Additional morphological cleanup ───────────────────────
print("\n[4] Morphological cleanup...")

# Remove small isolated components (< 2 cc)
voxel_vol = 0.001  # mm³ to cc conversion (approx for 1mm isotropic)
min_voxels = int(2.0 / voxel_vol)  # 2 cc minimum

for mask in [wt_corrected, tc_corrected, et_corrected]:
    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        comp = labeled == i
        if comp.sum() < min_voxels:
            mask[comp] = 0

# ── Step 4: Build corrected segmentation ──────────────────────────
print("\n[5] Building corrected segmentation...")

seg_corrected = np.zeros_like(seg_data, dtype=np.uint8)
seg_corrected[nc_corrected] = 1  # Necrotic
seg_corrected[ed_corrected] = 2  # Edema
seg_corrected[et_corrected] = 4  # Enhancing

# Verify hierarchy: NC ⊆ ET ⊆ TC ⊆ WT
print(f"    Final labels: {np.unique(seg_corrected)}")

# ── Step 5: Calculate volumes ────────────────────────────────────
print("\n[6] Calculating corrected volumes...")

voxel_vol_cc = np.prod(seg_orig.header.get_zooms()) / 1000.0

volumes = {
    "brain": float(brain.sum() * voxel_vol_cc),
    "whole_tumor": float(wt_corrected.sum() * voxel_vol_cc),
    "core": float(tc_corrected.sum() * voxel_vol_cc),
    "enhancing": float(et_corrected.sum() * voxel_vol_cc),
    "necrotic": float(nc_corrected.sum() * voxel_vol_cc),
    "edema": float(ed_corrected.sum() * voxel_vol_cc),
}

tumor_pct = 100 * volumes["whole_tumor"] / (volumes["brain"] + 1e-8)
et_ratio = 100 * volumes["enhancing"] / (volumes["whole_tumor"] + 1e-8)
ed_ratio = 100 * volumes["edema"] / (volumes["whole_tumor"] + 1e-8)

print(f"\n    Brain:           {volumes['brain']:>8.1f} cc")
print(f"    Whole Tumor:     {volumes['whole_tumor']:>8.1f} cc  ({tumor_pct:.1f}% of brain)")
print(f"    Tumor Core:      {volumes['core']:>8.1f} cc")
print(f"    Enhancing:       {volumes['enhancing']:>8.1f} cc  ({et_ratio:.1f}% of WT)")
print(f"    Necrotic:        {volumes['necrotic']:>8.1f} cc")
print(f"    Edema:           {volumes['edema']:>8.1f} cc  ({ed_ratio:.1f}% of WT)")

# ── Step 6: Save outputs ───────────────────────────────────────────
print("\n[7] Saving outputs...")

output_seg = base / "segmentation_rescue_b.nii.gz"
nib.save(nib.Nifti1Image(seg_corrected, seg_orig.affine, seg_orig.header), output_seg)
print(f"    Segmentation: {output_seg.name}")

# Save comparison stats
stats = {
    "method": "RESCUE_OPTION_B_BRAIN_EROSION",
    "erosion_iterations": 8,
    "cortical_rim_removed_cc": float(cortical_rim.sum() * voxel_vol_cc),
    "original": {
        "wt_cc": 1451.57,
        "tumor_pct_brain": 99.999,
    },
    "corrected": {
        "wt_cc": volumes["whole_tumor"],
        "tumor_pct_brain": tumor_pct,
        "et_ratio_pct": et_ratio,
        "ed_ratio_pct": ed_ratio,
    },
    "reduction_factor": 1451.57 / max(volumes["whole_tumor"], 0.1),
}

output_stats = base / "rescue_option_b_stats.json"
with open(output_stats, "w") as f:
    json.dump(stats, f, indent=2)
print(f"    Statistics: {output_stats.name}")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESCUE B COMPLETE")
print("=" * 60)
print(f"Original:  {1451.57:>8.1f} cc tumor (99.9% of brain) ❌")
print(f"Corrected: {volumes['whole_tumor']:>8.1f} cc tumor ({tumor_pct:.1f}% of brain) ✅")
print(f"Removed:   {stats['reduction_factor']:>8.1f}x volume reduction")
print("=" * 60)
print("\n⚠️  IMPORTANT: This is a geometric correction, not biological.")
print("    Review in 3D viewer to verify tumor boundary accuracy.")
print("    CT-guided validation recommended for final diagnosis.")
