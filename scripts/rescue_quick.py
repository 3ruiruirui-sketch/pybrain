#!/usr/bin/env python3
"""Quick rescue for failed segmentation."""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
from scipy import ndimage

print("Loading files...")
base = Path("/Users/ssoares/Downloads/PY-BRAIN/results/SOARES_MARIA_CELESTE_20260405_210116")

# Load
prob = nib.load(base / "ensemble_probability.nii.gz", mmap=True).get_fdata()
brain = nib.load(base / "brain_mask.nii.gz", mmap=True).get_fdata()
ct = nib.load(base / "ct_tumour_density.nii.gz", mmap=True).get_fdata()

print(f"Prob shape: {prob.shape}")
voxel_vol = np.prod(nib.load(base / "ensemble_probability.nii.gz", mmap=True).header.get_zooms())

# Higher threshold rescue
wt_prob = prob[1] if prob.ndim == 4 else prob
wt = (wt_prob > 0.65) & (brain > 0) & (ct > 0.3)

# Remove small components
labeled, n = ndimage.label(wt)
wt_clean = np.zeros_like(wt)
min_vox = int(1.0 * 1000 / voxel_vol)
for i in range(1, n + 1):
    if (labeled == i).sum() >= min_vox:
        wt_clean[labeled == i] = 1

# Volumes
wt_cc = wt_clean.sum() * voxel_vol / 1000
brain_cc = brain.sum() * voxel_vol / 1000
pct = 100 * wt_cc / brain_cc

print(f"Rescued WT: {wt_cc:.1f} cc ({pct:.1f}% of brain)")
print("Original: 1451.6 cc (99.9%)")
print(f"Reduction: {1451.6 / max(wt_cc, 0.1):.1f}x")

# Save
seg = wt_clean.astype(np.uint8)
nib.save(
    nib.Nifti1Image(seg, nib.load(base / "brain_mask.nii.gz", mmap=True).affine), base / "segmentation_rescue.nii.gz"
)
print("Saved: segmentation_rescue.nii.gz")

with open(base / "rescue_stats.json", "w") as f:
    json.dump({"wt_cc": wt_cc, "brain_cc": brain_cc, "pct": pct, "reduction": 1451.6 / max(wt_cc, 0.1)}, f)
print("Saved: rescue_stats.json")
