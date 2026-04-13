---
description: Data Loading Skill
---
# Data Loading Protocol

## Input Requirements
- Required modalities: T1, T1c, T2, FLAIR NIfTI files
- T1 defines the fixed reference space geometry (e.g., shape 160x256x256, 1mm iso)
- Optional CT volume

## Data Algorithm
1. Ensure all modalities are properly registered/resampled to T1 geometry natively.
2. Load all arrays via nibabel (float32 scaling).
3. Generate skull-stripping `brain_mask`.
4. Run z-score (1-99% range) standardization individually on FLAIR, T1, T1c, and T2, utilizing the `brain_mask` strictly.
5. Create Stacked Tensor matching native BraTS format (Channel 0: FLAIR, 1: T1, 2: T1c, 3: T2).

## Quality Constraints
- Check affine match and shape matches.
- CT arrays should preserve original Hounsfield Units (do not normalize CT like MRI).
