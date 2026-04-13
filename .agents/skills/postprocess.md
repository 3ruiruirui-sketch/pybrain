---
description: Postprocess Skill
---
# Postprocess Protocol

## Input Requirements
- ensemble_prob: shape (3, H, W, D) fused outputs (0: Core, 1: Whole, 2: Enhancing)
- Local clinical configs: thresholds for core, whole, enhancing.

## Interpretation Algorithm
1. Apply dynamic threshold filtering on ensemble targets for Tumor Core, Whole Tumor, and Enhancing Tumor subsets.
2. Apply logic derivation:
   - `necrotic = np.clip(tc_bin - et_bin, 0, 1)`
   - `edema = np.clip(wt_bin - tc_bin, 0, 1)`
   - `enhancing = et_bin`
3. Resolve into semantic multi-class (0: background, 1: necrotic, 2: edema, 3: enhancing).

## Quality Constraints
- Must aggressively isolate the largest connected component to filter out scattered false positives (noise removal), maintaining components above 0.5 minimum cc.
- Ensure strict compliance with physical brain_mask boundary.
