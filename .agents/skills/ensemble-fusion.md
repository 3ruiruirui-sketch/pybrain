---
description: Ensemble Fusion Skill
---
# Ensemble Fusion Protocol

## Input Requirements
- segresnet_prob: shape (4, H, W, D) probability maps
- tta4_prob: shape (4, H, W, D) probability maps
- ct_data: optional CT volume for boost

## Fusion Algorithm
1. Weighted average: ensemble = 0.6 * segresnet + 0.4 * tta4
2. Apply CT boost if available:
   - Create prior from CT voxels in [25, 60] HU range
   - Boost whole tumor channel by factor 0.4
3. Output channels:
   - channel 0: tumor core (TC)
   - channel 1: whole tumor (WT)
   - channel 2: enhancing tumor (ET)

## Quality Constraints
- Probability values must remain in [0,1]
- Maintain brain mask boundaries
- Log uncertainty metrics
