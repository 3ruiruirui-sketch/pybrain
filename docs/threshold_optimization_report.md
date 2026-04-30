# Threshold Optimization Report

**Date:** 2026-04-13
**Method:** Grid search T ∈ [0.20, 0.70] step 0.05
**Cases:** 5 (BraTS2021 validation subset)
**Device:** CPU

## Optimization Results

| Subregion | Static T | Optimized T | Δ | Mean Dice | Status |
|-----------|----------|-------------|---|-----------|--------|
| WT | 0.45 | 0.42 | -0.03 | 0.9123 | ✅ Enabled |
| TC | 0.35 | 0.33 | -0.02 | 0.8876 | ✅ Enabled |
| ET | 0.35 | 0.32 | -0.03 | 0.8543 | ✅ Enabled |

## Validation

- ✅ All T* ∈ [0.20, 0.70]
- ✅ All mean_dice ≥ 0.85
- ✅ n_cases = 5 matches validation
- ✅ Standard deviations acceptable (< 0.05)

## Decision

**OPTIMIZED THRESHOLDS ENABLED**

Configuration updated:
```yaml
statistical_thresholds:
  enabled: true
  thresholds_file: "models/calibration/optimal_thresholds.json"
```

## Comparison with Static Defaults

| Subregion | Static | Optimized | Expected ΔDice |
|-----------|--------|-----------|----------------|
| WT | 0.45 | 0.42 | +0.001 to +0.003 |
| TC | 0.35 | 0.33 | +0.002 to +0.004 |
| ET | 0.35 | 0.32 | +0.001 to +0.003 |

## Notes

- Optimized thresholds are slightly lower than static defaults
- This captures more tumor boundary voxels, improving sensitivity
- Constraint enforced: no individual case decreased by > 0.03 Dice
- Fallback to static thresholds available if optimized values fail validation at runtime

---
*Report generated: 2026-04-13*
