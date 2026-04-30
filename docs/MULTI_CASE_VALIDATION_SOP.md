# Multi-Case Validation Standard Operating Procedure (SOP)

**Document ID:** PYBRAIN-VAL-SOP-001  
**Version:** 1.0  
**Date:** 2026-04-13  
**Project:** PY-BRAIN Brain Tumor Segmentation Pipeline

---

## 1. Purpose and Scope

### 1.1 Objective
This document establishes the standard operating procedure for conducting reproducible, multi-case validation studies of the PY-BRAIN segmentation pipeline using the BraTS 2021 dataset.

### 1.2 Scope
- Automated execution on multiple BraTS cases
- Metric aggregation and statistical analysis
- Professional documentation for scientific publication
- Reproducible workflow suitable for GitHub/research papers

### 1.3 Target Audience
- Clinical researchers validating segmentation algorithms
- Developers conducting regression testing
- Reviewers assessing pipeline performance

---

## 2. Prerequisites

### 2.1 Hardware Requirements
- **CPU:** 4+ cores recommended
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 20GB free space for validation runs
- **GPU:** Apple Silicon (MPS) or CUDA-compatible GPU optional but recommended

### 2.2 Software Requirements
```bash
# Python environment
Python 3.10+
PyTorch 2.0+
MONAI 1.3+
nibabel, numpy, pandas, matplotlib

# Project structure
git clone <repository>
cd PY-BRAIN
source .venv/bin/activate  # or equivalent
```

### 2.3 Data Requirements
- BraTS 2021 Training Data (or subset)
- At least 3 cases for meaningful validation
- Ground truth segmentations available

---

## 3. Validation Workflow

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-CASE VALIDATION FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CONFIGURATION                                               │
│     └── Define case list, device, output paths                   │
│                                                                  │
│  2. PREPARATION (per case)                                      │
│     └── prepare_brats_case.py → session.json                     │
│                                                                  │
│  3. EXECUTION (per case)                                        │
│     └── 3_brain_tumor_analysis.py → outputs                      │
│                                                                  │
│  4. AGGREGATION                                                 │
│     └── Extract metrics from validation_metrics.json           │
│     └── Generate CSV, JSON, Markdown reports                   │
│                                                                  │
│  5. VISUALIZATION                                               │
│     └── Generate summary plots and figures                     │
│                                                                  │
│  6. DOCUMENTATION                                               │
│     └── Scientific report ready for publication                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Procedure

### Step 1: Environment Setup

```bash
# Navigate to project root
cd /Users/ssoares/Downloads/PY-BRAIN

# Activate virtual environment
source .venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
```

### Step 2: Configure Validation Run

Edit validation parameters in the script call:

```bash
# Option A: Specific cases
python scripts/run_multi_case_validation.py \
    --brats-root data/datasets/BraTS2021/raw/BraTS2021_Training_Data \
    --cases BraTS2021_00000 BraTS2021_00002 BraTS2021_00003 \
    --output-dir results/validation_runs \
    --summary-dir results/validation_summary \
    --device mps \
    --timeout 600

# Option B: Auto-discover first N cases
python scripts/run_multi_case_validation.py \
    --brats-root data/datasets/BraTS2021/raw/BraTS2021_Training_Data \
    --auto-discover 10 \
    --device mps
```

**Parameter Reference:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--brats-root` | Path to BraTS data directory | Required |
| `--cases` | List of case IDs | Required if not using --auto-discover |
| `--auto-discover` | Auto-select N cases | - |
| `--device` | Compute device (cpu/cuda/mps) | mps |
| `--timeout` | Seconds per case | 600 |
| `--skip-existing` | Skip if outputs exist | False |
| `--continue-on-error` | Continue if case fails | True |

### Step 3: Execute Validation

```bash
# Run with logging
time python scripts/run_multi_case_validation.py \
    --brats-root data/datasets/BraTS2021/raw/BraTS2021_Training_Data \
    --cases BraTS2021_00000 BraTS2021_00002 BraTS2021_00003 \
    2>&1 | tee validation_run_$(date +%Y%m%d_%H%M%S).log
```

**Expected Runtime:**
- Per case: ~40-60 seconds (with MC-Dropout)
- 3 cases: ~3-5 minutes total

### Step 4: Verify Outputs

```bash
# Check directory structure
ls -R results/validation_runs/

# Expected structure:
# results/validation_runs/
# ├── BraTS2021_00000/
# │   ├── session.json
# │   ├── segmentation_full.nii.gz
# │   ├── validation_metrics.json
# │   └── run.log
# ├── BraTS2021_00002/
# └── BraTS2021_00003/
```

### Step 5: Generate Summary Reports

```bash
# The validation script generates these automatically:
# - results/validation_summary/multi_case_metrics.csv
# - results/validation_summary/multi_case_summary.json
# - results/validation_summary/MULTI_CASE_VALIDATION_REPORT.md

# Verify they exist:
ls -lh results/validation_summary/
```

### Step 6: Generate Visualizations

```bash
# Create visualization plots
python scripts/plot_validation_summary.py \
    --csv results/validation_summary/multi_case_metrics.csv \
    --output-dir results/validation_summary/figures \
    --dpi 300

# Expected outputs:
# results/validation_summary/figures/
# ├── dice_summary.png
# ├── hd95_summary.png
# ├── volume_diff_summary.png
# └── overall_summary.png
```

---

## 5. Output Specifications

### 5.1 CSV Format: `multi_case_metrics.csv`

| Column | Type | Description |
|--------|------|-------------|
| `case_id` | string | BraTS case identifier |
| `status` | string | success/failed |
| `dice_wt` | float | Dice Whole Tumor (0-1) |
| `dice_tc` | float | Dice Tumor Core (0-1) |
| `dice_et` | float | Dice Enhancing Tumor (0-1) |
| `hd95_wt` | float | Hausdorff Distance 95th percentile (mm) |
| `asd` | float | Average Surface Distance (mm) |
| `volume_pred_cc` | float | Predicted volume (cm³) |
| `volume_gt_cc` | float | Ground truth volume (cm³) |
| `volume_diff_percent` | float | Volume difference (%) |
| `runtime_seconds` | float | Processing time |
| `error` | string | Error message if failed |

### 5.2 JSON Format: `multi_case_summary.json`

```json
{
  "n_cases": 3,
  "n_successful": 3,
  "n_failed": 0,
  "statistics": {
    "dice_wt": {
      "mean": 0.9320,
      "median": 0.9332,
      "std": 0.0156,
      "min": 0.7016,
      "max": 0.9650
    }
  },
  "cases": [...]
}
```

### 5.3 Markdown Report: `MULTI_CASE_VALIDATION_REPORT.md`

Contains:
- Executive summary
- Per-case results table
- Summary statistics
- Failure analysis
- Technical conclusions
- Recommendations

---

## 6. Quality Criteria

### 6.1 Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Dice WT | ≥ 0.90 | Excellent |
| Dice TC | ≥ 0.85 | Good |
| Dice ET | ≥ 0.80 | Acceptable |
| HD95 WT | ≤ 15 mm | Good |
| ASD | ≤ 2.5 mm | Good |
| Volume Diff | ≤ 15% | Acceptable |

### 6.2 Stability Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Dice WT Std | < 0.05 | Very stable |
| Dice WT Std | 0.05-0.10 | Stable |
| Dice WT Std | > 0.10 | Needs review |

### 6.3 Failure Handling

- **Partial Failure:** If 1 of 3 cases fails, evaluate the 2 successful cases
- **Complete Failure:** If >50% cases fail, investigate systemic issue
- **Error Logging:** All errors logged to `run.log` per case

---

## 7. Scientific Documentation

### 7.1 For Publication

**Methods Section Template:**

```markdown
## Validation

The PY-BRAIN pipeline was validated on N cases from the BraTS 2021 
training dataset. For each case, we computed:

- Dice Similarity Coefficient (DSC) for WT, TC, and ET
- 95th percentile Hausdorff Distance (HD95)
- Average Surface Distance (ASD)
- Volume difference (predicted vs. ground truth)

Multi-case validation was conducted using an automated workflow 
(run_multi_case_validation.py) ensuring reproducibility. Results 
were aggregated and analyzed for consistency across cases.
```

**Results Section Template:**

```markdown
## Results

The pipeline achieved mean Dice scores of:
- WT: 0.93 ± 0.02 (range: 0.90-0.97)
- TC: 0.89 ± 0.03 (range: 0.83-0.94)
- ET: 0.84 ± 0.05 (range: 0.68-0.92)

Mean HD95 was 10.4 ± 5.2 mm, indicating good boundary delineation.
Volume predictions were within 5.2 ± 3.8% of ground truth.
```

### 7.2 For GitHub Repository

Include in README:
- Link to validation results
- Instructions for reproducing
- Performance benchmarks
- Known limitations

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue:** `Case preparation failed`
- **Cause:** Missing required files (FLAIR, T1, T1ce, T2, seg)
- **Solution:** Verify BraTS case directory structure

**Issue:** `Pipeline timeout`
- **Cause:** Case too large or device too slow
- **Solution:** Increase `--timeout` or use faster device

**Issue:** `Segmentation shape mismatch`
- **Cause:** Inconsistent image dimensions
- **Solution:** Check input NIfTI files with `nib-ls`

### 8.2 Validation Checklist

- [ ] BraTS data directory exists and is accessible
- [ ] Virtual environment activated
- [ ] Sufficient disk space (>10GB free)
- [ ] GPU available (if using MPS/CUDA)
- [ ] Output directories writable
- [ ] All required Python packages installed

---

## 9. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-13 | Cascade | Initial SOP creation |

---

## 10. References

1. Menze et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE TMI.
2. Bakas et al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels.
3. PY-BRAIN Technical Documentation (2026).

---

**Document Control:**
- Review Date: 2026-07-13
- Approved By: [TBD]
- Distribution: Development Team, Clinical Partners
