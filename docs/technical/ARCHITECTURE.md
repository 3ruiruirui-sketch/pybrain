# Architecture

## Overview

PY-BRAIN is a modular Python pipeline for brain tumor segmentation from
multi-modal MRI. The system is organized as a Python package (`pybrain/`)
plus a suite of pipeline scripts (`scripts/`).

```
PY-BRAIN/
├── pybrain/              # Python package
│   ├── config/           # Configuration management
│   ├── core/             # Core utilities (metrics, normalization, brainmask)
│   ├── io/               # I/O (NIfTI, session, logging)
│   ├── models/           # Model definitions and ensembles
│   └── clinical/         # Clinical rules and WHO grading
├── scripts/              # Pipeline stage scripts (numbered 0-12)
├── config/               # Legacy root-level config
├── models/               # Model weights and calibration data
│   ├── calibration/      # Platt coefficients, ensemble weights
│   └── brats_bundle/     # MONAI pre-trained model weights
└── docs/                 # Documentation
```

---

## Pipeline Stages

The pipeline executes in numbered stages:

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► ...
             (DICOM    (CT       (AI        (Manual
              →NIfTI)  Regis-    Segm.)     Review)
                       tration)
                                            Stage 5 ──► Stage 6 ──► ...
                                            (Validate) (Location)
```

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `0_preflight_check.py` | Verify environment, paths, model availability |
| 1 | `1_dicom_to_nifti.py` | Convert DICOM MRI series to NIfTI format |
| 1b | `1b_brats_preproc.py` | BraTS-format preprocessing |
| 2 | `2_ct_integration.py` | Register CT, generate calcification prior |
| **3** | **`3_brain_tumor_analysis.py`** | **AI segmentation (ensemble inference)** |
| 5 | `5_validate_segmentation.py` | Dice, HD95, ASD vs ground truth |
| 6 | `6_tumour_location.py` | Anatomical atlas location |
| 7 | `7_tumour_morphology.py` | Volume, surface area, compactness |
| 8 | `8_radiomics_analysis.py` | Radiomics + XGBoost IDH classifier |
| 9 | `9_generate_report.py` | PDF report generation |

---

## Core Module: `pybrain/`

### `pybrain/config/`

Manages all pipeline configuration via YAML files.

| File | Purpose |
|------|---------|
| `defaults.yaml` | Master thresholds, ensemble weights, model configs |
| `hardware_profiles.yaml` | Device configs (MPS/CUDA/CPU) |
| `config.py` | Config loader with environment variable overrides |

### `pybrain/core/`

| File | Purpose |
|------|---------|
| `metrics.py` | Dice, HD95, ASD, VolumeDiff implementations |
| `normalization.py` | MRI histogram normalization, intensity clipping |
| `brainmask.py` | Skull-stripping utilities |
| `session.py` | Session tracking and artifact management |

### `pybrain/io/`

| File | Purpose |
|------|---------|
| `nifti_io.py` | NIfTI read/write with compression |
| `session.py` | Session and output directory management |
| `logging_utils.py` | Structured logging with `rich` |
| `config.py` | Config serialization helpers |

### `pybrain/models/`

| File | Purpose |
|------|---------|
| `segresnet.py` | SegResNet MONAI model wrapper |
| `swinunetr.py` | SwinUNETR MONAI model wrapper |
| `mc_dropout.py` | MC-Dropout wrapper for uncertainty |
| `ensemble.py` | Weighted probability averaging |
| `subregion_ensemble.py` | Per-subregion ensemble weighting |
| `staple_ensemble.py` | STAPLE algorithm fusion |
| `fusion_idh.py` | IDH mutation prediction from segmentation |

### `pybrain/clinical/`

| File | Purpose |
|------|---------|
| `consistency.py` | Hierarchical consistency checks (WT⊇TC⊇ET) |
| `genomics.py` | IDH mutation prediction from radiomics |
| `who_rules.py` | WHO CNS tumor grading rules |

---

## Stage 3 — AI Segmentation Architecture

Stage 3 is the core inference step. Its internal flow:

```
Input MRI (T1, T1c, T2, FLAIR)
    │
    ▼
ROI Localization (whole-brain crop to ROI)
    │
    ▼
┌─────────────────────────────────────────┐
│         Standard Inference Path         │
│  SegResNet ──────────────────────► probs_std │
│  TTA-4 ──────────────────────────► probs_tta │
│                                         │
│  Ensemble: 0.6*probs_std + 0.4*probs_tta │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      MC-Dropout Path (if enabled)        │
│  SegResNet × 15 forward passes          │
│  → mean_probs + std_map                  │
│  Platt calibration on mean_probs         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│         CT Boost (solid tissue prior)    │
│  If CT available: HU[35-75] → boost     │
└─────────────────────────────────────────┘
    │
    ▼
Per-subregion thresholding (WT/TC/ET)
    │
    ▼
Hierarchical consistency enforcement
    │
    ▼
Output: segmentation_full.nii.gz
        probabilities_*.nii.gz
        uncertainty_*.nii.gz (if MC-Dropout)
```

---

## Ensemble Strategy

The ensemble uses a **fixed-weight linear combination** of model
probabilities:

```
P_final = 0.60 × P_SegResNet + 0.40 × P_TTA-4
```

Where `P_TTA-4` is the average of 4 test-time augmented predictions
(axis-aligned flips).

SwinUNETR and nnU-Net are currently weight-0 (disabled).

---

## Calibration

### Platt Scaling

Temperature scaling per subregion (WT, TC, ET):

```
P_calibrated = sigmoid((logits - t) / T)
```

Where `T` (temperature) and `t` (bias offset) are fitted per-subregion
on validation data.

### MC-Dropout

Epistemic uncertainty is estimated via the variance of 15 stochastic
forward passes with dropout enabled:

```
σ²(x) ≈ (1/M) Σᵢ p(x|ωᵢ)² - [ (1/M) Σᵢ p(x|ωᵢ) ]²
```

---

## Configuration Hierarchy

```
pybrain/config/defaults.yaml    ← base configuration
    │
    ├── environment variables  ← PYBRAIN_THRESH_WT, etc. (overrides)
    │
    └── script arguments        ← command-line overrides
```

---

## Session System

Each pipeline run creates a session:

```bash
export PYBRAIN_SESSION="patient001_baseline"
```

Results are stored in `results/{PYBRAIN_SESSION}/` with the following
structure:

```
results/{session}/
├── segmentation_full.nii.gz      # Final segmentation
├── probabilities_wt.nii.gz       # WT probability map
├── probabilities_tc.nii.gz       # TC probability map
├── probabilities_et.nii.gz       # ET probability map
├── uncertainty_mc.nii.gz         # MC-Dropout std (if enabled)
├── tumor_stats.json               # Volume and metric summary
├── platt_calibration.json        # Calibration metadata
└── report_{session}.pdf          # Generated PDF report
```

---

## Dependencies

```
# Core
torch>=2.0.0
torchvision
monai>=1.2.0

# I/O
pydicom
nibabel
SimpleITK
pynrrd

# Processing
scikit-image
scipy
scikit-learn
pyradiomics

# Visualization / Reports
matplotlib
plotly
reportlab
rich

# Utilities
pandas
numpy<2.0.0
tqdm
PyYAML
python-dotenv
```
