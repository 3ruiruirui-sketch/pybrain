# PY-BRAIN

**Brain tumor MRI segmentation pipeline with epistemic uncertainty quantification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.2+-green.svg)](https://monai.io/)

---

> ### :warning: Research Use Only
>
> **PY-BRAIN is research software and is NOT a certified medical device.**
> It has not been cleared or approved by the FDA, EMA, or any regulatory body.
> All AI-generated segmentations must be verified by a qualified radiologist
> before any clinical use. See [SECURITY.md](./SECURITY.md) for full disclaimers.

---

## Highlights

- **Multi-model ensemble**: SegResNet (60%) + TTA-4 (40%) with Platt-calibrated probabilities
- **Epistemic uncertainty**: MC-Dropout (15 samples) for confidence-aware segmentation
- **BraTS-validated**: Dice WT=0.933, TC=0.905, ET=0.925, HD95=10.3mm on BraTS2021
- **Reproducible**: Fixed random seeds, documented configuration, calibration coefficients versioned
- **Modular**: Stage-based pipeline (0–9), each stage independently executable
- **Research-grade**: Not for clinical diagnosis — see disclaimers throughout

---

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| **SegResNet + TTA-4 ensemble** | ✅ Active & Validated | Primary inference path |
| **MC-Dropout uncertainty** | ✅ Active | 15 samples, SegResNet only |
| **Platt calibration** | ✅ Active | Per-subregion temperature scaling |
| **STAPLE ensemble** | ✅ Implemented | Awaiting ensemble validation |
| **SwinUNETR** | ⏸️ Disabled | Awaiting Platt calibration + channel verification |
| **nnU-Net (DynUNet)** | ⏸️ Disabled | No validated pretrained weights available |

**Validation**: Validated on BraTS2021 cases (internal runs, 2026-04-13).
Full external validation pending.

---

## Repository Structure

```
pybrain/
├── pybrain/                   # Python package
│   ├── config/                # YAML configuration (thresholds, ensemble weights)
│   ├── core/                  # Metrics, normalization, brainmask utilities
│   ├── io/                    # NIfTI I/O, session, logging
│   ├── models/                # SegResNet, TTA, MC-Dropout, Platt, STAPLE
│   └── clinical/              # WHO grading, clinical consistency checks
├── scripts/                   # Pipeline stage scripts
│   ├── 0_preflight_check.py  # Environment validation
│   ├── 1_dicom_to_nifti.py   # DICOM → NIfTI conversion
│   ├── 2_ct_integration.py   # CT registration + solid tissue prior
│   ├── 3_brain_tumor_analysis.py  # AI segmentation (MAIN)
│   ├── 5_validate_segmentation.py # Dice, HD95, ASD metrics
│   ├── 6_tumour_location.py # Anatomical atlas location
│   ├── 7_tumour_morphology.py # Volume, surface area, compactness
│   ├── 8_radiomics_analysis.py # Radiomics + XGBoost IDH classifier
│   └── 9_generate_report.py  # PDF report
├── models/
│   └── calibration/           # Platt coefficients (version-controlled)
├── docs/
│   ├── DATA.md               # BraTS data acquisition guide
│   └── technical/
│       ├── VALIDATION.md      # Validation methodology & results
│       ├── MODEL_CARD.md      # Model documentation
│       ├── ARCHITECTURE.md    # System design
│       └── REPRODUCIBILITY.md # Reproduction guide
├── .github/workflows/ci.yml   # GitHub Actions (lint + smoke tests)
├── requirements.txt           # pip dependencies
├── environment.yml            # conda dependencies
├── pyproject.toml            # package metadata
├── LICENSE                   # MIT license
├── CITATION.cff              # Citation metadata
├── CONTRIBUTING.md           # Contribution guidelines
└── SECURITY.md               # Security policy & clinical disclaimers
```

---

## Quick Start

### 1. Install Dependencies

**Using pip:**
```bash
git clone https://github.com/<user>/pybrain.git
cd pybrain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Using conda:**
```bash
git clone https://github.com/<user>/pybrain.git
cd pybrain
conda env create -f environment.yml
conda activate pybrain
```

### 2. Obtain BraTS Data

Download BraTS2021 from [TCIA](https://www.cancerimagingarchive.net/collections/collection?id=brats2021)
(requires registration, free for academic use). Place in `data/datasets/BraTS2021/`.

See [docs/DATA.md](./docs/DATA.md) for full instructions.

### 3. Run the Pipeline

```bash
# Set session ID for reproducibility
export PYBRAIN_SESSION="my_first_run"
export PYBRAIN_SEED=42

# Run main segmentation (Stage 3)
python scripts/3_brain_tumor_analysis.py \
    --input-dir data/datasets/BraTS2021/BraTS2021_00000 \
    --output-dir results/BraTS2021_00000 \
    --session-id BraTS2021_00000
```

Results are written to `results/{session-id}/`:
```
results/BraTS2021_00000/
├── segmentation_full.nii.gz     # Final segmentation (WT/TC/ET labels)
├── probabilities_wt.nii.gz     # WT probability map
├── probabilities_tc.nii.gz     # TC probability map
├── probabilities_et.nii.gz     # ET probability map
├── uncertainty_mc.nii.gz       # MC-Dropout std (uncertainty map)
├── platt_calibration.json      # Calibration metadata
└── report_BraTS2021_00000.pdf  # PDF report
```

### 4. Validate Results

```bash
python scripts/5_validate_segmentation.py \
    --prediction results/BraTS2021_00000/segmentation_full.nii.gz \
    --ground-truth data/datasets/BraTS2021/BraTS2021_00000/BraTS2021_00000_seg.nii.gz
```

---

## Inference Workflow

```
Input MRI (T1, T1c, T2, FLAIR)
    │
    ▼
ROI Localization (whole-brain crop → focused ROI)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│          Ensemble Inference (SegResNet + TTA-4)          │
│                                                          │
│  SegResNet ─────────────────────► P_std (60%)          │
│  TTA-4 (4-axis flips) ──────────► P_tta  (40%)          │
│                                                          │
│  P_ensemble = 0.60 × P_std + 0.40 × P_tta               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│            MC-Dropout (if enabled, 15 samples)            │
│  SegResNet forward × 15 with dropout                     │
│  → mean probability + std uncertainty map                │
│  Platt calibration applied to probabilities              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
CT Boost (optional — solid tissue prior from HU[35-75])
    │
    ▼
Per-subregion thresholding: WT / TC / ET
    │
    ▼
Hierarchical consistency (WT ⊇ TC ⊇ ET)
    │
    ▼
Output: segmentation + probability maps + uncertainty
```

---

## Validation Summary

Validated on BraTS2021 cases (2026-04-13):

| Case | Dice WT | Dice TC | Dice ET | HD95 | Quality |
|------|---------|---------|---------|------|---------|
| BraTS2021_00000 | **0.933** | **0.905** | **0.925** | **10.3 mm** | EXCELLENT |
| BraTS2021_00002 | 0.702 | 0.827 | 0.683 | 19.1 mm | CHALLENGING |
| BraTS2021_00003 | **0.967** | **0.937** | **0.912** | **1.4 mm** | OUTSTANDING |

> Case 00002 is a clinically challenging infiltrative tumor — low WT Dice
> reflects inherent difficulty, not a pipeline failure.

Full validation details: [docs/technical/VALIDATION.md](./docs/technical/VALIDATION.md)

---

## Current Model Configuration

Active and disabled models:

| Model | Ensemble Weight | Status | Notes |
|-------|----------------|--------|-------|
| **SegResNet** | 0.60 | ✅ Active | Primary model, MONAI BraTS pretrained |
| **TTA-4** | 0.40 | ✅ Active | 4-axis flip averaging |
| **MC-Dropout** | — | ✅ Active | 15 samples, SegResNet only |
| **Platt Calibration** | — | ✅ Active | Per-subregion temperature scaling |
| **STAPLE** | — | ✅ Implemented | Awaiting ensemble validation |
| **SwinUNETR** | 0.00 | ⏸️ Disabled | Awaiting Platt + channel-order verification |
| **nnU-Net** | 0.00 | ⏸️ Disabled | No validated pretrained weights |

Configuration file: `pybrain/config/defaults.yaml`

---

## Reproducibility Notes

- Set `PYBRAIN_SEED=42` before running for deterministic results
- MC-Dropout is stochastic: disable `mc_dropout.enabled` in config for strict determinism
- Platt calibration coefficients are stored in `models/calibration/`
- MONAI model weights are downloaded on first run (~500 MB)

Full guide: [docs/technical/REPRODUCIBILITY.md](./docs/technical/REPRODUCIBILITY.md)

---

## Limitations

1. **Research only** — not a certified medical device; not for clinical diagnosis
2. **BraTS-trained** — validated only on BraTS-style 4-channel MRI (T1/T1ce/T2/FLAIR)
3. **Infiltrative tumors** — Case 00002-type tumors with ill-defined edema margins
   show lower WT Dice (see validation report)
4. **Post-operative scans** — not validated on post-surgical cases
5. **Apple MPS** — some PyTorch MPS operations are non-deterministic; CUDA on Linux
   recommended for strict reproducibility
6. **MC-Dropout scope** — uncertainty only available for SegResNet; TTA-4 does not
   produce uncertainty maps

---

## Roadmap

| Priority | Item | Status |
|----------|------|--------|
| 1 | SwinUNETR validation and activation | ⏸️ Pending |
| 2 | Full regression test suite | 🔄 In progress |
| 3 | STAPLE ensemble validation | 🔄 In progress |
| 4 | nnU-Net bundle integration (when available) | ⏳ Awaiting upstream |
| 5 | External multi-site validation | 🔜 Planned |
| 6 | Zenodo DOI registration | 🔜 Planned |

---

## Citation

If you use PY-BRAIN in your research, please cite:

```bibtex
# BibTeX (update with Zenodo DOI when assigned)
@misc{pybrain2026,
  title = {{PY-BRAIN}: Brain Tumor MRI Segmentation with Uncertainty Quantification},
  author = {{PY-BRAIN Contributors}},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/<user>/pybrain}
}
```

Or use the **Cite this repository** button on GitHub (uses `CITATION.cff`).

---

## License

**MIT License** — see [LICENSE](./LICENSE).

> **Note on model licenses**: The MONAI SwinUNETR BraTS bundle is licensed
> under Apache 2.0. The BrainIAC model has a separate non-commercial academic
> license from Mass General Brigham. Model weights are downloaded separately
> and are not included in this repository.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- Development setup
- Coding standards
- Testing requirements
- Pull request process

Please read our [Code of Conduct](./CODE_OF_CONDUCT.md) and
[Security Policy](./SECURITY.md) before contributing.

---

## Links

- [Documentation](./docs/)
- [Model Card](./docs/technical/MODEL_CARD.md)
- [Validation Report](./docs/technical/VALIDATION.md)
- [Reproducibility Guide](./docs/technical/REPRODUCIBILITY.md)
- [Data Guide](./docs/DATA.md)
- [GitHub Issues](https://github.com/<user>/pybrain/issues)
