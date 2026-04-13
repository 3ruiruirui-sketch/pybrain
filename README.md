# PY-BRAIN — Brain Tumour Analysis Pipeline

AI-assisted brain tumour segmentation, analysis and reporting
from DICOM MRI + CT images using MONAI BraTS pre-trained models.

---

## Directory Structure

```
PY-BRAIN/                           ← PROJECT ROOT
│
├── pybrain/                         ← Python package
│   ├── clinical/                    ← Clinical rules, WHO grading, consistency checks
│   ├── config/                     ← Configuration loaders
│   ├── core/                       ← Core utilities (metrics, normalization, brainmask)
│   ├── io/                         ← I/O utilities (NIfTI, session, logging)
│   └── models/                     ← Model definitions (SwinUNETR, SegResNet, ensemble)
│
├── scripts/                         ← Pipeline and utility scripts
│   ├── 0_download_sample.py        ← Download sample BraTS data
│   ├── 0_preflight_check.py       ← Pre-flight checks before pipeline run
│   ├── 1_dicom_to_nifti.py          ← DICOM → NIfTI conversion
│   ├── 1b_brats_preproc.py         ← BraTS-specific preprocessing
│   ├── 2_ct_integration.py         ← CT registration + calcification masks
│   ├── 3_brain_tumor_analysis.py    ← AI tumour segmentation (MAIN)
│   ├── 5_validate_segmentation.py  ← Validation metrics
│   ├── 6_tumour_location.py        ← Tumour location analysis
│   ├── 6_finetune_swinunetr.py     ← Fine-tune SwinUNETR on custom data
│   ├── 7_tumour_morphology.py       ← Morphology analysis
│   ├── 8_radiomics_analysis.py      ← Radiomics + ML classification
│   ├── 8b_brainiac_prediction.py   ← BrainIAC prediction
│   ├── 9_generate_report.py        ← PDF report generation
│   ├── 9_generate_report_pt.py      ← PDF report (PyTorch version)
│   ├── 10_enhanced_visualisation.py ← Enhanced 3D visualisation
│   ├── 11_mricrogl_visualisation.py ← MRICroGL visualisation
│   ├── 11b_mricrogl_interactive.py  ← Interactive MRICroGL viewer
│   ├── 12_brats_figure1.py         ← Generate Figure 1 for BraTS paper
│   ├── benchmark_swin.py           ← Benchmark SwinUNETR performance
│   ├── check_model.py              ← Model integrity checker
│   ├── fetch_brats_sample.py       ← Fetch BraTS sample data
│   ├── session_loader.py           ← Session loading utilities
│   ├── test_pipeline_integration.py← Integration tests
│   ├── train_gwo_selector.py       ← Train GWO feature selector
│   ├── train_xgboost.py            ← Train XGBoost classifier
│   ├── utils.py                    ← Shared helpers
│   └── verify_clinical_alignment.py ← Verify clinical alignment
│
├── config/                          ← Configuration files
│   └── config.py                    ← ALL paths and settings
│
├── models/                          ← Model files and bundles
│   └── brats_bundle/               ← auto-downloaded on first run (~500 MB)
│
├── data/                            ← Datasets
│   └── datasets/BraTS2021/         ← BraTS 2021 training data
│
├── nifti/                           ← NIfTI files (auto-filled by pipeline)
│   ├── monai_ready/                ← t1.nii.gz, t1c_resampled.nii.gz,
│   │                                t2_resampled.nii.gz, flair_resampled.nii.gz
│   └── extra_sequences/            ← dwi, adc, t2star, ct_brain_registered, ...
│
├── results/                         ← All analysis outputs
│   ├── ground_truth.nii.gz         ← manual correction (external editor)
│   └── results_TIMESTAMP/          ← one folder per pipeline run
│       ├── segmentation_full.nii.gz
│       ├── view_axial/coronal/sagittal.png
│       ├── interactive_viewer.html
│       ├── tumor_stats.json
│       ├── radiomics_features.json
│       ├── morphology.json
│       ├── tumour_location.json
│       └── report_TIMESTAMP.pdf
│
├── input_data/                     ← Input DICOM files
├── output/                         ← Pipeline output
├── cache/                          ← Cached intermediate results
├── logs/                           ← Run logs
├── notebooks/                      ← Jupyter exploration
├── tests/                          ← Unit tests
├── docs/                           ← Documentation
│
├── .venv/                          ← Python virtual environment
├── fusion_model.pt                 ← Fusion model weights
├── run_pipeline.py                 ← Main pipeline runner
├── auto_run.py                     ← Automated pipeline runner
├── translate_report.py             ← Translate reports
├── validate_v7_update.py           ← Validation updates
├── patch_script.py                 ← Patching utilities
├── setup_mock_data.py              ← Mock data setup
├── requirements.txt
├── setup_project.sh
├── portable_setup.sh
├── PORTABILITY_GUIDE.md
└── README.md
```

---

## Modelos Atualmente Ativos

### ✅ Ativos e Validados

| Modelo | Status | Peso Ensemble | Validação |
|--------|--------|---------------|-----------|
| **SegResNet** | ✅ Ativo | 0.60 | Dice WT ≈ 0.93 em BraTS2021_00000 |
| **TTA-4** | ✅ Ativo | 0.40 | Noise reduction via test-time augmentation |
| **Platt Calibration** | ✅ Ativo | — | Calibração de probabilidades por subregião |
| **MC-Dropout** | ✅ Ativo | — | Incerteza epistémica para SegResNet (15 samples) |

### ⏸️ Inativos por Design

| Modelo | Status | Motivo | Próximos Passos |
|--------|--------|--------|-----------------|
| **SwinUNETR** | ⏸️ Desativado | Aguarda calibração Platt | 1. Correr `compute_platt_calibration.py` sem `--skip-swinunetr`<br>2. Verificar ordem canais (TC/WT/ET vs WT/TC/ET)<br>3. Ajustar `ensemble_weights.swinunetr` para ~0.30 |
| **nnU-Net (DynUNet)** | ⏸️ Desativado | Sem pesos pré-treinados | Não treinar no Mac mini — usar GPU server quando pesos disponíveis |

### Configuração Atual

Ver `pybrain/config/defaults.yaml`:
```yaml
ensemble_weights:
  segresnet: 0.60   # Primary
  tta4: 0.40        # Secondary
  swinunetr: 0.0    # DISABLED: awaiting calibration
  nnunet: 0.0       # DISABLED: no pretrained weights

models:
  nnunet:
    enabled: false  # PERMANENTLY DISABLED until weights available
  mc_dropout:
    enabled: true   # Uncertainty quantification for SegResNet
    n_samples: 15
```

---

## Quick Start

```bash
# 1. Install dependencies
chmod +x setup_project.sh
./setup_project.sh

# 2. Activate venv
source .venv/bin/activate

# 3. Run the full pipeline
python3 run_pipeline.py
```

### PyCharm Setup
```
File → Open → /Users/ssoares/Downloads/PY-BRAIN
Settings → Python Interpreter → Add Interpreter
  → Existing environment
  → Path: /Users/ssoares/Downloads/PY-BRAIN/.venv/bin/python3
Settings → Run/Debug → Working directory: /Users/ssoares/Downloads/PY-BRAIN
```

---

## Pipeline — Stage Run Order

```bash
# Activate venv first
source .venv/bin/activate

# Stage 0 — Preflight checks
python3 scripts/0_preflight_check.py

# Stage 1 — DICOM → NIfTI (~5 min)
python3 scripts/1_dicom_to_nifti.py

# Stage 2 — CT registration + calcification masks (~5 min)
python3 scripts/2_ct_integration.py

# Stage 3 — AI tumour segmentation (~90 sec)
python3 scripts/3_brain_tumor_analysis.py

# Stage 4 — Manual review (optional)
#   Open the segmentation in any medical image viewer or segment editor.
#   Correct the mask if needed → save as results/ground_truth.nii.gz

# Stage 5 — Validation metrics (~1 min)
python3 scripts/5_validate_segmentation.py

# Stage 6 — Location analysis (~2 min)
python3 scripts/6_tumour_location.py

# Stage 7 — Morphology analysis (~3 min)
python3 scripts/7_tumour_morphology.py

# Stage 8 — Radiomics + ML classification (~5 min)
python3 scripts/8_radiomics_analysis.py

# Stage 9 — PDF report (~1 min)
python3 scripts/9_generate_report.py
```

Or run the full pipeline at once:
```bash
python3 run_pipeline.py
```

---

## Configuration

Edit `config/config.py` for each new patient:
- `PATIENT` dict — name, DOB, exam date, radiologist reference
- `WT_THRESH / TC_THRESH / ET_THRESH` — segmentation thresholds

All data paths auto-resolve from `PROJECT_ROOT` — no manual path editing needed.

---

## Requirements

- macOS Apple Silicon M-series (M1/M2/M3/M4)
- Python 3.9+ (3.9 recommended)
- 16 GB RAM minimum (32 GB recommended)
- ~6 GB disk space (model bundle + results)
- Internet access for first run (model download)

---

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Main pipeline orchestrator — runs all stages |
| `auto_run.py` | Automated pipeline runner |
| `3_brain_tumor_analysis.py` | Core AI segmentation using MONAI BraTS models |
| `8_radiomics_analysis.py` | Radiomics feature extraction + ML classification |
| `9_generate_report.py` | PDF report generation |
| `translate_report.py` | Report translation utilities |
| `train_xgboost.py` | Train XGBoost classifier |
| `6_finetune_swinunetr.py` | Fine-tune SwinUNETR on custom data |
| `verify_clinical_alignment.py` | Verify clinical alignment |

---

## ⚠️ Disclaimer

**FOR RESEARCH USE ONLY.**
Not a clinical diagnostic tool. All AI findings must be verified
by a qualified radiologist before any clinical decision is made.
