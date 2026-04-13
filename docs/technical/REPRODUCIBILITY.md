# Reproducibility Guide

This document describes how to reproduce results from the PY-BRAIN
brain tumor segmentation pipeline.

---

## Environment

### Hardware

| Component | Specification |
|-----------|--------------|
| Platform | macOS Apple Silicon (M4 Pro tested) |
| RAM | 36 GB (16 GB minimum) |
| Disk | ~6 GB for models + results |
| Compute | Apple MPS (Metal Performance Shaders) |

> **Note**: The pipeline is developed and tested on Apple Silicon Mac.
> NVIDIA GPU support is available via PyTorch CUDA when MONAI bundles are
> downloaded on a CUDA machine, but this is not actively tested.

### Software

| Component | Version |
|-----------|---------|
| Python | 3.9+ (3.9 recommended) |
| PyTorch | >= 2.0.0 |
| MONAI | >= 1.2.0 |
| Operating System | macOS 25.x, Linux (CUDA) |

### Key Dependencies

```
torch>=2.0.0
torchvision
monai>=1.2.0
einops
pydicom
nibabel
SimpleITK
pynrrd
scikit-image
scipy
pyradiomics
scikit-learn
pandas
numpy<2.0.0
reportlab
matplotlib
plotly
rich
tqdm
PyYAML
python-dotenv
```

Install with: `pip install -r requirements.txt`

---

## Random Seeds

The pipeline uses `PYBRAIN_SEED` for deterministic behavior:

```bash
export PYBRAIN_SEED=42
```

For fully reproducible results, also set:

```bash
export PYBRAIN_SESSION="reproducible_run_001"
```

> **Known limitations**: MC-Dropout samples (`n_samples=15`) introduce
> stochasticity. Set `mc_dropout.enabled: false` in `pybrain/config/defaults.yaml`
> for deterministic inference.

---

## Expected Folder Structure

```
PY-BRAIN/                          # Project root
├── pybrain/                        # Python package
│   └── config/defaults.yaml       # Main configuration
├── scripts/
│   ├── 3_brain_tumor_analysis.py   # Main segmentation entry point
│   └── ...
├── models/
│   ├── brats_bundle/              # MONAI BraTS model weights (~500 MB)
│   │   ├── folds/                 # SwinUNETR fold weights (5 folds)
│   │   └── *.pth
│   └── calibration/               # Platt calibration coefficients
│       ├── ensemble_weights.json
│       ├── per_model_dice.json
│       └── platt_coefficients.json
├── data/
│   └── datasets/BraTS2021/        # BraTS dataset (user-downloaded)
│       ├── BraTS2021_00000/
│       │   ├── BraTS2021_00000_t1.nii.gz
│       │   ├── BraTS2021_00000_t1ce.nii.gz
│       │   ├── BraTS2021_00000_t2.nii.gz
│       │   └── BraTS2021_00000_flair.nii.gz
│       └── BraTS2021_00002/
│           └── ...
└── results/                       # Pipeline outputs (git-ignored)
    └── <session_id>/
        ├── segmentation_full.nii.gz
        ├── tumor_stats.json
        └── ...
```

---

## Data Acquisition

The pipeline requires BraTS2021 multi-modal MRI data. See [docs/DATA.md](./DATA.md)
for legal acquisition instructions.

Minimum required files per case:
- `*_t1.nii.gz` — T1-weighted
- `*_t1ce.nii.gz` — T1-weighted with contrast agent
- `*_t2.nii.gz` — T2-weighted
- `*_flair.nii.gz` — FLAIR

Optional ground truth for validation:
- `*_seg.nii.gz` — BraTS label map (labels: 0=background, 1=NCR/NET, 2=ED, 4=ET)

---

## Command Examples

### Reproduce Segmentation on a Single Case

```bash
# Set environment
export PYBRAIN_SEED=42
export PYBRAIN_SESSION="BraTS2021_00000_repro"

# Run main segmentation
python scripts/3_brain_tumor_analysis.py \
    --input-dir data/datasets/BraTS2021/BraTS2021_00000 \
    --output-dir results/BraTS2021_00000_repro \
    --session-id BraTS2021_00000_repro
```

### Reproduce Full Pipeline

```bash
# Stage 1: Convert DICOM to NIfTI
python scripts/1_dicom_to_nifti.py

# Stage 3: AI segmentation
python scripts/3_brain_tumor_analysis.py

# Stage 5: Validate
python scripts/5_validate_segmentation.py
```

### Reproduce with Different Thresholds

Edit `pybrain/config/defaults.yaml`:

```yaml
thresholds:
  wt: 0.45   # Whole Tumor
  tc: 0.35   # Tumor Core
  et: 0.35   # Enhancing Tumor
```

Then re-run Stage 3 without re-downloading models.

---

## Verification Checklist

To verify reproducibility:

- [ ] Same `PYBRAIN_SEED` produces identical segmentation (hash output)
- [ ] Model weights unchanged (verify checksums)
- [ ] Same BraTS case version used
- [ ] Same `pybrain/config/defaults.yaml` configuration
- [ ] Same Python environment (pip hash of requirements.txt)

---

## Limitations

1. **Apple MPS non-determinism**: Some PyTorch MPS operations are
   non-deterministic. For strict determinism, use CUDA on Linux.

2. **MC-Dropout stochasticity**: Each run produces slightly different
   uncertainty maps. Disable MC-Dropout for deterministic inference.

3. **Model downloads**: MONAI bundles are downloaded on first run.
   Hash verification is not performed automatically.

4. **OS-level differences**: Minor numerical differences may occur between
   macOS MPS and CUDA due to floating-point implementation differences.

5. **Software versions**: MONAI and PyTorch updates may affect numerical
   results slightly. Pin exact versions for strict reproducibility.
