# Data — BraTS Dataset Acquisition

## Overview

PY-BRAIN uses the **BraTS2021** multi-modal brain tumor MRI dataset for
algorithm development and validation. This document explains how to obtain
the data legally and how to set it up for use with the pipeline.

> **⚠️ IMPORTANT**: No patient data, DICOM files, or NIfTI images are
> included in this repository. You must download BraTS data separately
> following the steps below.

---

## What is BraTS?

The **Brain Tumor Segmentation (BraTS)** challenge provides multi-institutional
MRI scans with ground-truth annotations for brain tumor segmentation research.

BraTS2021 contains:
- **1,291 training cases** (with expert annotations)
- **219 validation cases** (annotations withheld)
- **530+ test cases** (labels withheld — for challenge submission)

Each case includes four MRI modalities:
| Modality | Description | Typical Timing |
|----------|-------------|----------------|
| T1 | T1-weighted | ~5 min |
| T1ce | T1 + contrast agent (gadolinium) | ~5 min post-injection |
| T2 | T2-weighted | ~5 min |
| FLAIR | Fluid-Attenuated Inversion Recovery | ~10 min |

**BraTS labels** (segmentation values in `*_seg.nii.gz`):
| Value | Label | Description |
|-------|-------|-------------|
| 0 | Background | Non-tumor |
| 1 | NCR/NET | Necrotic and non-enhancing tumor core |
| 2 | ED | Peritumoral edema |
| 4 | ET | GD-enhancing tumor |

---

## How to Download BraTS Data

### Step 1: Register at TCIA

BraTS data is hosted on **The Cancer Imaging Archive (TCIA)**.

1. Go to: https://www.cancerimagingarchive.net/
2. Click **Register** → create an account with your institutional email
3. Verify your email address

### Step 2: Request Access to BraTS Collection

1. Log in to TCIA
2. Navigate to: **Collections → BraTS2021**
   (https://www.cancerimagingarchive.net/collections/collection?id=brats2021)
3. Read and accept the **Data Usage Agreement**
4. Wait for access approval (typically 1-3 business days for academic users)

> **Note**: BraTS data is freely available for academic research.
> Commercial use requires a separate license agreement.

### Step 3: Download the Data

After approval:

1. Go to the BraTS2021 collection page
2. Select the **Training Data** (1,291 cases, ~26 GB compressed)
3. Download to your local machine
4. Extract the archive

```
BraTS2021_Training_Data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz  # Ground truth (training only)
├── BraTS2021_00001/
│   └── ...
```

---

## Setting Up the Data Directory

After downloading:

```bash
# Move the extracted data to the project data directory
mv /path/to/BraTS2021_Training_Data data/datasets/BraTS2021

# Verify structure
ls data/datasets/BraTS2021/BraTS2021_00000/
# Should show: t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz, seg.nii.gz
```

The pipeline auto-detects cases in `data/datasets/BraTS2021/` and
does not require manual registration of each case.

---

## Sample Data (for quick testing)

If you only want to test the pipeline without downloading 26 GB:

```bash
# Download a single sample case using the provided script
python scripts/0_download_sample.py
# or
python scripts/fetch_brats_sample.py
```

This downloads one annotated BraTS2021 case to `data/datasets/BraTS2021/`.

---

## Required vs Optional Files

### Required (for inference)

For segmentation without ground truth:
```
BraTS2021_XXXXX_t1.nii.gz
BraTS2021_XXXXX_t1ce.nii.gz
BraTS2021_XXXXX_t2.nii.gz
BraTS2021_XXXXX_flair.nii.gz
```

### Optional (for validation)

To compute validation metrics against ground truth:
```
BraTS2021_XXXXX_seg.nii.gz  # Expert annotations
```

---

## Legal and Ethical Notes

### Data License

BraTS data is released under the **Creative Commons
Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)** license.

**You may NOT:**
- Use the data for commercial purposes without separate agreement
- Distribute the data to third parties
- Attempt to identify patients from the imaging data
- Include BraTS data in public repositories

**You may:**
- Use for academic research and publication
- Use for algorithm development and validation
- Share results and derived segmentation masks

### GDPR / HIPAA Considerations

If working with data from European patients or US healthcare settings:

1. Ensure you have appropriate IRB/ethics approval
2. De-identify all DICOM data before processing (use `dcm2niix` with
   `-d` flag for de-identification)
3. Do not include any patient-identifiable information in research outputs
4. Consult your institution's data protection officer

### Synthetic Data

For development and CI testing without real patient data:

```bash
# Generate synthetic MRI-like volumes
python setup_mock_data.py
```

This creates artificial 4-channel MRI volumes in `data/mock/` that
follow the same file naming convention.

---

## Troubleshooting

### "No cases found in data/datasets/BraTS2021/"

The pipeline uses a specific file naming convention. Ensure:
- Directory name matches: `BraTS2021_XXXXX/` (not `BraTS2021_00000_something/`)
- All 4 modality files are present inside each case directory
- File extensions are exactly `.nii.gz`

### "Error loading NIfTI: not a valid NIfTI file"

This usually means the file was downloaded incompletely. Re-download
the specific case.

### "Expected 4 channels but got N"

Some BraTS cases have additional sequences (e.g., DWI, ADC). These
are optional and can be safely ignored. The pipeline uses only T1,
T1ce, T2, and FLAIR.

---

## References

1. Baid, U., et al. (2021). "A New Cohort of Glioblastoma Patients
   from the TCIA and RSNA NA-MIC/MRG/ITS Jelf." arXiv:2105.06468
2. Menze, B., et al. (2015). "The Multimodal Brain Tumor Image
   Segmentation Benchmark (BraTS)." IEEE TMI 34(10): 1993-2024
3. Kickingereder, P., et al. (2016). "Automated skull-stripping
   in MR images." NeuroImage 125: 704-723
4. TCIA: https://www.cancerimagingarchive.net/
5. BraTS Challenge: https://www.syndata.org/
