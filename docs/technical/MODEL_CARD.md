# Model Card — PY-BRAIN Brain Tumor Segmentation Pipeline

> **Model cards** document the provenance, performance characteristics,
> and known limitations of machine learning models. This card describes
> the PY-BRAIN ensemble pipeline.

---

## Basic Information

| Field | Value |
|-------|-------|
| **Model name** | PY-BRAIN Ensemble |
| **Version** | 0.1.0 |
| **Model type** | Multi-model ensemble with uncertainty quantification |
| **Input modalities** | T1, T1ce (T1+contrast), T2, FLAIR (4-channel MRI) |
| **Output** | Tumor segmentations: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET) |
| **License** | MIT (pipeline); Apache 2.0 (MONAI SwinUNETR); Non-commercial (BrainIAC) |
| **Paper / citation** | See `CITATION.cff` |

---

## Pipeline Architecture

### Active Models

#### 1. SegResNet (Primary — weight 0.60)

| Property | Value |
|----------|-------|
| Architecture | Residual encoder + decoder (MONAI `SegResNet`) |
| Pretrained | MONAI BraTS2021 pretrained bundle |
| ROI size | 240 × 240 × 160 |
| Overlap | 0.5 |
| Role | Primary segmentation model |
| Ensemble weight | 0.60 |

#### 2. TTA-4 — Test-Time Augmentation (Secondary — weight 0.40)

| Property | Value |
|----------|-------|
| Method | 4-fold axis-aligned flips (L↔R, A↔P, I↔S combinations) |
| Inference | Standard inference on each augmentation, average probabilities |
| Role | Noise reduction and boundary refinement |
| Ensemble weight | 0.40 |

#### 3. MC-Dropout — Epistemic Uncertainty (Enabled)

| Property | Value |
|----------|-------|
| Samples | 15 forward passes with dropout enabled |
| Model | SegResNet only |
| Uncertainty type | Epistemic (model uncertainty) |
| Output | Per-voxel standard deviation map |
| Calibration | Platt scaling applied post-hoc |
| Runtime overhead | ~34% additional time |

#### 4. Platt Calibration

| Property | Value |
|----------|-------|
| Method | Temperature scaling per subregion (WT, TC, ET) |
| Coefficients | Stored in `models/calibration/platt_coefficients.json` |
| Purpose | Calibrate probability outputs to better reflect true likelihood |

#### 5. STAPLE Ensemble

| Property | Value |
|----------|-------|
| Method | STAPLE (Simultaneous Truth and Performance Level Estimation) |
| Purpose | Fuse multiple segmentations with ground-truth estimation |
| Status | Implemented, not yet validated in ensemble |

---

## Disabled Models

#### SwinUNETR — DISABLED

| Property | Value |
|---------|-------|
| Reason | Awaiting Platt calibration validation and channel-order verification |
| Ensemble weight | 0.0 |
| Status | Temporary — see roadmap in README |
| Required steps | 1. Run `compute_platt_calibration.py` without `--skip-swinunetr`<br>2. Verify channel order (WT/TC/ET vs TC/WT/ET)<br>3. Set `ensemble_weights.swinunetr: 0.30` |

---

## Training Data

| Dataset | Cases | Source |
|---------|-------|--------|
| BraTS2021 Training | 1,291 | TCIA BraTS2021 Collection |

The pipeline was developed using multi-institutional MRI data from the
BraTS2021 challenge. All data is pre-processed to the same resolution
(1 mm³ isotropic) and co-registered to the FLAIR template.

> **Data disclaimer**: BraTS data is not distributed with this repository.
> Users must download separately as described in `docs/DATA.md`.

---

## Evaluation Data

| Case ID | Dice WT | Dice TC | Dice ET | HD95 (mm) | Quality |
|---------|----------|---------|---------|-----------|---------|
| BraTS2021_00000 | 0.933 | 0.905 | 0.925 | 10.3 | EXCELLENT |
| BraTS2021_00002 | 0.702 | 0.827 | 0.683 | 19.1 | CHALLENGING |
| BraTS2021_00003 | 0.967 | 0.937 | 0.912 | 1.4 | OUTSTANDING |

> Case 00002 is a clinically challenging case (large tumor mass with
> minimal enhancing component). The low WT Dice reflects tumor infiltration
> patterns that are inherently difficult to segment.

Full validation details: [docs/technical/VALIDATION.md](./VALIDATION.md)

---

## Metrics

### Target Thresholds

| Metric | Target | Status |
|--------|--------|--------|
| Dice WT | ≥ 0.93 | Validated |
| Dice TC | ≥ 0.85 | Validated |
| Dice ET | ≥ 0.80 | Validated |
| HD95 | < 15 mm | Validated |
| ASD | < 2.5 mm | Pending full validation |
| Volume Diff | < 10% | Validated (~3.5%) |

### Uncertainty Quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean MC std | 0.0014–0.002 | Very low — confident predictions |
| Max MC std | 0.227 | Higher at boundaries |
| WT boundary vs interior ratio | 1.32× | Moderately higher at boundary |
| TC boundary vs interior ratio | 8.01× | Significantly higher at boundary |
| Dice agreement (MC vs standard) | 0.940–0.950 | Excellent agreement |

---

## Brain Metastases Analysis

### Mets Detection Model

| Property | Value |
|----------|-------|
| Method | Fallback threshold (intensity-based) |
| Alternative methods | nnDetection (preferred), 3D RetinaNet (via MONAI) |
| Input | T1c post-contrast image |
| Output | Lesion candidates with centroids, bboxes, volumes |
| Min lesion volume | 0.05 cc (configurable) |
| Confidence threshold | 0.5 (configurable) |
| Max lesions | 50 (configurable) |

### Mets Segmentation Model

| Property | Value |
|----------|-------|
| Architecture | SegResNet (per-lesion patch segmentation) |
| Patch size | 64³ voxels (configurable) |
| Training data | Stanford BrainMetShare (~156 patients) |
| Training approach | Per-patient cross-validation (to avoid leakage) |
| Output | Binary segmentation per lesion |

### Mets Metrics (Target)

| Metric | Target | Status |
|--------|--------|--------|
| Detection sensitivity | ≥ 0.85 (lesions > 0.1 cc) | Pending validation |
| Detection false positives/scan | ≤ 2 | Pending validation |
| Per-lesion Dice (segmentation) | ≥ 0.80 | Pending validation |
| Volume error per lesion | ≤ 10% | Validated on synthetic phantoms |
| Auto mode classification accuracy | ≥ 0.90 | Validated on synthetic phantoms |

### Mets vs Glioma Auto-Classification

| Heuristic | Threshold |
|-----------|-----------|
| Multiple lesions → mets | ≥ 3 significant lesions |
| Single large lesion → glioma | 1 lesion > 50 cc |
| Uncertain → manual review | 2 lesions or intermediate size |

> **Note**: Mets analysis is currently in research development stage. Metrics are target values pending validation on clinical data. Use synthetic phantoms for testing (see `tests/test_mets.py`).

---

## Ethical Considerations

### Intended Use

**Research and experimental clinical investigation only.** This pipeline
is designed for:

- Retrospective medical imaging research
- Algorithm development and validation
- Exploratory clinical studies with radiologist oversight
- Educational purposes in medical imaging

### Not Intended For

- Direct clinical diagnosis without proper regulatory clearance
- Treatment planning without radiologist verification
- Any use that would constitute a certified medical device

### Known Limitations

1. **Out-of-distribution tumors**: Highly infiltrative or necrotic-heavy tumors
   may be undersegmented
2. **Post-operative cases**: Not validated on post-surgical scans
3. **Non-standard protocols**: Imaging acquired with non-standard sequences
   may degrade performance
4. **Multi-parametric assumptions**: Pipeline assumes standard 4-channel
   BraTS-style input (T1/T1ce/T2/FLAIR)
5. **MC-Dropout scope**: Uncertainty estimated only for SegResNet; TTA-4
   and SwinUNETR do not produce uncertainty maps

---

## Model Access

Model weights are **not included** in this repository due to size.

### MONAI SwinUNETR BraTS Bundle

Automatically downloaded by MONAI on first inference:
```
monai.apps.download_url(...)
# Destination: ~/.cache/torch/monai/...
```

### BrainIAC Weights

Available at: https://github.com/.... (Mass General Brigham repository)
Subject to non-commercial academic research license.

---

## Maintenance

| Field | Value |
|-------|-------|
| Last validated | 2026-04-13 |
| Validation framework | Internal regression suite (`tests/regression_baseline.py`) |
| Contact | Open GitHub issue for bugs/features |

---

*This model card follows the format recommended by Mitchell et al. (2018)
and the model card transparency framework.*
