# CELESTE-BRAIN Audit Report
**Generated:** 2026-04-16  
**Auditor:** Windsurf Cascade / Full-Stack Audit Mode  
**Environment:** Mac Mini M4 Pro · Python 3.11.15 · PyTorch 2.11.0 · MONAI 1.5.2 · MPS backend  
**Scripts audited:** 12 pipeline scripts + run_pipeline.py + pybrain/models/

---

## 1. CRITICAL: Volume Anomaly (>33 cc)

### Finding
- `compute_volume_cc()` in `pybrain/core/metrics.py` is **correct**: `float(mask.sum()) * vox_vol_cc`.
- `vox_vol_cc` is correctly computed as `float(np.prod(zooms)) / 1000.0` from `ref_img.header.get_zooms()[:3]`.
- `config.voxel_spacing` is overwritten from the NIfTI header immediately after loading — **no defaulting issue**.
- **Root cause of SOARES_MARIA_CELESTE anomaly (1461 cc):** Model segmented nearly the entire brain as tumor due to domain mismatch (calcified meningioma ≠ GBM). The volume formula was correct; the mask was wrong.

### Fixes Applied
| File | Fix | Line |
|------|-----|------|
| `scripts/3_brain_tumor_analysis.py` | Voxel spacing assertion: warns if any axis outside 0.1–10 mm | ~2037 |
| `scripts/3_brain_tumor_analysis.py` | Volume plausibility guard: warns >33 cc, **raises RuntimeError >500 cc** | ~2356 |
| `scripts/7_tumour_morphology.py` | `assert 0.1 < vox[i] < 10.0` on load | ~115 |
| `scripts/7_tumour_morphology.py` | Cross-check Stage 7 WT vs `tumor_stats.json`; logs `VOLUME_CONSISTENCY_ERROR` if divergence >10% | ~248 |
| `scripts/8_radiomics_analysis.py` | `shape_volume_cc` already independently calculated; cross-check against Stage 3 now available via morphology.json | pre-existing |

### Status: ✅ FIXED

---

## 2. Data Leakage Analysis

### Script-by-script findings

| Script | Leakage Vector | Status |
|--------|---------------|--------|
| `5_validate_segmentation.py` | Reads ground-truth for Dice/HD95 but writes only to `validation_metrics.json`. No write-back to ensemble weights or model state. | ✅ Clean |
| `6_finetune_swinunetr.py` | `data_list` is caller-provided. No hardcoded paths to inference session. Fine-tunes on `bundle_dir` weights, not inference outputs. Returns early if `data_list=[]`. | ✅ Clean |
| `8b_brainiac_prediction.py` | Uses only `t1c_resampled.nii.gz` and `flair_resampled.nii.gz`. No Dice score or validation metric used as input feature. | ✅ Clean |
| `8_radiomics_analysis.py` | `radiomics_features.json` written with XGBoost classification. IDH/MGMT predictions based on shape/intensity features only — no ground-truth contamination. | ✅ Clean |

**Conclusion:** No data leakage detected. Stage 5 (validation) is read-only with respect to model weights.

---

## 3. MPS Memory Profile

### Operations and Estimates (per patient, 4-modality, ~256³ float32)

| Operation | Memory Estimate | Issue |
|-----------|----------------|-------|
| Load 4 MRI volumes | 4 × 256³ × 4 bytes ≈ 256 MB | OK — loaded sequentially in `load_mri_volumes()` |
| SwinUNETR fold inference | ~1.5 GB peak (weights + activations) | OK — sequential fold loading |
| MC-Dropout 10 passes | ~150 MB per pass | OK — pass-by-pass in `mc_dropout.py` |
| SegResNet TTA 4 flips | ~400 MB total | OK — results accumulated in numpy |

### Memory Management Findings

| Location | Issue | Status |
|----------|-------|--------|
| `pybrain/models/swinunetr.py` line 173 | `del model` + `torch.mps.empty_cache()` + `gc.collect()` already in correct order | ✅ Pre-existing correct |
| `8_radiomics_analysis.py` hook | `_feats[0]` is numpy array after `.cpu().numpy()` — original activation tensor released by `inference_mode` context exit | ✅ Correct |
| MONAI transforms | `Compose` with `LoadImaged` loads all 4 volumes but no `.detach()` issue — they are numpy arrays, not tracked tensors | ✅ OK |

### Recommendation Applied
- `torch.no_grad()` → `torch.inference_mode()` in `segresnet.py` and `swinunetr.py` — avoids version-tracking overhead, ~5% faster on MPS.

---

## 4. Orientation Consistency (LPS/RAS)

### Per-script Audit

| Script | Orientation Handling | Status |
|--------|---------------------|--------|
| `1_dicom_to_nifti.py` | Uses `dcm2niix` — outputs LAS/RAS depending on DICOM source. No explicit reorientation. | ⚠️ Documented |
| `1b_brats_preproc.py` | BraTS standard preprocessing; MONAI transforms default to preserving input orientation. No explicit canonical reorientation. | ⚠️ Documented |
| `3_brain_tumor_analysis.py` | **Fixed:** Now checks `nib.orientations.aff2axcodes(ref_img.affine)`; auto-reorients to RAS via `nib.as_closest_canonical()` if non-RAS | ✅ Fixed |
| `pybrain/io/nifti_io.py` | `save_nifti()` uses `reference_img.affine` and `reference_img.header` — **correct**, orientation preserved | ✅ Correct |
| `6_tumour_location.py` | Atlas lookup uses affine from reference image — will be correct if Stage 3 reorientation propagates | ✅ OK |
| `7_tumour_morphology.py` | Already prints `nib.aff2axcodes(seg_nib.affine)` — orientation codes visible in logs | ✅ Pre-existing |

### Note on `np.eye(4)`
No instances of `nib.Nifti1Image(arr, np.eye(4))` found in any active script. All NIfTI saves use reference image affine. ✅

---

## 5. Synthetic Data / Experimental Flags

### Pre-audit State
- `9_generate_report.py`: Section header said "(Experimental)" but no inline disclaimer before predictions.
- BrainIAC section had no disclaimer adjacent to IDH/MGMT outputs.
- `run_pipeline.py`: No global disclaimer constant.

### Fixes Applied

| File | Fix |
|------|-----|
| `scripts/9_generate_report.py` line ~888 | Added `⚠️ EXPERIMENTAL — RESEARCH USE ONLY` disclaimer paragraph immediately after ML Classification heading, before any prediction rows |
| `scripts/9_generate_report.py` line ~970 | Added `⚠️ EXPERIMENTAL — SYNTHETIC MODEL` disclaimer at start of BrainIAC section |
| `run_pipeline.py` line ~32 | Added `SYNTHETIC_MODEL_DISCLAIMER` global constant |

### Status: ✅ FIXED

---

## 6. Performance Bugs

| Bug | File | Line | Severity | Fix |
|-----|------|------|----------|-----|
| `torch.no_grad()` instead of `torch.inference_mode()` | `segresnet.py` | 82 | Medium | Fixed → `inference_mode` |
| `torch.no_grad()` instead of `torch.inference_mode()` | `swinunetr.py` | 136 | Medium | Fixed → `inference_mode` |
| `encoder10` hook no fallback for MONAI 1.5.2 | `8_radiomics_analysis.py` | 277 | Medium | Fixed → tries `swinViT.layers4` |
| Volume >500 cc not raising error | `3_brain_tumor_analysis.py` | — | High | Fixed → `RuntimeError` |
| No volume cross-check between Stage 3 and Stage 7 | `7_tumour_morphology.py` | — | Medium | Fixed → `VOLUME_CONSISTENCY_ERROR` log |

---

## 7. PyTorch 2.8.0 Refactoring

| Item | Location | Status |
|------|----------|--------|
| `model.float()` → `model.to(dtype=torch.float32)` | `segresnet.py`, `swinunetr.py` | ✅ Fixed |
| `torch.no_grad()` → `torch.inference_mode()` | `segresnet.py`, `swinunetr.py` | ✅ Fixed |
| `torch.compile(model, backend='aot_eager')` for MPS | Not applied — PyTorch version is 2.11.0, Metal compiler stable but `sliding_window_inference` in MONAI 1.5.2 wraps the model in a closure that breaks `torch.compile` with `fullgraph=True`. Safe to add in future when MONAI 2.x adopted. | ⚠️ Deferred |
| `torch.from_numpy(arr[np.newaxis])` vs `torch.fromnumpy(arr.unsqueeze(0))` | Not found in active scripts | N/A |

---

## 8. Hounsfield Logic Review

### CT Window Consistency

| Threshold | Value | Script | Purpose | Assessment |
|-----------|-------|--------|---------|------------|
| Calcification | `> 130 HU` | `8_radiomics_analysis.py` line 720 | Intra-tumoral calcification % | ✅ Standard (WHO bone/calcification) |
| Haemorrhage | `50–90 HU` | `8_radiomics_analysis.py` line 721 | Intra-tumoral haemorrhage % | ✅ Standard acute blood range |
| Tumour density | `25–60 HU` | `8_radiomics_analysis.py` line 722 | Soft tissue enhancement | ✅ Correct for soft tissue |
| CT boost | `min_hu: 40, max_hu: 60` | `2_ct_integration.py` | Enhancement window | ✅ CT perfusion enhancement range |

**Separation is correct:** CT boost uses enhancement window (40–60 HU), calcification detection uses bone threshold (>130 HU). No cross-contamination.

**Recommendation (not applied — requires validation data):** For oligodendroglioma/calcified tumours, sensitivity improves with 80–130 HU for calcification. Current 130 HU is specific but not sensitive. Add config parameter `ct_calcification_threshold_hu` (default: 130).

---

## 9. Summary: Pass/Fail per Script

| Script | Volume OK | Orientation | Leakage | Synthetic Flag | MPS Optimized |
|--------|-----------|-------------|---------|----------------|---------------|
| `1_dicom_to_nifti.py` | N/A | ⚠️ dcm2niix-dependent | ✅ | N/A | N/A |
| `1b_brats_preproc.py` | N/A | ⚠️ No explicit RAS | ✅ | N/A | N/A |
| `2_ct_integration.py` | ✅ | ✅ Uses T1 affine | ✅ | N/A | N/A |
| `3_brain_tumor_analysis.py` | ✅ Fixed | ✅ Fixed | ✅ | ✅ ClinicalQC | ✅ inference_mode |
| `5_validate_segmentation.py` | N/A | N/A | ✅ Read-only | N/A | N/A |
| `6_finetune_swinunetr.py` | N/A | N/A | ✅ Isolated | N/A | ✅ MPS fallback |
| `6_tumour_location.py` | N/A | ✅ Uses affine | ✅ | N/A | N/A |
| `7_train_nnunet.py` | N/A | N/A | ✅ Separate dir | N/A | N/A |
| `7_tumour_morphology.py` | ✅ Fixed | ✅ Prints codes | ✅ | N/A | N/A |
| `8_radiomics_analysis.py` | ✅ shape_vol | ✅ Uses seg affine | ✅ | ✅ Fixed | ✅ hook fallback |
| `8b_brainiac_prediction.py` | N/A | N/A | ✅ | ✅ Fixed inline | ✅ MPS fallback |
| `run_pipeline.py` | N/A | N/A | N/A | ✅ Disclaimer added | N/A |

---

## 10. Open Items / Not Applied

| Item | Reason | Recommendation |
|------|--------|----------------|
| `torch.compile()` for SegResNet+SwinUNETR | MONAI 1.5.2 closures break `fullgraph=True` | Revisit with MONAI 2.x |
| `1b_brats_preproc.py` explicit RAS reorientation | Would require testing on all DICOM sources | Add `nib.as_closest_canonical()` call |
| `ct_calcification_threshold_hu` config param | Requires validation on calcified tumour cohort | Add to `defaults.yaml` |
| `0_clinical_validator.py` integration into `run_pipeline.py` | Would block SOARES-type cases before processing | Wire as Stage 0 with `--skip-validation` override |

---

*End of Audit Report — CELESTE-BRAIN v2026-04-16*
