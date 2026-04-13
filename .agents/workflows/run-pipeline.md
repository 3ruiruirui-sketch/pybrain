---
name: Run Brain Tumor Pipeline
command: /run-pipeline
---

# Autonomous Pipeline Execution

## Objective
Execute complete brain tumor segmentation from raw MRI to final NIfTI outputs.

## Execution Flow

### Phase 1: Data Loading (Data Engineer)
1. Load T1, T1c, T2, FLAIR from MONAI directory
2. Compute voxel geometry from reference image
3. Generate robust brain mask using T1
4. **Checkpoint**: Verify all files loaded, log volumes

### Phase 2: Preprocessing (Data Engineer)
1. Apply tissue fidelity patch (enhancement_map + CLAHE)
2. Normalize each modality using zscore_robust
3. Stack into 4-channel tensor (FLAIR, T1, T1c, T2)
4. Move to appropriate device (CUDA/MPS/CPU)

### Phase 3: Model Inference (Model Inference Specialist)
1. Load SegResNet from bundle directory
2. Run inference → segresnet_prob
3. Run TTA4 ensemble → tta4_prob
4. **Note**: SwinUNETR disabled (weights mismatch)
5. Log completion status for each model

### Phase 4: Ensemble Fusion (Ensemble Fusion Engineer)
1. Weighted fusion using config weights
2. Apply CT boost if CT data available
3. Compute voxel-wise uncertainty
4. **Checkpoint**: Save ensemble_prob.nii.gz

### Phase 5: Post-Processing (Post-Processing Specialist)
1. Apply thresholds from config (WT, TC, ET)
2. Derive necrotic, edema, enhancing masks
3. Remove components < 0.5 cc
4. Clinical validation against reference volume
5. **Checkpoint**: Save final segmentation

### Phase 6: Visualization (Visualization Expert)
1. Generate axial grid (8 slices, 4 modalities)
2. Generate coronal grid
3. Generate sagittal grid
4. Save PNGs to output directory

### Phase 7: Quality Reporting (Quality Assurance Agent)
1. Compute tumor volumes (WT, TC, ET, necrosis, edema)
2. Validate against ground truth if available
3. Write JSON reports
4. Log final summary

## Success Criteria
- All 7 phases complete without errors
- All output files generated in output directory
- Final volume within 15% of radiologist reference (if provided)

## Failure Handling
If any phase fails:
1. Log error with full stack trace
2. Do NOT proceed to next phase
3. Report which phase failed
4. Preserve intermediate outputs for debugging
