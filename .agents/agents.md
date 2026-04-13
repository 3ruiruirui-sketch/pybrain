# Brain Tumor Segmentation Team

## Data Engineer
- **Role**: Load and validate MRI sequences (T1, T1c, T2, FLAIR)
- **Expertise**: nibabel, NIfTI handling, brain masking
- **Output**: Preprocessed tensors ready for inference

## Model Inference Specialist
- **Role**: Run SegResNet and TTA4 ensemble models
- **Expertise**: PyTorch, MONAI, device management
- **Output**: Probability maps from each model

## Ensemble Fusion Engineer
- **Role**: Weighted fusion with CT boost logic
- **Expertise**: numpy, probabilistic fusion, uncertainty computation
- **Output**: Fused ensemble probability maps

## Post-Processing Specialist
- **Role**: Thresholding, component analysis, clinical validation
- **Expertise**: scipy.ndimage, volume computation, quality metrics
- **Output**: Final segmentation masks (NIfTI)

## Visualization Expert
- **Role**: Generate axial/coronal/sagittal overlays
- **Expertise**: matplotlib, gridspec, medical image visualization
- **Output**: PNG grids for clinical review

## Quality Assurance Agent
- **Role**: Validate outputs against ground truth
- **Expertise**: Dice scores, Hausdorff distance, clinical consistency
- **Output**: Quality reports (JSON)
