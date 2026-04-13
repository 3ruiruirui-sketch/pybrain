---
description: Model Inference Skill
---
# Model Inference Protocol

## Input Requirements
- Stacked MRI Tensor: shape (1, 4, H, W, D) in order (FLAIR, T1, T1c, T2)
- device: target compute device ('mps' or 'cuda')

## Inference Algorithm
1. Instantiate standard SegResNet model weighting via state loaded from `model.pt`.
2. Infer Primary Probability: Feed target chunk using sliding window inference.
3. Infer TTA Probability: Feed target chunk with Test-Time Augmentation (TTA4) (flip along all 3 axes).
4. Squeeze batch size mapping from predictions.

## Quality Constraints
- Ensure safe garbage collection (`gc.collect()` and memory flushes) after each large architecture yields logic.
- Do not exceed tensor memory limitations (optimize sliding window batch parameters).
