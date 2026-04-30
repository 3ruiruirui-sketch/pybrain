#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnU-Net (DynUNet) model wrapper for PY-BRAIN.
Uses MONAI's DynUNet which implements the nnU-Net architecture.
Optimized for Mac Mini M1/MPS and CPU fallback.
"""

import gc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from monai.networks.nets.dynunet import DynUNet
from monai.inferers.utils import sliding_window_inference
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.nnunet")


def normalize_nonzero(tensor: torch.Tensor) -> torch.Tensor:
    """
    Per-channel z-score normalization using only non-zero voxels.
    Matches nnU-Net preprocessing convention.
    """
    for c in range(tensor.shape[1]):
        channel = tensor[0, c]
        mask = channel > 0
        if mask.any():
            mean = channel[mask].mean()
            std = channel[mask].std()
            tensor[0, c] = (channel - mean) / (std + 1e-8)
    return tensor


def load_nnunet(device: torch.device, model_cfg: Optional[Dict] = None) -> torch.nn.Module:
    """
    Load DynUNet (nnU-Net architecture) with domain-adapted configuration.

    Architecture: 3-stage encoder-decoder (filters [32,64,128]).
    Uses residual blocks (res_block=True) for better gradient flow.
    """
    model_cfg = model_cfg or {}

    # nnU-Net style configuration for brain MRI (1mm isotropic).
    # Architecture MUST match the pretrained weights file at
    # models/brats_bundle/nnunet_weights.pth — verified by checkpoint
    # introspection 2026-04-29:
    #   input_block (32) → downsamples 0/1/2 (64, 128, 256) → bottleneck (512)
    #   upsamples 0/1/2/3 (256, 128, 64, 32) → output_block (3 = WT/TC/ET)
    # → 5-level DynUNet, filters=[32,64,128,256,512], strides=[1,2,2,2,2].
    #
    # DynUNet requires: len(kernel_size) == len(strides) == len(filters);
    # upsample_kernel_size has len(strides) - 1 entries (one per upsample).
    filters = model_cfg.get("filters", [32, 64, 128, 256, 512])
    n_stages = len(filters)

    # Build kernel/stride lists matching the number of stages
    kernel_size = [[3, 3, 3]] * n_stages
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    upsample_ks = [[2, 2, 2]] * (n_stages - 1)

    model = DynUNet(
        spatial_dims=3,
        in_channels=4,  # T1, T1c, T2, FLAIR
        out_channels=3,  # WT, TC, ET
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_ks,
        filters=filters,
        res_block=True,
        deep_supervision=model_cfg.get("deep_supervision", False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"DynUNet: {total_params / 1e6:.1f}M total params ({trainable / 1e6:.1f}M trainable)")

    model.eval()
    return model


def run_nnunet_inference(
    input_tensor: torch.Tensor, device: torch.device, model_cfg: Optional[Dict] = None
) -> np.ndarray:
    """
    Run nnU-Net (DynUNet) inference using MONAI sliding window.

    Returns:
        np.ndarray shape (3, D, H, W) with channels [TC, WT, ET]
        - TC: Tumour Core (labels 1 + 3)       — channel 0
        - WT: Whole Tumour (labels 1 + 2 + 3)  — channel 1
        - ET: Enhancing Tumour (label 3)        — channel 2
    """
    model_cfg = model_cfg or {}

    # ── Hard gate: bail out immediately if disabled or weight=0 ──────────────
    # This prevents any accidental model load when nnunet is excluded from
    # the pipeline (ensemble_weights.nnunet=0.0 / models.nnunet.enabled=false).
    if not model_cfg.get("enabled", False):
        logger.info("nnU-Net disabled (models.nnunet.enabled=false) — skipping inference")
        spatial = tuple(input_tensor.shape[2:])
        return np.zeros((3,) + spatial, dtype=np.float32)

    # Load model
    model = load_nnunet(device, model_cfg)

    # Load pretrained weights if available
    bundle_dir = Path(model_cfg.get("bundle_dir", "models/brats_bundle"))
    weights_path = bundle_dir / "nnunet_weights.pth"
    if weights_path.exists():
        logger.info(f"DynUNet: Loading pretrained weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            logger.warning(f"DynUNet: {len(msg.missing_keys)} missing keys (will use random init for those)")
        if msg.unexpected_keys:
            logger.warning(f"DynUNet: {len(msg.unexpected_keys)} unexpected keys in checkpoint")
    else:
        logger.error("=" * 60)
        logger.error("NNUNET WEIGHTS MISSING — MODEL RUNS ON RANDOM INITIALIZATION")
        logger.error(f"  Expected: {weights_path}")
        logger.error("  All nnU-Net predictions will be RANDOM NOISE.")
        logger.error("  Set ensemble_weights.nnunet=0.0 in defaults.yaml until")
        logger.error("  validated BraTS weights are placed at the path above.")
        logger.error("=" * 60)

    # INPUT CONTRACT: caller (preprocess_volumes) has already applied per-channel
    # z-score normalization.  Do NOT re-normalize here — double normalization
    # compresses already-zero-centered values and degrades model predictions.
    # normalize_nonzero() was previously called here and has been removed.
    input_normalized = input_tensor.clone().to(device)

    logger.info(f"DynUNet: Input tensor shape: {input_tensor.shape}")

    # Sliding window inference
    roi_size = tuple(model_cfg.get("roi_size", [128, 128, 128]))
    overlap = model_cfg.get("overlap", 0.5)

    logger.info(f"DynUNet: ROI size: {roi_size}, overlap: {overlap}")

    device_type = "cpu" if device.type == "cpu" else ("mps" if device.type == "mps" else "cuda")
    use_amp = device_type != "cpu"

    try:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=use_amp):
                outputs = sliding_window_inference(
                    inputs=input_normalized,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )

            # Handle different output formats
            if isinstance(outputs, dict):
                # Some models return dict
                output_tensor = list(outputs.values())[0]
            elif isinstance(outputs, tuple):
                # Some models return tuple
                output_tensor = outputs[0]
            else:
                # Standard tensor output
                output_tensor = outputs

            # Apply sigmoid to convert logits to probabilities
            prob = torch.sigmoid(output_tensor).cpu().numpy().squeeze(0)

            # Handle different output formats
            if prob.ndim == 5:  # (batch, channel, x, y, z)
                prob = prob.squeeze(0)  # Remove batch dimension

        # Orientation handling — generic, no hardcoded shapes.
        # MONAI sliding_window_inference preserves spatial dims of the input.
        # Input is (1, C_in, D, H, W) so output is (1, C_out, D, H, W).
        # After squeeze(0) we have (C, D, H, W) which matches the input order.
        roi_mode = model_cfg.get("roi_mode", False)
        expected_shape = model_cfg.get("padded_roi_shape", None)

        if roi_mode and expected_shape:
            expected_shape = tuple(expected_shape)
            if prob.shape == (3,) + expected_shape:
                logger.info(f"DynUNet: ROI mode - using correct orientation {prob.shape}")
            elif set(prob.shape[1:]) == set(expected_shape) and prob.shape[1:] != expected_shape:
                # Spatial dims are a permutation of expected — find the right one
                for perm in [(0, 1, 2, 3), (0, 3, 2, 1), (0, 2, 3, 1), (0, 3, 1, 2), (0, 2, 1, 3), (0, 1, 3, 2)]:
                    candidate = np.transpose(prob, perm)
                    if candidate.shape == (3,) + expected_shape:
                        prob = candidate
                        logger.info(f"DynUNet: ROI mode - transposed {perm} → {prob.shape}")
                        break
                else:
                    logger.warning(f"DynUNet: Could not match expected shape {expected_shape} from {prob.shape}")
            else:
                logger.info(f"DynUNet: ROI mode - shape {prob.shape} (expected (3,)+{expected_shape})")
        else:
            # Standard (non-ROI) mode: prob is already (C, D, H, W) from MONAI
            if prob.ndim == 3:
                prob = prob[np.newaxis, ...]
            logger.info(f"DynUNet: Standard mode - output shape {prob.shape}")

        logger.info(f"DynUNet: Final prob shape: {prob.shape}")

        # NOTE: sigmoid already applied via torch.sigmoid() above (line ~160).
        # Do NOT apply a second sigmoid here — that would map [0,1] → [0.5, 0.73]
        # and destroy all discriminative information in the probability map.
        prob = np.clip(prob, 0.0, 1.0)  # safety clamp only

        # Channel order: [TC, WT, ET] — matches SegResNet / SwinUNETR convention
        # No swapping needed

    finally:
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"DynUNet: Output shape {prob.shape}, range [{prob.min():.3f}, {prob.max():.3f}]")
    return prob
