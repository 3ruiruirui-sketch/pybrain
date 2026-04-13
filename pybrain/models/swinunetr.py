# pybrain/models/swinunetr.py
"""
SwinUNETR model loader and inference wrapper (Optimized for Mac Mini/MPS).
Implementation follows INSTIG8R/swin-unetr best practices.
"""

import torch
import numpy as np
import os
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers.utils import sliding_window_inference
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.swinunetr")

def load_swinunetr(bundle_dir: Path, device: torch.device, model_cfg: Optional[Dict] = None) -> torch.nn.Module:
    """
    Industral-grade SwinUNETR loader.
    - Handles 'img_size' signature mismatch across MONAI versions.
    - Verifies weight loading (logs missing keys).
    - Uses flexible weight paths from config.
    """
    weights_name = model_cfg.get("weights", "fold1_swin_unetr.pth") if model_cfg else "fold1_swin_unetr.pth"
    weights_path = bundle_dir / weights_name
    
    if not weights_path.exists():
        raise FileNotFoundError(f"SwinUNETR weights not found at {weights_path}")

    # 1. Signature-aware Initialization
    # Some MONAI versions (research) require img_size, stable 1.x often omits it.
    sig = inspect.signature(SwinUNETR.__init__)
    kwargs = {
        "in_channels": 4,           
        "out_channels": 3,          
        "feature_size": 48,
        "use_checkpoint": True,
    }
    
    roi_size = model_cfg.get("roi_size", [128, 128, 128]) if model_cfg else [128, 128, 128]
    if "img_size" in sig.parameters:
        kwargs["img_size"] = tuple(roi_size)
        logger.info(f"SwinUNETR: Initializing with img_size={roi_size}")
    else:
        logger.info(f"SwinUNETR: Initializing without img_size (architecture defaults to ROI {roi_size})")

    model = SwinUNETR(**kwargs).to(device)
    
    # 2. Verified Weight Loading with Strict Validation
    logger.info(f"SwinUNETR: Loading weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    
    # Strict weight loading: try exact match first, fall back with full diagnostics
    try:
        model.load_state_dict(state_dict, strict=True)
        # strict=True raises RuntimeError on any mismatch, so this means clean load
        logger.info("SwinUNETR weights loaded successfully (strict=True, no key mismatches)")
    except RuntimeError as e:
        logger.warning(f"SwinUNETR strict weight loading failed: {e}")
        logger.warning("Falling back to strict=False — check for architecture drift")
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            logger.error(f"SwinUNETR Fallback: {len(msg.missing_keys)} missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            logger.warning(f"SwinUNETR Fallback: {len(msg.unexpected_keys)} unexpected keys: {msg.unexpected_keys}")
        
    model.eval()
    return model

def normalize_nonzero(tensor: torch.Tensor) -> torch.Tensor:
    """
    Per-channel normalization using only non-zero voxels.
    Matches INSTIG8R/swin-unetr official preprocessing.
    """
    for c in range(tensor.shape[1]):
        channel = tensor[0, c]
        mask = channel > 0
        if mask.any():
            mean = channel[mask].mean()
            std = channel[mask].std()
            tensor[0, c] = (channel - mean) / (std + 1e-8)
    return tensor

def run_swinunetr_inference(
    input_tensor: torch.Tensor,
    bundle_dir: Path,
    device: torch.device,
    model_cfg: Optional[Dict] = None
) -> np.ndarray:
    """
    Run SwinUNETR inference. Supports single model or multi-fold averaging. 
    Memory-safe: Loads and unloads folds sequentially.
    """
    import gc
    model_cfg = model_cfg or {}
    folds = model_cfg.get("use_folds", [])
    
    # If no folds specified, fallback to single weight from config or default
    if not folds:
        weights_name = model_cfg.get("weights", "fold1_swin_unetr.pth")
        folds = [weights_name]
    else:
        # Map fold indices to standardized filenames
        folds = [f"fold{i}_swin_unetr.pth" for i in folds]

    roi_size = tuple(model_cfg.get("roi_size", (128, 128, 128)))
    overlap = model_cfg.get("overlap", 0.5)

    # INPUT CONTRACT: caller (preprocess_volumes) has already applied per-channel
    # z-score normalization.  Do NOT re-normalize here — double normalization
    # compresses already-zero-centered values and degrades model predictions.
    # normalize_nonzero() was previously called here and has been removed.

    # CHANNEL PERMUTATION: pipeline stacks [FLAIR, T1, T1c, T2] at indices [0,1,2,3].
    # BraTS 2021 SwinUNETR weights were trained with [T1, T1ce, T2, FLAIR] order
    # (confirmed from MONAI BraTS 2021 SwinUNETR training config and checkpoint
    # patch_embed weight shape [48, 4, 2, 2, 2]).
    # Permutation (1, 2, 3, 0): T1[1]→ch0, T1c[2]→ch1, T2[3]→ch2, FLAIR[0]→ch3.
    input_normalized = input_tensor[:, [1, 2, 3, 0], ...].clone().to(device)
    accumulated_probs = None
    count = 0

    device_type = "cpu" if device.type == "cpu" else ("mps" if device.type == "mps" else "cuda")

    for weights_name in folds:
        try:
            logger.info(f"SwinUNETR Ensemble: Processing fold {weights_name}...")
            # Load specific fold
            fold_cfg = model_cfg.copy()
            fold_cfg["weights"] = weights_name
            model = load_swinunetr(bundle_dir, device, model_cfg=fold_cfg)
            
            with torch.no_grad():
                logger.info(f"SwinUNETR: Inference fold {weights_name} | roi={roi_size} | device={device_type}")
                with torch.autocast(device_type=device_type, enabled=(device_type != "cpu")):
                    outputs = sliding_window_inference(
                        inputs=input_normalized,
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=overlap,
                        mode="gaussian"
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
                prob = torch.sigmoid(output_tensor).cpu().numpy()
                
                # Handle different output formats
                if prob.ndim == 5:  # (batch, channel, x, y, z)
                    prob = prob.squeeze(0)  # Remove batch dimension
                
                if accumulated_probs is None:
                    accumulated_probs = prob
                else:
                    accumulated_probs += prob
                count += 1
            
            # Memory Cleanup
            del model
            if device_type == "mps":
                torch.mps.empty_cache()
            elif device_type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                logger.warning(f"SwinUNETR: OOM on fold {weights_name}. Skipping this fold.")
                continue
            raise e
        except Exception as e:
            logger.error(f"SwinUNETR: Error in fold {weights_name}: {e}")
            continue

    if accumulated_probs is None:
        raise RuntimeError("SwinUNETR: All folds failed or no folds provided.")

    # 2. Average results
    res = accumulated_probs / count
    
    # [IMPORTANT] BraTS 2021 Models (SwinUNETR) typically output:
    # Channel 0: TC (Tumor Core)
    # Channel 1: WT (Whole Tumor)
    # Channel 2: ET (Enhancing Tumor)
    #
    # TODO: Verify channel order matches pipeline expectation (WT/TC/ET).
    # Pipeline expects: Channel 0 = WT, Channel 1 = TC, Channel 2 = ET
    # If SwinUNETR outputs TC/WT/ET, need permutation before returning.
    # Validate on BraTS2021_00000 with known ground truth before enabling ensemble.
    #
    # NOTE: A ×1.05 WT boost was previously applied here (reduced from ×1.15).
    # Both were removed — unvalidated multipliers inside a model wrapper
    # systematically inflate edema volumes and bias ensemble fusion.
    # Any calibration should be done downstream via Platt scaling on a
    # held-out cohort, not as a hardcoded scalar in inference code.
    
    return res
