# pybrain/models/segresnet.py
"""
SegResNet model loader and inference wrapper (BraTS).
"""

import torch
import gc
import numpy as np
from pathlib import Path
from typing import cast
from monai.bundle.config_parser import ConfigParser
from monai.bundle.scripts import download
from monai.inferers.inferer import SlidingWindowInferer
from monai.utils.type_conversion import convert_to_numpy
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.segresnet")

def load_segresnet(bundle_dir: Path, device: torch.device):
    """Load SegResNet model from MONAI bundle."""
    bundle_name = "brats_mri_segmentation"
    bundle_path = bundle_dir / bundle_name
    
    if not bundle_path.exists():
        logger.info(f"Downloading {bundle_name}...")
        download(name=bundle_name, bundle_dir=str(bundle_dir))
    
    parser = ConfigParser()
    parser.read_config(str(bundle_path / "configs" / "inference.json"))
    model = parser.get_parsed_content("network_def", instantiate=True)
    
    ckpt_files = list((bundle_path / "models").glob("*.pt*"))
    if not ckpt_files:
        raise FileNotFoundError(f"Missing weights in {bundle_name}")
    
    logger.info(f"Loading weights from {ckpt_files[0].name}")
    ckpt = torch.load(str(ckpt_files[0]), map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    
    # Strict weight loading: try exact match first, fall back with full diagnostics
    try:
        model.load_state_dict(sd, strict=True)
        # strict=True raises RuntimeError on any mismatch,
        # so reaching here guarantees zero missing / unexpected keys.
        logger.info("Weights loaded successfully (strict=True, no key mismatches)")
    except RuntimeError as e:
        # Architecture / key mismatch — fall back to strict=False and log everything
        logger.warning(f"Strict weight loading failed: {e}")
        logger.warning("Falling back to strict=False — check for architecture drift")
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys:
            logger.error(f"Fallback: {len(result.missing_keys)} missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Fallback: {len(result.unexpected_keys)} unexpected keys: {result.unexpected_keys}")
    
    model = model.to(device).eval()
    return model

def run_segresnet_inference(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device, sw_config: dict, model_device=None):
    """Perform sliding window inference with SegResNet."""
    _dev = model_device if model_device is not None else device
    model = model.to(_dev)

    # ── Channel-order correction ─────────────────────────────────────────────
    # MONAI brats_mri_segmentation bundle (BraTS2018) expects input channels:
    #   [0]=T1c, [1]=T1, [2]=T2, [3]=FLAIR
    # PY-BRAIN pipeline's preprocess_volumes() stacks:
    #   [0]=FLAIR, [1]=T1, [2]=T1c, [3]=T2
    # Permutation: current [FLAIR,T1,T1c,T2] → correct [T1c,T1,T2,FLAIR]
    #               indices [0,  1,   2,   3  ] → indices [2,  1,  3,  0 ]
    _input = input_tensor[:, (2, 1, 3, 0), :, :, :].clone()
    # ─────────────────────────────────────────────────────────────────────────

    _input = _input.to(_dev)

    roi_size = tuple(sw_config.get("roi_size", (240, 240, 160)))
    sw_batch_size = sw_config.get("sw_batch_size", 1)
    overlap = sw_config.get("overlap", 0.5)
    
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian")
    
    with torch.no_grad():
        logits = cast(torch.Tensor, inferer(_input, model))
        
    probs = torch.sigmoid(logits.cpu())
    p = convert_to_numpy(probs[0])
    
    # BraTS format conversion: [NCR+ET, NCR+ED+ET, ET]
    if p.shape[0] == 4:
        # [BG, NCR, ED, ET] -> [TC, WT, ET]
        p = np.stack([p[1]+p[3], p[1]+p[2]+p[3], p[3]], axis=0)
        
    return p

def run_tta_ensemble(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device, sw_config: dict, model_device=None):
    """Perform Test-Time Augmentation (TTA) with enhanced capabilities."""
    from pybrain.io.logging_utils import get_logger
    logger = get_logger("models.segresnet")
    _dev = model_device if model_device is not None else device
    
    # Check if enhanced TTA is enabled
    enhanced_tta = sw_config.get("enhanced_tta", False)
    
    if enhanced_tta:
        # Use enhanced TTA
        from pybrain.models.enhanced_tta import run_enhanced_tta_ensemble
        logger.info("Using enhanced TTA with rotations and additional transforms")
        
        tta_config = sw_config.get("tta_config", {})
        return run_enhanced_tta_ensemble(model, input_tensor, device, sw_config, tta_config)
    else:
        # Original 4-flip TTA
        logger.info("Using original 4-flip TTA")
        flip_axes = [[2], [3], [4], [2, 3]]  # Axial, Coronal, Sagittal, and dual X-Y
        tta_results = []
        
        for axes in flip_axes:
            # Flip input
            t_flip = torch.flip(input_tensor, dims=axes)
            # Inference
            p_flip = run_segresnet_inference(model, t_flip, device, sw_config, model_device=_dev)
            # Flip back prediction
            np_axes = tuple(a - 1 for a in axes) # adjust for lack of batch dim
            p_flip = np.flip(p_flip, axis=np_axes).copy()
            tta_results.append(p_flip)
            
        return np.mean(tta_results, axis=0)
