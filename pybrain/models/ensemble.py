import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.ensemble")

def run_weighted_ensemble(
    model_probs: List[Tuple[str, np.ndarray, float]],
    ct_data: Optional[np.ndarray] = None,
    ct_config: Optional[Dict] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Fuses multiple probability maps using assigned weights and applies CT-density boost.
    Enhancement: Uncertainty-aware re-weighting (Optional).
    """
    valid_models = [(name, p, w) for name, p, w in model_probs if p is not None]
    if not valid_models:
        raise ValueError("No valid probability maps provided for ensemble.")

    total_weight = sum(w for _, _, w in valid_models)
    if total_weight <= 0:
        raise ValueError(f"Total ensemble weight must be positive, got {total_weight}. Check ensemble_weights in config.")

    # Simple weighted average
    p_ensemble = sum(p * w for _, p, w in valid_models) / total_weight
    contributed = [name for name, _, _ in valid_models]

    # CT-Boost Logic (Physical Density Prior)
    if ct_data is not None and ct_config:
        # HU [35, 75] captures hyperdense tumour while excluding normal brain
        hu_min = ct_config.get("min_hu", ct_config.get("hu_min", 35))
        hu_max = ct_config.get("max_hu", ct_config.get("hu_max", 75))
        boost_weight = ct_config.get("boost_factor", ct_config.get("weight", 0.30))

        if ct_data.shape == p_ensemble[1].shape:
            ct_tumor_prior = ((ct_data >= hu_min) & (ct_data <= hu_max)).astype(np.float32)
            # Boost WT (channel 1) - matches main pipeline's apply_ct_boost
            p_ensemble[1] = np.clip(p_ensemble[1] + (ct_tumor_prior * boost_weight), 0.0, 1.0)
            # Maintain BraTS invariant: WT (channel 1) >= TC (channel 0)
            p_ensemble[1] = np.maximum(p_ensemble[1], p_ensemble[0])
            logger.info(f"Ensemble: Applied CT boost (factor={boost_weight})")
        else:
            logger.warning(
                f"CT boost skipped: CT shape {ct_data.shape} != probability shape {p_ensemble[0].shape}. "
                "Ensure CT is resampled to match MRI segmentation grid.")

    return p_ensemble, contributed


def run_nnunet_inference(
    input_tensor: torch.Tensor,
    device: torch.device,
    model_cfg: Optional[Dict] = None
) -> Optional[np.ndarray]:
    """
    nnU-Net (DynUNet) inference via MONAI sliding window.
    Imports from pybrain.models.nnunet to avoid circular deps.
    """
    try:
        from pybrain.models.nnunet import run_nnunet_inference as _run
        return _run(input_tensor, device, model_cfg)
    except Exception as e:
        logger.warning(f"nnU-Net inference failed: {e}")
        return None

def compute_uncertainty(p_ensemble: np.ndarray, model_probs_list: List[np.ndarray]) -> np.ndarray:
    """
    Computes voxel-wise uncertainty using a combination of average entropy 
    and inter-model variance. Enhanced for better clinical sensitivity.
    """
    # Channel-wise entropy of ensemble
    p_safe = np.clip(p_ensemble, 1e-8, 1.0)
    ent = - (p_safe * np.log(p_safe) + (1 - p_safe) * np.log(1 - p_safe + 1e-8))
    
    if len(model_probs_list) > 1:
        # Enhanced variance computation between models
        p_stack = np.stack(model_probs_list, axis=0)
        
        # Per-channel variance
        variance_per_channel = p_stack.var(axis=0)  # Shape: (3, D, H, W)
        variance = variance_per_channel.mean(axis=0)  # Average across channels
        
        # Standard deviation (more sensitive to outliers)
        std_dev = p_stack.std(axis=0).mean(axis=0)
        
        # Maximum disagreement (range between min and max predictions)
        max_disagreement = (p_stack.max(axis=0) - p_stack.min(axis=0)).mean(axis=0)
        
        # Combined uncertainty with weighted components
        # Entropy captures prediction confidence
        # Variance captures model disagreement
        # Std dev captures outlier predictions
        # Max disagreement captures worst-case disagreement
        uncertainty = (
            0.4 * ent.mean(axis=0) +           # Entropy weight
            0.3 * variance +                    # Variance weight
            0.2 * std_dev +                     # Std dev weight
            0.1 * max_disagreement              # Max disagreement weight
        )
    else:
        # Single model: only entropy available
        uncertainty = ent.mean(axis=0)
        
    # Do NOT apply per-volume min-max normalization here.
    # Normalizing to [0,1] per volume destroys the absolute meaning of uncertainty:
    # a globally confident ensemble and a globally uncertain one would produce
    # identical normalized maps, making flag_high_uncertainty_regions unreliable.
    # The raw combined metric is bounded by construction:
    #   entropy    ∈ [0, ln2 ≈ 0.693]
    #   variance   ∈ [0, 0.25]
    #   std_dev    ∈ [0, 0.50]
    #   max_disagr ∈ [0, 1.00]
    # Weighted sum maximum ≈ 0.4*0.693 + 0.3*0.25 + 0.2*0.5 + 0.1*1.0 ≈ 0.577
    # Clamp to [0,1] as a safety guard only.
    return np.clip(uncertainty, 0.0, 1.0)
