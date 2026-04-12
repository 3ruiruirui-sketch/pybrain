"""
Subregion-Specific Ensemble Implementation
Medical Engineering Enhancement - Phase 2 Quality Assurance

Implements tumor subregion-specific ensemble weights to optimize Dice scores
for different tumor components: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.subregion_ensemble")

def run_subregion_weighted_ensemble(
    model_probs: List[Tuple[str, np.ndarray, float]],
    subregion_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Fuses probability maps using subregion-specific weights.

    CT boost is NOT applied here.  It is applied exclusively in
    apply_ct_boost() in 3_brain_tumor_analysis.py, which includes NMI
    registration validation.  Having a boost path here would silently
    double-apply the boost and bypass the registration check.
    """
    valid_models = [(name, p, w) for name, p, w in model_probs if p is not None]
    if not valid_models:
        raise ValueError("No valid probability maps provided for ensemble.")

    # Default to uniform weights if no subregion weights provided
    if subregion_weights is None:
        subregion_weights = get_default_subregion_weights(valid_models)
    
    # Initialize ensemble probabilities
    first_prob = valid_models[0][1]
    ensemble_prob = np.zeros_like(first_prob)
    
    # Apply subregion-specific weights for each channel
    # Channel order: [TC, WT, ET] (0, 1, 2)
    channel_names = ["TC", "WT", "ET"]
    
    for channel_idx, channel_name in enumerate(channel_names):
        channel_weighted_sum = np.zeros_like(first_prob[channel_idx])
        total_channel_weight = 0.0
        
        for model_name, model_prob, base_weight in valid_models:
            # Get subregion-specific weight for this model and channel
            model_weight = subregion_weights.get(channel_name, {}).get(model_name, base_weight)
            
            if model_weight > 0:
                channel_weighted_sum += model_prob[channel_idx] * model_weight
                total_channel_weight += model_weight
        
        # Normalize by total weight for this channel
        if total_channel_weight > 0:
            ensemble_prob[channel_idx] = channel_weighted_sum / total_channel_weight
        else:
            # Fallback to uniform weighting if no weights for this channel
            ensemble_prob[channel_idx] = np.mean([p[channel_idx] for _, p, _ in valid_models], axis=0)
    
    contributed = [name for name, _, _ in valid_models]
    
    # Log weight usage
    logger.info("Subregion-specific ensemble weights applied:")
    for channel_name in channel_names:
        channel_weights = subregion_weights.get(channel_name, {})
        weight_str = ", ".join([f"{name}:{w:.2f}" for name, w in channel_weights.items()])
        logger.info(f"  {channel_name}: {weight_str}")

    return ensemble_prob, contributed

def get_default_subregion_weights(valid_models: List[Tuple[str, np.ndarray, float]]) -> Dict[str, Dict[str, float]]:
    """
    Get default subregion-specific weights based on model strengths.
    
    Model Strengths (based on literature and empirical evidence):
    - SwinUNETR: Excellent at WT (edema boundaries), good at ET
    - SegResNet: Strong at ET (enhancing tumor core), good at TC
    - TTA-4: Reduces noise, good for all regions but conservative
    - nnU-Net: Good generalist, strong at TC
    """
    model_names = [name for name, _, _ in valid_models]
    
    # Default weights based on model characteristics
    default_weights = {
        "WT": {  # Whole Tumor (edema + core)
            "swinunetr": 0.40,  # Best at edema boundaries
            "segresnet": 0.25,  # Moderate
            "tta4": 0.20,       # Noise reduction
            "nnunet": 0.15      # Generalist
        },
        "TC": {  # Tumor Core (necrotic + enhancing)
            "segresnet": 0.35,  # Strong at core structures
            "nnunet": 0.30,     # Good generalist
            "swinunetr": 0.20,  # Moderate
            "tta4": 0.15        # Conservative
        },
        "ET": {  # Enhancing Tumor
            "swinunetr": 0.35,  # Good at enhancing regions
            "segresnet": 0.35,  # Strong at enhancing tumor
            "tta4": 0.20,       # Reduces false positives
            "nnunet": 0.10      # Weaker at enhancing
        }
    }
    
    # Filter weights to only include available models
    filtered_weights = {}
    for region, weights in default_weights.items():
        filtered_weights[region] = {
            name: weights.get(name, 0.25)  # Default fallback weight
            for name in model_names
        }
    
    return filtered_weights


def adaptive_subregion_weights(
    model_probs: List[Tuple[str, np.ndarray, float]],
    uncertainty_map: Optional[np.ndarray] = None,
    validation_performance: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generate adaptive subregion weights based on uncertainty or validation performance.
    
    Args:
        model_probs: List of (model_name, probability_map, base_weight) tuples
        uncertainty_map: Voxel-wise uncertainty map (higher = more uncertain)
        validation_performance: Dict mapping region -> model -> performance_metric
        
    Returns:
        Adaptive subregion weights
    """
    model_names = [name for name, _, _ in model_probs]
    
    if validation_performance is not None:
        # Base weights on validation performance
        return weights_from_performance(validation_performance, model_names)
    
    elif uncertainty_map is not None:
        # Base weights on uncertainty (give more weight to confident models)
        return weights_from_uncertainty(model_probs, uncertainty_map)
    
    else:
        # Fallback to default weights
        return get_default_subregion_weights(model_probs)

def weights_from_performance(
    validation_performance: Dict[str, Dict[str, float]],
    model_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Generate weights from validation performance metrics."""
    weights = {}
    
    for region, model_performances in validation_performance.items():
        # Normalize performance to sum to 1
        total_perf = sum(model_performances.get(name, 0.0) for name in model_names)
        
        if total_perf > 0:
            weights[region] = {
                name: model_performances.get(name, 0.0) / total_perf
                for name in model_names
            }
        else:
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(model_names)
            weights[region] = {name: uniform_weight for name in model_names}
    
    return weights

def weights_from_uncertainty(
    model_probs: List[Tuple[str, np.ndarray, float]],
    uncertainty_map: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Generate weights based on model uncertainty in different regions."""
    # This is a simplified implementation
    # In practice, you'd compute per-model uncertainty maps
    
    model_names = [name for name, _, _ in model_probs]
    default_weights = get_default_subregion_weights(model_probs)
    
    # Modify weights based on global uncertainty
    # Higher uncertainty -> more conservative weighting
    global_uncertainty = uncertainty_map.mean()
    
    if global_uncertainty > 0.7:  # High uncertainty
        # Give more weight to conservative models (TTA-4)
        for region in default_weights:
            if "tta4" in default_weights[region]:
                default_weights[region]["tta4"] *= 1.5
            # Renormalize
            total = sum(default_weights[region].values())
            for name in default_weights[region]:
                default_weights[region][name] /= total
    
    return default_weights

def validate_subregion_weights(
    subregion_weights: Dict[str, Dict[str, float]],
    model_names: List[str]
) -> bool:
    """Validate subregion weights for consistency."""
    # Check that all regions have weights for all models
    required_regions = ["WT", "TC", "ET"]
    
    for region in required_regions:
        if region not in subregion_weights:
            logger.error(f"Missing weights for region: {region}")
            return False
        
        for model_name in model_names:
            if model_name not in subregion_weights[region]:
                logger.error(f"Missing weight for model {model_name} in region {region}")
                return False
        
        # Check that weights sum to approximately 1
        total_weight = sum(subregion_weights[region].values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights for region {region} sum to {total_weight:.3f}, expected 1.0")
    
    logger.info("✅ Subregion weights validation passed")
    return True
