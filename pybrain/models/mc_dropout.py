"""
MC-Dropout Uncertainty Quantification for Brain Tumor Segmentation
Implements Monte Carlo Dropout for calibrated uncertainty estimation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, cast
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.mc_dropout")


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Enable dropout layers during inference for MC-Dropout.

    Args:
        model: PyTorch model to enable dropout for
    """
    model.train()  # Enable dropout layers
    # Disable batch normalization updates
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()

    logger.debug("MC-Dropout enabled for model")


def run_mc_dropout_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    n_samples: int = 20,
    sw_config: Optional[Dict] = None,
    model_device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MC-Dropout inference for uncertainty quantification.

    Args:
        model: PyTorch model
        input_tensor: Input tensor (1, C, D, H, W)
        device: Device for computation
        n_samples: Number of MC samples
        sw_config: Sliding window configuration
        model_device: Model device (different from computation device)

    Returns:
        mean_prob: Mean probability across samples (C, D, H, W)
        std_prob: Standard deviation across samples (C, D, H, W)
    """
    if model_device is None:
        model_device = device

    _dev = model_device if model_device is not None else device
    model = model.to(_dev)

    # Enable MC-Dropout
    enable_mc_dropout(model)

    # Get sliding window parameters
    if sw_config is None:
        sw_config = {}

    roi_size = tuple(sw_config.get("roi_size", (240, 240, 160)))  # Match SegResNet training
    sw_batch_size = sw_config.get("sw_batch_size", 1)
    overlap = sw_config.get("overlap", 0.5)

    # Import sliding window inferer
    from monai.inferers.inferer import SlidingWindowInferer

    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian")

    # Store predictions
    predictions = []

    logger.info(f"Running MC-Dropout inference with {n_samples} samples...")

    # SegResNet bundle expects [T1c, T1, T2, FLAIR] — apply same permutation as
    # run_segresnet_inference: pipeline stacks [FLAIR, T1, T1c, T2] → indices (2,1,3,0)
    input_permuted = input_tensor[:, (2, 1, 3, 0), :, :, :].clone().to(_dev)

    with torch.no_grad():
        for sample_idx in range(n_samples):
            # Apply stochastic forward pass with correctly ordered channels
            logits = cast(torch.Tensor, inferer(input_permuted, model))
            probs = torch.sigmoid(logits.cpu())

            # Convert to numpy and store
            prob_np = probs[0].numpy()  # Remove batch dimension
            predictions.append(prob_np)

            if (sample_idx + 1) % 5 == 0:
                logger.debug(f"MC-Dropout sample {sample_idx + 1}/{n_samples} completed")

    # Convert to numpy array
    predictions = np.stack(predictions, axis=0)  # Shape: (n_samples, C, D, H, W)

    # Compute statistics
    mean_prob = predictions.mean(axis=0)
    std_prob = predictions.std(axis=0)

    # Additional uncertainty metrics
    entropy = -np.sum(mean_prob * np.log(mean_prob + 1e-8), axis=0)
    entropy - np.mean(-np.sum(predictions * np.log(predictions + 1e-8), axis=1), axis=0)

    logger.info(f"MC-Dropout completed: mean std={std_prob.mean():.4f}, mean entropy={entropy.mean():.4f}")

    return mean_prob, std_prob


def run_mc_dropout_ensemble(
    model_probs: Dict[str, np.ndarray],
    models: Dict[str, torch.nn.Module],
    input_tensor: torch.Tensor,
    device: torch.device,
    n_samples: int = 10,
    sw_configs: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run MC-Dropout on multiple models for enhanced uncertainty estimation.

    Args:
        model_probs: Existing model probability maps
        models: Dictionary of loaded models
        input_tensor: Input tensor
        device: Device for computation
        n_samples: Number of MC samples per model
        sw_configs: Sliding window configurations per model

    Returns:
        mc_probs: MC-Dropout enhanced probability maps
        mc_uncertainties: MC-Dropout uncertainty maps
    """
    logger = get_logger("models.mc_dropout")

    mc_probs = {}
    mc_uncertainties = {}

    for model_name, model in models.items():
        if model_name not in model_probs:
            continue

        logger.info(f"Running MC-Dropout for {model_name}...")

        sw_config = sw_configs.get(model_name, {}) if sw_configs else {}

        try:
            mean_prob, std_prob = run_mc_dropout_inference(model, input_tensor, device, n_samples, sw_config)

            # Blend original prediction with MC-Dropout
            original_prob = model_probs[model_name]
            blend_factor = 0.7  # Favor MC-Dropout but retain original

            blended_prob = blend_factor * mean_prob + (1 - blend_factor) * original_prob

            mc_probs[model_name] = blended_prob
            mc_uncertainties[model_name] = std_prob

        except Exception as e:
            logger.warning(f"MC-Dropout failed for {model_name}: {e}")
            # Fallback to original probability and zero uncertainty
            mc_probs[model_name] = model_probs[model_name]
            mc_uncertainties[model_name] = np.zeros_like(model_probs[model_name])

    return mc_probs, mc_uncertainties


def validate_mc_dropout_config(config: Dict) -> bool:
    """
    Validate MC-Dropout configuration.

    Args:
        config: MC-Dropout configuration

    Returns:
        bool: True if configuration is valid
    """
    n_samples = config.get("n_samples", 20)

    if n_samples < 5:
        logger.warning(f"MC-Dropout n_samples={n_samples} is too low, recommend >=10")
        return False

    if n_samples > 100:
        logger.warning(f"MC-Dropout n_samples={n_samples} is very high, may be slow")

    return True


def compute_uncertainty_metrics(
    predictions: np.ndarray, mean_prob: np.ndarray, std_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive uncertainty metrics.

    Args:
        predictions: MC predictions (n_samples, C, D, H, W)
        mean_prob: Mean probability (C, D, H, W)
        std_prob: Standard deviation (C, D, H, W)

    Returns:
        Dictionary of uncertainty metrics
    """
    metrics = {}

    # Basic statistics
    metrics["mean_std"] = float(std_prob.mean())
    metrics["max_std"] = float(std_prob.max())
    metrics["mean_entropy"] = float(-np.sum(mean_prob * np.log(mean_prob + 1e-8), axis=0).mean())

    # Predictive uncertainty decomposition
    epistemic = std_prob**2  # Model uncertainty
    aleatoric = np.mean(predictions * (1 - predictions), axis=0)  # Data uncertainty

    metrics["mean_epistemic"] = float(epistemic.mean())
    metrics["mean_aleatoric"] = float(aleatoric.mean())

    # Mutual information
    entropy = -np.sum(mean_prob * np.log(mean_prob + 1e-8), axis=0)
    expected_entropy = -np.mean(np.sum(predictions * np.log(predictions + 1e-8), axis=1), axis=0)
    mutual_info = entropy - expected_entropy

    metrics["mean_mutual_info"] = float(mutual_info.mean())

    return metrics
