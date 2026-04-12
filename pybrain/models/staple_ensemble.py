"""
STAPLE (Simultaneous Truth and Performance Level Estimation) Ensemble
Data-driven ensemble weighting for brain tumor segmentation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.staple")


class STAPLEEnsemble:
    """
    STAPLE algorithm for combining multiple segmentations.
    Estimates sensitivity and specificity of each rater and computes
    probabilistic fusion of segmentations.
    """
    
    def __init__(self, max_iter: int = 100, tolerance: float = 1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def _compute_performance_parameters(
        self, 
        segmentations: np.ndarray,
        truth_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sensitivity and specificity for each rater.
        
        Args:
            segmentations: Shape (n_raters, n_voxels) binary segmentations
            truth_prob: Shape (n_voxels,) current estimate of truth probability
            
        Returns:
            sensitivities: Shape (n_raters,) sensitivity for each rater
            specificities: Shape (n_raters,) specificity for each rater
        """
        n_raters = segmentations.shape[0]
        sensitivities = np.zeros(n_raters)
        specificities = np.zeros(n_raters)
        
        for i in range(n_raters):
            seg_i = segmentations[i]
            
            # Sensitivity: P(seg=1|truth=1)
            truth_mask = truth_prob > 0.5
            if truth_mask.sum() > 0:
                sensitivities[i] = (seg_i[truth_mask]).sum() / truth_mask.sum()
            
            # Specificity: P(seg=0|truth=0)
            non_truth_mask = ~truth_mask
            if non_truth_mask.sum() > 0:
                specificities[i] = (1 - seg_i[non_truth_mask]).sum() / non_truth_mask.sum()
                
        # Clip to valid range
        sensitivities = np.clip(sensitivities, 0.01, 0.99)
        specificities = np.clip(specificities, 0.01, 0.99)
        
        return sensitivities, specificities
    
    def _compute_truth_probability(
        self,
        segmentations: np.ndarray,
        sensitivities: np.ndarray,
        specificities: np.ndarray
    ) -> np.ndarray:
        """
        Compute truth probability given rater performance parameters.

        Vectorised implementation — replaces the original per-voxel Python
        loop which was O(n_raters × n_voxels) in pure Python and would take
        hours on a full BraTS volume (~8.9 M voxels).

        Args:
            segmentations: Shape (n_raters, n_voxels) binary {0,1}
            sensitivities: Shape (n_raters,)
            specificities: Shape (n_raters,)

        Returns:
            truth_prob: Shape (n_voxels,) updated truth probability
        """
        # log-odds contribution when rater says 1: log(p_i / (1-q_i))
        # log-odds contribution when rater says 0: log((1-p_i) / q_i)
        # where p_i = sensitivity, q_i = specificity
        eps = 1e-8
        log_when_1 = np.log(sensitivities / (1.0 - specificities + eps) + eps)   # (n_raters,)
        log_when_0 = np.log((1.0 - sensitivities + eps) / (specificities + eps))  # (n_raters,)

        # segmentations: (n_raters, n_voxels)  float32  {0, 1}
        # log_when_1 / log_when_0: (n_raters, 1)  broadcast over voxels
        log_odds = (
            segmentations       * log_when_1[:, np.newaxis] +
            (1 - segmentations) * log_when_0[:, np.newaxis]
        ).sum(axis=0)  # (n_voxels,)

        truth_prob = 1.0 / (1.0 + np.exp(-log_odds))
        return np.clip(truth_prob, 0.001, 0.999)
    
    def fit(
        self,
        segmentations: np.ndarray,
        initial_truth: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit STAPLE model to segmentations.
        
        Args:
            segmentations: Shape (n_raters, n_voxels) binary segmentations
            initial_truth: Optional initial truth probability estimate
            
        Returns:
            truth_prob: Shape (n_voxels,) final truth probability
            sensitivities: Shape (n_raters,) sensitivity for each rater
            specificities: Shape (n_raters,) specificity for each rater
        """
        n_raters, n_voxels = segmentations.shape
        
        # Initialize truth probability
        if initial_truth is not None:
            truth_prob = initial_truth.copy()
        else:
            # Simple majority vote as initialization
            truth_prob = segmentations.mean(axis=0)
        
        # EM algorithm
        prev_truth = truth_prob.copy()
        
        for iteration in range(self.max_iter):
            # E-step: Compute performance parameters
            sensitivities, specificities = self._compute_performance_parameters(
                segmentations, truth_prob
            )
            
            # M-step: Update truth probability
            truth_prob = self._compute_truth_probability(
                segmentations, sensitivities, specificities
            )
            
            # Check convergence
            diff = np.mean(np.abs(truth_prob - prev_truth))
            if diff < self.tolerance:
                logger.debug(f"STAPLE converged at iteration {iteration}")
                break
                
            prev_truth = truth_prob.copy()
        else:
            logger.warning(f"STAPLE did not converge after {self.max_iter} iterations")
        
        logger.info(f"STAPLE completed: avg sensitivity={sensitivities.mean():.3f}, "
                   f"avg specificity={specificities.mean():.3f}")
        
        return truth_prob, sensitivities, specificities


def run_staple_ensemble(
    model_probs: Dict[str, np.ndarray],
    subregion_weights: Optional[Dict] = None,
) -> np.ndarray:
    """
    Run STAPLE ensemble on model probability maps.

    Channel mapping (BraTS convention):  0=TC, 1=WT, 2=ET.
    Binarisation thresholds match pipeline calibrated values (tc=0.35, wt=0.35, et=0.35).

    Args:
        model_probs: Dict[model_name -> (3, D, H, W) probability array]
        subregion_weights: Unused placeholder for future subregion blending.

    Returns:
        ensemble_prob: (3, D, H, W) STAPLE-fused probability array
    """
    logger = get_logger("models.staple")

    if len(model_probs) < 2:
        logger.warning("STAPLE requires at least 2 models, returning single model output")
        return list(model_probs.values())[0]

    # Calibrated per-channel binarisation thresholds  [TC, WT, ET]
    CHANNEL_THRESHOLDS = [0.35, 0.35, 0.35]

    model_names = list(model_probs.keys())
    n_models = len(model_names)
    n_channels = model_probs[model_names[0]].shape[0]
    spatial_shape = model_probs[model_names[0]].shape[1:]

    ensemble_prob = np.zeros_like(model_probs[model_names[0]])

    for channel in range(n_channels):
        threshold = CHANNEL_THRESHOLDS[channel] if channel < len(CHANNEL_THRESHOLDS) else 0.35
        binary_segs = []

        for model_name in model_names:
            prob_map = model_probs[model_name][channel]
            binary_seg = (prob_map > threshold).astype(np.float32)
            binary_segs.append(binary_seg.flatten())

        segmentations = np.stack(binary_segs, axis=0)

        staple = STAPLEEnsemble()
        truth_prob, sensitivities, specificities = staple.fit(segmentations)

        ensemble_prob[channel] = truth_prob.reshape(spatial_shape)

    logger.info(f"STAPLE ensemble completed for {n_models} models, {n_channels} channels")
    return ensemble_prob


def validate_staple_weights(
    model_probs: Dict[str, np.ndarray],
    ensemble_weights: Dict[str, float]
) -> bool:
    """
    Validate that ensemble weights are suitable for STAPLE.

    Only considers weights for models that are actually present in
    model_probs (i.e. ran and produced output).  Zero-weight models
    like swinunetr=0.0 or nnunet=0.0 are absent from model_probs at
    runtime and must not be included — they would make weight_ratio
    infinite and always return False.
    """
    if len(model_probs) < 2:
        return False

    # Only look at active models (present in model_probs AND have a nonzero weight)
    active_weights = [
        ensemble_weights.get(name, 1.0)
        for name in model_probs
        if ensemble_weights.get(name, 1.0) > 0
    ]
    if len(active_weights) < 2:
        logger.warning("STAPLE requires at least 2 models with non-zero weight")
        return False

    weight_ratio = max(active_weights) / (min(active_weights) + 1e-8)
    if weight_ratio > 10.0:
        logger.warning(
            f"Active ensemble weights highly skewed (ratio={weight_ratio:.1f}); "
            "STAPLE may still run but results may be unbalanced"
        )
        # Still allow STAPLE — skewed weights are a warning, not a blocker
    return True
