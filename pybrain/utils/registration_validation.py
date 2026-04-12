"""
CT-MRI Registration Validation Utilities
Medical Engineering Quality Assurance - Phase 2

Provides NMI-based validation for CT-MRI registration quality
to ensure safe CT boost application in clinical pipeline.
"""

import numpy as np
from typing import Optional, Dict, Any
from pybrain.io.logging_utils import get_logger

logger = get_logger("utils.registration_validation")

def compute_nmi(mri_data: np.ndarray, ct_data: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Normalized Mutual Information between MRI and CT data.
    
    Args:
        mri_data: MRI volume (e.g., FLAIR or T1c)
        ct_data: CT volume
        brain_mask: Optional brain mask to focus computation
        
    Returns:
        NMI value (higher = better alignment)
    """
    # Apply brain mask if provided
    if brain_mask is not None:
        mri_data = mri_data[brain_mask > 0]
        ct_data = ct_data[brain_mask > 0]
    else:
        # Flatten arrays
        mri_data = mri_data.flatten()
        ct_data = ct_data.flatten()
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(mri_data) & np.isfinite(ct_data)
    mri_data = mri_data[valid_mask]
    ct_data = ct_data[valid_mask]
    
    if len(mri_data) == 0:
        logger.warning("No valid data for NMI computation")
        return 0.0
    
    # Discretize data for mutual information computation
    # Use 64 bins for good balance between precision and computational efficiency
    n_bins = 64
    
    # Create 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
        mri_data, ct_data, bins=n_bins, density=True
    )
    
    # Compute marginal entropies
    p_x = hist_2d.sum(axis=1)
    p_y = hist_2d.sum(axis=0)
    
    # Remove zero probabilities
    p_x = p_x[p_x > 0]
    p_y = p_y[p_y > 0]
    hist_2d = hist_2d[hist_2d > 0]
    
    # Compute entropies
    H_x = -np.sum(p_x * np.log2(p_x))
    H_y = -np.sum(p_y * np.log2(p_y))
    H_xy = -np.sum(hist_2d * np.log2(hist_2d))
    
    # NMI = (H_x + H_y) / H_xy  — values > 1.0; well-aligned volumes typically 1.2–2.0.
    # Note: H_xy is joint entropy so must be > 0 for any non-trivial joint distribution.
    if H_xy > 0:
        nmi = (H_x + H_y) / H_xy
    else:
        nmi = 0.0

    return nmi

def validate_ct_mri_registration(
    mri_data: np.ndarray,
    ct_data: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    nmi_threshold: float = 1.3
) -> Dict[str, Any]:
    """
    Validate CT-MRI registration quality using NMI.
    
    Args:
        mri_data: MRI volume (single modality or first channel of multi-modal)
        ct_data: CT volume
        brain_mask: Optional brain mask
        nmi_threshold: Minimum NMI for acceptable alignment
        
    Returns:
        Validation results dictionary
    """
    logger.info("Validating CT-MRI registration quality...")
    
    try:
        # Check shape compatibility
        if mri_data.shape != ct_data.shape:
            logger.error(f"Shape mismatch: MRI {mri_data.shape} vs CT {ct_data.shape}")
            return {
                "valid": False,
                "error": "Shape mismatch",
                "nmi": 0.0,
                "threshold": nmi_threshold
            }
        
        # Compute NMI
        nmi = compute_nmi(mri_data, ct_data, brain_mask)
        valid = nmi >= nmi_threshold
        
        logger.info(f"NMI: {nmi:.4f} (threshold: {nmi_threshold})")
        
        if valid:
            logger.info("✅ CT-MRI registration validation PASSED")
        else:
            logger.warning("⚠️  CT-MRI registration validation FAILED")
            logger.warning("Consider disabling CT boost or checking registration quality")
        
        return {
            "valid": valid,
            "nmi": nmi,
            "threshold": nmi_threshold,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"CT-MRI registration validation failed: {e}")
        return {
            "valid": False,
            "error": str(e),
            "nmi": 0.0,
            "threshold": nmi_threshold
        }

def should_apply_ct_boost(
    mri_data: np.ndarray,
    ct_data: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    nmi_threshold: float = 1.3,
    force_enable: bool = False
) -> tuple[bool, Dict[str, Any]]:
    """
    Determine whether CT boost should be applied based on registration quality.
    
    Args:
        mri_data: MRI volume
        ct_data: CT volume
        brain_mask: Optional brain mask
        nmi_threshold: Minimum NMI for acceptable alignment
        force_enable: Force enable CT boost regardless of validation
        
    Returns:
        Tuple of (should_apply, validation_results)
    """
    if force_enable:
        logger.info("CT boost force enabled - skipping validation")
        return True, {"valid": True, "forced": True}
    
    validation_results = validate_ct_mri_registration(
        mri_data, ct_data, brain_mask, nmi_threshold
    )
    
    should_apply = validation_results["valid"]
    
    if should_apply:
        logger.info("CT boost approved based on registration quality")
    else:
        logger.warning("CT boost rejected due to poor registration quality")
        logger.warning("Falling back to MRI-only segmentation")
    
    return should_apply, validation_results
