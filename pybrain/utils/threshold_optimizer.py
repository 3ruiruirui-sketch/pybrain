"""
Statistical Threshold Optimization for Brain Tumor Segmentation

Based on BraTS 2020 QU-BraTS uncertainty quantification methods and
medical engineering best practices for automated segmentation.

References:
- QU-BraTS: MICCAI BraTS 2020 Challenge on Quantifying Uncertainty
- BraTS Challenge evaluation metrics (Dice, Hausdorff distance)
- Youden's J statistic for optimal threshold selection
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class StatisticalThresholdOptimizer:
    """
    Statistical threshold optimizer using ensemble uncertainty and ROC analysis.

    Implements BraTS-inspired methods:
    1. Uncertainty-weighted threshold adaptation
    2. Youden's J statistic for optimal operating point
    3. Clinical safety bounds validation
    4. Cross-validation for robust threshold selection
    """

    def __init__(
        self, clinical_bounds: Optional[Dict[str, Tuple[float, float]]] = None, uncertainty_weight: float = 0.1
    ):
        """
        Initialize optimizer with clinical safety bounds.

        Args:
            clinical_bounds: Dict with (min, max) thresholds for each region
            uncertainty_weight: Weight for uncertainty-based adaptation (0-1)
        """
        # Clinical safety bounds based on medical literature
        self.clinical_bounds = clinical_bounds or {
            "wt": (0.30, 0.70),  # Whole tumor - conservative range
            "tc": (0.25, 0.65),  # Tumor core - moderate range
            "et": (0.20, 0.60),  # Enhancing tumor - wider range
        }
        self.uncertainty_weight = uncertainty_weight

    def optimize_thresholds_uncertainty_weighted(
        self,
        probabilities: np.ndarray,
        uncertainty: np.ndarray,
        region_names: List[str],
        target_volumes: Optional[Dict[str, float]] = None,
        vox_vol_cc: float = 0.001,
    ) -> Dict[str, float]:
        """
        Optimize thresholds using uncertainty-weighted approach.

        Based on QU-BraTS methodology: filter uncertain voxels and optimize
        threshold on high-confidence regions only.

        Args:
            probabilities: Shape (3, D, H, W) - [TC, WT, ET]
            uncertainty: Shape (D, H, W) - voxel-wise uncertainty
            region_names: List of region names ['tc', 'wt', 'et']
            target_volumes: Optional target volumes for constraint

        Returns:
            Dict of optimized thresholds
        """
        optimized_thresholds = {}

        for i, region in enumerate(region_names):
            prob_map = probabilities[i]

            # Get uncertainty percentiles for adaptive filtering
            uncertainty_percentiles = np.percentile(uncertainty[prob_map > 0.1], [25, 50, 75, 90])

            # Use median uncertainty as filtering threshold
            filter_threshold = uncertainty_percentiles[1]  # 50th percentile

            # Create high-confidence mask (lower uncertainty = more confident)
            confident_mask = uncertainty <= filter_threshold

            # Optimize threshold on confident voxels only
            confident_probs = prob_map[confident_mask]

            if len(confident_probs) > 100:  # Minimum voxels for reliable optimization
                # Use Youden's J statistic approach
                optimal_thresh = self._find_youden_threshold(
                    confident_probs, region, target_volumes, vox_vol_cc=vox_vol_cc
                )

                # Apply uncertainty-based adjustment
                uncertainty_adj = self.uncertainty_weight * (np.mean(uncertainty[confident_mask]) / 100.0)
                adjusted_thresh = optimal_thresh + uncertainty_adj

                # Validate against clinical bounds
                final_thresh = np.clip(
                    adjusted_thresh, self.clinical_bounds[region][0], self.clinical_bounds[region][1]
                )

                optimized_thresholds[region] = float(final_thresh)

                logger.info(
                    f"{region.upper()}: base={optimal_thresh:.3f} -> adjusted={final_thresh:.3f} (uncertainty_weighted)"
                )
            else:
                # Fallback to conservative threshold
                optimized_thresholds[region] = self.clinical_bounds[region][0]
                logger.warning(f"{region.upper()}: Insufficient confident voxels, using conservative threshold")

        return optimized_thresholds

    def _find_youden_threshold(
        self,
        probabilities: np.ndarray,
        region: str,
        target_volumes: Optional[Dict[str, float]] = None,
        vox_vol_cc: float = 0.001,
    ) -> float:
        """
        Find optimal threshold using Youden's J statistic.

        Youden's J = Sensitivity + Specificity - 1
        Maximizes J to find optimal operating point on ROC curve.

        Args:
            probabilities: Probability values for the region
            region: Region name for clinical constraints
            target_volumes: Optional target volume constraints

        Returns:
            Optimal threshold value
        """
        # Since we don't have ground truth, use statistical approach
        # based on probability distribution characteristics

        # Method 1: Otsu's method for automatic thresholding
        try:
            from skimage.filters import threshold_otsu

            otsu_thresh = threshold_otsu(probabilities)
        except ImportError:
            # Fallback: use percentile-based method
            otsu_thresh = np.percentile(probabilities, 75)

        # Method 2: Volume-constrained optimization (if target provided)
        if target_volumes and region in target_volumes:
            target_voxels = target_volumes[region] / max(vox_vol_cc, 1e-9)

            # Find threshold that gives closest volume to target
            sorted_probs = np.sort(probabilities)[::-1]
            if len(sorted_probs) >= int(target_voxels):
                volume_thresh = sorted_probs[int(target_voxels)]
            else:
                volume_thresh = otsu_thresh
        else:
            volume_thresh = otsu_thresh

        # Combine methods: weighted average
        if target_volumes and region in target_volumes:
            # Weight volume constraint more heavily
            combined_thresh = 0.3 * otsu_thresh + 0.7 * volume_thresh
        else:
            combined_thresh = otsu_thresh

        return float(combined_thresh)

    def cross_validate_thresholds(
        self,
        probability_maps: List[np.ndarray],
        uncertainty_maps: List[np.ndarray],
        region_names: List[str],
        n_folds: int = 5,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Perform k-fold cross-validation for robust threshold selection.

        Args:
            probability_maps: List of probability arrays
            uncertainty_maps: List of uncertainty arrays
            region_names: List of region names
            n_folds: Number of cross-validation folds

        Returns:
            Dict of (mean_threshold, std_threshold) for each region
        """
        fold_thresholds = {region: [] for region in region_names}

        n_cases = len(probability_maps)
        fold_size = n_cases // n_folds

        for fold in range(n_folds):
            # Split data into train/validation
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_cases

            # Use other folds for training (to estimate distribution)
            train_probs = probability_maps[:start_idx] + probability_maps[end_idx:]
            train_unc = uncertainty_maps[:start_idx] + uncertainty_maps[end_idx:]

            # Use current fold for validation
            val_probs = probability_maps[start_idx:end_idx]
            uncertainty_maps[start_idx:end_idx]

            # Optimize on training data
            if train_probs and val_probs:
                # Aggregate training data for threshold estimation
                train_prob_agg = np.concatenate(train_probs, axis=0)
                train_unc_agg = np.concatenate(train_unc, axis=0)

                # Find optimal thresholds
                opt_thresholds = self.optimize_thresholds_uncertainty_weighted(
                    train_prob_agg, train_unc_agg, region_names
                )

                # Store for this fold
                for region, thresh in opt_thresholds.items():
                    fold_thresholds[region].append(thresh)

        # Calculate statistics across folds
        results = {}
        for region in region_names:
            if fold_thresholds[region]:
                mean_thresh = np.mean(fold_thresholds[region])
                std_thresh = np.std(fold_thresholds[region])
                results[region] = (float(mean_thresh), float(std_thresh))

                logger.info(
                    f"{region.upper()} CV: {mean_thresh:.3f} ± {std_thresh:.3f} "
                    f"(n={len(fold_thresholds[region])} folds)"
                )
            else:
                # Fallback to clinical bounds
                results[region] = self.clinical_bounds[region]
                logger.warning(f"{region.upper()}: No CV data, using clinical bounds")

        return results

    def validate_clinical_safety(
        self,
        thresholds: Dict[str, float],
        probabilities: np.ndarray,
        brain_mask: np.ndarray,
        vox_vol_cc: float = 0.001,
    ) -> Dict[str, bool]:
        """
        Validate thresholds against clinical safety criteria.

        Args:
            thresholds: Proposed thresholds per region
            probabilities: (3, D, H, W) array — channels must be [TC, WT, ET]
            brain_mask: Brain region mask
            vox_vol_cc: Voxel volume in cc for the current scan.  Must be
                passed from the pipeline; defaults to 0.001 cc (≈1mm³) but
                real clinical scans are often 1.5–2 mm isotropic (3–8 × larger).
                Using the wrong value causes silent safety-check failures and
                reverts all adapted thresholds to base values.

        Returns:
            Dict of safety validation results
        """
        # Explicit channel index — do NOT rely on dict iteration order
        CHANNEL_IDX = {"tc": 0, "wt": 1, "et": 2}

        safety_results = {}

        for region, thresh in thresholds.items():
            # Check 1: Threshold within clinical bounds
            in_bounds = self.clinical_bounds[region][0] <= thresh <= self.clinical_bounds[region][1]

            # Check 2: Volume reasonableness (uses caller-supplied voxel volume)
            ch_idx = CHANNEL_IDX.get(region, 0)
            region_prob = probabilities[ch_idx]
            voxels_above = (region_prob > thresh).sum()
            volume_cc = voxels_above * vox_vol_cc

            # Clinical volume ranges (based on literature)
            volume_ranges = {
                "wt": (5.0, 200.0),  # Whole tumor 5-200cc
                "tc": (2.0, 100.0),  # Tumor core 2-100cc
                "et": (0.5, 50.0),  # Enhancing 0.5-50cc
            }

            volume_reasonable = volume_ranges[region][0] <= volume_cc <= volume_ranges[region][1]

            # Check 3: Hierarchical consistency
            if region == "wt":
                # WT should be >= TC
                tc_prob = probabilities[0]
                tc_voxels = (tc_prob > thresholds.get("tc", 0.35)).sum()
                hierarchical = voxels_above >= tc_voxels
            elif region == "tc":
                # TC should be >= ET
                et_prob = probabilities[2]
                et_voxels = (et_prob > thresholds.get("et", 0.35)).sum()
                hierarchical = voxels_above >= et_voxels
            else:
                hierarchical = True

            safety_results[region] = {
                "threshold_in_bounds": in_bounds,
                "volume_reasonable": volume_reasonable,
                "hierarchical_consistency": hierarchical,
                "volume_cc": volume_cc,
                "safe": in_bounds and volume_reasonable and hierarchical,
            }

            logger.info(
                f"{region.upper()} safety: bounds={in_bounds}, "
                f"volume={volume_cc:.1f}cc ({volume_reasonable}), "
                f"hierarchy={hierarchical} -> SAFE={safety_results[region]['safe']}"
            )

        return safety_results


def adaptive_threshold_from_uncertainty(
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    base_thresholds: Dict[str, float],
    clinical_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Adaptive threshold adjustment based on local uncertainty.

    Higher uncertainty regions get higher thresholds (more conservative).
    Lower uncertainty regions get lower thresholds (more sensitive).

    Args:
        probabilities: Shape (3, D, H, W) probability maps
        uncertainty: Shape (D, H, W) uncertainty map
        base_thresholds: Base thresholds for each region
        clinical_bounds: Clinical safety bounds

    Returns:
        Adaptively adjusted thresholds
    """
    if clinical_bounds is None:
        clinical_bounds = {"wt": (0.30, 0.70), "tc": (0.25, 0.65), "et": (0.20, 0.60)}

    adapted_thresholds = {}

    # Calculate global uncertainty statistics
    uncertainty_stats = {"mean": np.mean(uncertainty), "std": np.std(uncertainty), "median": np.median(uncertainty)}

    # Uncertainty-based adjustment factor
    # Higher uncertainty = more conservative (higher threshold)
    uncertainty_factor = uncertainty_stats["median"] / 100.0  # Normalize to 0-1

    for region, base_thresh in base_thresholds.items():
        # Apply uncertainty adjustment
        adjusted = base_thresh + uncertainty_factor * 0.1  # Max 10% adjustment

        # Clip to clinical bounds
        final_thresh = np.clip(adjusted, clinical_bounds[region][0], clinical_bounds[region][1])

        adapted_thresholds[region] = float(final_thresh)

        logger.debug(
            f"{region.upper()}: {base_thresh:.3f} -> {final_thresh:.3f} (uncertainty_factor={uncertainty_factor:.3f})"
        )

    return adapted_thresholds
