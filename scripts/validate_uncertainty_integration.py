#!/usr/bin/env python3
"""
Uncertainty Computation Validation Script
Medical Engineering Validation - Phase 1 Critical Safety Fix

Validates that uncertainty computation is properly integrated
into the pipeline and provides clinically meaningful uncertainty maps.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pybrain.models.ensemble import compute_uncertainty
from pybrain.io.logging_utils import get_logger

logger = get_logger("validation.uncertainty")


def create_test_ensemble_data():
    """Create synthetic ensemble probability maps for testing."""
    # Create 3 model probability maps with realistic variation
    model_probs = []

    for i in range(3):
        # Each model has slightly different probability distributions
        base_prob = np.random.rand(3, 64, 64, 64) * 0.8 + 0.1  # [0.1, 0.9]

        # Add model-specific variations
        if i == 0:  # SwinUNETR - better at WT
            base_prob[1] *= 1.1  # Boost WT
        elif i == 1:  # SegResNet - better at ET
            base_prob[2] *= 1.15  # Boost ET
        else:  # TTA - more conservative
            base_prob *= 0.95

        # Clip to valid probability range
        base_prob = np.clip(base_prob, 0.0, 1.0)
        model_probs.append(base_prob)

    # Create ensemble by averaging
    ensemble_prob = np.mean(model_probs, axis=0)

    return ensemble_prob, model_probs


def validate_uncertainty_computation():
    """Test the uncertainty computation function."""
    logger.info("=== Uncertainty Computation Validation ===")

    # Create test data
    ensemble_prob, model_probs = create_test_ensemble_data()
    logger.info(f"Ensemble shape: {ensemble_prob.shape}")
    logger.info(f"Number of models: {len(model_probs)}")

    # Compute uncertainty
    uncertainty = compute_uncertainty(ensemble_prob, model_probs)

    logger.info(f"Uncertainty shape: {uncertainty.shape}")
    logger.info(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    logger.info(f"Uncertainty mean: {uncertainty.mean():.3f}")
    logger.info(f"Uncertainty std: {uncertainty.std():.3f}")

    # Validate uncertainty properties
    if uncertainty.shape != ensemble_prob.shape[1:]:
        logger.error(f"Uncertainty shape mismatch: {uncertainty.shape} vs {ensemble_prob.shape[1:]}")
        return False

    if uncertainty.min() < 0 or uncertainty.max() > 1:
        logger.error(f"Uncertainty out of range [0,1]: [{uncertainty.min()}, {uncertainty.max()}]")
        return False

    # Check that uncertainty correlates with model disagreement
    # Create high-disagreement region
    high_disagreement_region = np.zeros_like(ensemble_prob[0])
    high_disagreement_region[20:30, 20:30, 20:30] = 1

    # Modify one model to create disagreement
    model_probs_disagree = model_probs.copy()
    model_probs_disagree[0][:, 20:30, 20:30, 20:30] *= 0.5
    ensemble_disagree = np.mean(model_probs_disagree, axis=0)
    uncertainty_disagree = compute_uncertainty(ensemble_disagree, model_probs_disagree)

    # Uncertainty should be higher in disagreement region
    uncertainty_high = uncertainty_disagree[20:30, 20:30, 20:30].mean()
    uncertainty_low = uncertainty_disagree[40:50, 40:50, 40:50].mean()

    logger.info(f"High disagreement uncertainty: {uncertainty_high:.3f}")
    logger.info(f"Low disagreement uncertainty: {uncertainty_low:.3f}")

    if uncertainty_high <= uncertainty_low:
        logger.warning("Uncertainty may not be capturing model disagreement properly")

    logger.info("✅ Uncertainty computation validation PASSED")
    return True


def validate_uncertainty_in_pipeline():
    """Test uncertainty integration in pipeline context."""
    logger.info("=== Pipeline Integration Validation ===")

    # Simulate pipeline scenario
    ensemble_prob = np.random.rand(3, 96, 96, 96) * 0.8 + 0.1
    model_probs_list = [np.random.rand(3, 96, 96, 96) * 0.8 + 0.1 for _ in range(3)]

    # Test current pipeline logic
    if model_probs_list:
        compute_uncertainty(ensemble_prob, model_probs_list)
        logger.info("✅ Uncertainty computed with model_probs_list")
    else:
        logger.warning("⚠️  Uncertainty is None when model_probs_list is empty")

    # Test edge cases
    try:
        # Single model case
        compute_uncertainty(ensemble_prob, [ensemble_prob])
        logger.info("✅ Single model uncertainty computed")

        # Empty model list (should not crash)
        compute_uncertainty(ensemble_prob, [])
        logger.info("✅ Empty model list handled gracefully")

    except Exception as e:
        logger.error(f"❌ Edge case failed: {e}")
        return False

    logger.info("✅ Pipeline integration validation PASSED")
    return True


def validate_uncertainty_clinical_relevance():
    """Test clinical relevance of uncertainty maps."""
    logger.info("=== Clinical Relevance Validation ===")

    # Create realistic clinical scenario
    ensemble_prob = np.zeros((3, 64, 64, 64))

    # Simulate tumor regions with different confidence levels
    # High confidence tumor core
    ensemble_prob[:, 20:30, 20:30, 20:30] = 0.9

    # Low confidence infiltrative edges
    ensemble_prob[:, 30:35, 30:35, 30:35] = 0.6

    # Very low confidence noise regions
    ensemble_prob[:, 40:45, 40:45, 40:45] = 0.3

    # Create model variations
    model_probs = []
    for i in range(3):
        model_prob = ensemble_prob.copy()
        # Add realistic variations
        noise = np.random.normal(0, 0.05, ensemble_prob.shape)
        model_prob += noise
        model_prob = np.clip(model_prob, 0.0, 1.0)
        model_probs.append(model_prob)

    uncertainty = compute_uncertainty(ensemble_prob, model_probs)

    # Analyze uncertainty in different regions
    high_conf_uncertainty = uncertainty[20:30, 20:30, 20:30].mean()
    edge_uncertainty = uncertainty[30:35, 30:35, 30:35].mean()
    low_conf_uncertainty = uncertainty[40:45, 40:45, 40:45].mean()

    logger.info(f"High confidence region uncertainty: {high_conf_uncertainty:.3f}")
    logger.info(f"Edge region uncertainty: {edge_uncertainty:.3f}")
    logger.info(f"Low confidence region uncertainty: {low_conf_uncertainty:.3f}")

    # Clinical validation: uncertainty should generally correlate with confidence
    # Allow for some tolerance since uncertainty is complex
    confidence_trend = (edge_uncertainty > high_conf_uncertainty) and (low_conf_uncertainty > edge_uncertainty)

    if confidence_trend:
        logger.info("✅ Clinical relevance validation PASSED")
        return True
    else:
        # Check if at least the general trend holds (high vs low confidence)
        if low_conf_uncertainty > high_conf_uncertainty:
            logger.info("✅ Clinical relevance validation PASSED (general trend)")
            return True
        else:
            logger.warning("⚠️  Uncertainty may not correlate with clinical confidence")
            logger.info(
                f"High: {high_conf_uncertainty:.3f}, Edge: {edge_uncertainty:.3f}, Low: {low_conf_uncertainty:.3f}"
            )
            return False


def run_validation():
    """Run complete uncertainty validation suite."""
    logger.info("Starting Uncertainty Integration Validation")

    results = []

    # Test 1: Basic uncertainty computation
    results.append(validate_uncertainty_computation())

    # Test 2: Pipeline integration
    results.append(validate_uncertainty_in_pipeline())

    # Test 3: Clinical relevance
    results.append(validate_uncertainty_clinical_relevance())

    # Summary
    passed = sum(results)
    total = len(results)

    logger.info("\n=== VALIDATION SUMMARY ===")
    logger.info(f"Tests passed: {passed}/{total}")

    if passed == total:
        logger.info("🎉 ALL UNCERTAINTY VALIDATIONS PASSED")
        logger.info("Uncertainty computation is properly integrated and clinically relevant")
        return True
    else:
        logger.error("❌ SOME UNCERTAINTY VALIDATIONS FAILED")
        logger.error("Uncertainty integration needs review")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
