"""Core processing primitives for the PY-BRAIN segmentation pipeline."""

from pybrain.core.input_validator import (
    ValidationResult,
    validate_input,
    validate_input_tensor,
    validate_nifti_loadable,
    validate_shape_consistency,
)
from pybrain.core.labels import canonical_labels, is_pipeline_convention
from pybrain.core.metrics import compute_dice, compute_volume_cc
from pybrain.core.normalization import norm01, zscore_robust
from pybrain.core.output_checker import (
    OutputCheckResult,
    check_hierarchy_violations,
    check_output_not_empty,
    check_probability_range,
    sanity_check_batch,
    sanity_check_segmentation,
)

__all__ = [
    "norm01",
    "zscore_robust",
    "compute_dice",
    "compute_volume_cc",
    "canonical_labels",
    "is_pipeline_convention",
    "sanity_check_segmentation",
    "sanity_check_batch",
    "check_output_not_empty",
    "check_probability_range",
    "check_hierarchy_violations",
    "OutputCheckResult",
    "validate_input",
    "validate_input_tensor",
    "validate_nifti_loadable",
    "validate_shape_consistency",
    "ValidationResult",
]
