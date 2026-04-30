"""
Input validation for brain MRI data before segmentation.

Adapted from BrainLesion/BraTS input_sanity_check pattern:
- Validates NIfTI file integrity (loadable, non-empty)
- Checks required modalities (T1, T1c, T2, FLAIR)
- Verifies shape consistency across modalities
- Validates intensity distributions are reasonable for MRI
- Checks voxel spacing for physical-unit consistency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import nibabel as nib
import numpy as np

from pybrain.io.logging_utils import get_logger

logger = get_logger("core.input_validator")

# Required BraTS modalities
REQUIRED_MODALITIES = ["t1", "t1c", "t2", "flair"]

# Order matters: more specific patterns first
_MODALITY_PATTERNS = {
    "t1c":   [r"_t1ce(?:\W|$|_)", r"_t1_?ce", r"_t1c(?:\W|$|_)", r"_t1\+c", r"_t1_gd",
             r"^t1c(?:\W|$|_)", r"t1c\.(?:nii|nrrd)"],  # t1c.nii.gz
    "flair": [r"_flair(?:\W|$|_)", r"_t2_?flair", r"^flair(?:\W|$|_)"],
    "t2":    [r"_t2(?:\W|$|_)(?!flair)", r"^t2(?:\W|$|_)(?!flair)"],   # t2 but not t2_flair
    "t1":    [r"_t1(?:\W|$|_)(?!ce|_ce|c|\+c|_gd)", r"^t1(?:\W|$|_)(?!ce|_ce|c|\+c|_gd)"],  # t1 but not t1ce/t1c/t1+c/t1_gd
}

def _is_more_specific(mod_a, mod_b):
    """Return True if mod_a is more specific than mod_b (e.g. t1c > t1)."""
    specificity = {"t1c": 3, "flair": 3, "t2": 2, "t1": 1}
    return specificity.get(mod_a, 0) > specificity.get(mod_b, 0)

# Alternative naming conventions
MODALITY_ALIASES = {
    "t1": ["t1", "t1n", "t1_native"],
    "t1c": ["t1c", "t1ce", "t1_contrast"],
    "t2": ["t2", "t2w", "t2_weighted"],
    "flair": ["flair", "t2f", "t2_flair"],
}


@dataclass
class ValidationResult:
    """Result of an input validation check."""

    passed: bool
    modality: str = ""
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "modality": self.modality,
            "issues": self.issues,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def validate_nifti_loadable(path: Path) -> tuple[bool, str]:
    """Verify a NIfTI file exists and can be loaded.

    Adapted from BrainLesion/BraTS file existence + nib.load checks.

    Args:
        path: Path to NIfTI file.

    Returns:
        (passed, message) tuple.
    """
    if not path.exists():
        return False, f"File not found: {path}"

    try:
        img = nib.load(str(path))
        data = img.get_fdata()
        if data.size == 0:
            return False, f"File is empty: {path}"
        return True, f"OK: shape={data.shape}"
    except Exception as e:
        return False, f"Failed to load {path}: {e}"


def validate_intensity_distribution(
    data: np.ndarray,
    modality: str,
    brain_mask: np.ndarray | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Check that intensity distribution is reasonable for MRI.

    Args:
        data: Volume data array.
        modality: Modality name (t1, t1c, t2, flair).
        brain_mask: Optional brain mask for within-brain stats.

    Returns:
        (passed, issues, warnings) tuple.
    """
    issues = []
    warnings = []

    # Check for all-zeros
    if np.count_nonzero(data) == 0:
        issues.append(f"{modality}: volume is entirely zero")
        return False, issues, warnings

    # Check for constant volume
    if data.std() < 1e-6:
        issues.append(f"{modality}: volume is constant (std={data.std():.2e})")
        return False, issues, warnings

    # Check for extreme outliers (likely corruption)
    vals = data[data > 0] if np.any(data > 0) else data.flatten()
    p99 = np.percentile(vals, 99)
    p01 = np.percentile(vals, 1)
    if p99 / (p01 + 1e-8) > 1e6:
        warnings.append(
            f"{modality}: extreme dynamic range (p01={p01:.1f}, p99={p99:.1f}, ratio={p99 / (p01 + 1e-8):.0f})"
        )

    # Check for negative values (unusual in raw MRI)
    neg_frac = (data < 0).sum() / data.size
    if neg_frac > 0.01:
        warnings.append(f"{modality}: {neg_frac * 100:.1f}% negative values — may already be normalized")

    return True, issues, warnings


def validate_shape_consistency(
    volumes: dict[str, np.ndarray],
) -> tuple[bool, list[str]]:
    """Verify all modalities have the same spatial dimensions.

    Adapted from BrainLesion/BraTS shape checks before stacking.

    Args:
        volumes: Dict mapping modality names to volume arrays.

    Returns:
        (passed, issues) tuple.
    """
    issues = []
    shapes = {name: vol.shape for name, vol in volumes.items()}

    if len(set(shapes.values())) > 1:
        shape_str = ", ".join(f"{k}={v}" for k, v in shapes.items())
        issues.append(f"Shape mismatch across modalities: {shape_str}")
        return False, issues

    return True, issues


def validate_voxel_spacing(
    nifti_paths: dict[str, Path],
    tolerance: float = 0.1,
) -> tuple[bool, list[str], dict[str, tuple[float, ...]]]:
    """Check that voxel spacing is consistent across modalities.

    Args:
        nifti_paths: Dict mapping modality names to NIfTI file paths.
        tolerance: Maximum allowed difference in mm between modalities.

    Returns:
        (passed, issues, spacings) tuple.
    """
    issues = []
    spacings = {}

    for modality, path in nifti_paths.items():
        try:
            img = nib.load(str(path))
            spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
            spacings[modality] = spacing
        except Exception as e:
            issues.append(f"{modality}: failed to read spacing: {e}")

    if len(spacings) > 1:
        ref_spacing = next(iter(spacings.values()))
        for modality, spacing in spacings.items():
            for i, (s, r) in enumerate(zip(spacing, ref_spacing)):
                if abs(s - r) > tolerance:
                    axes = ["X", "Y", "Z"]
                    issues.append(
                        f"{modality}: {axes[i]}-spacing={s:.3f}mm differs from "
                        f"reference={r:.3f}mm (tolerance={tolerance}mm)"
                    )

    return len(issues) == 0, issues, spacings


def find_modality_file(
    subject_dir: Path,
    modality: str,
) -> Path | None:
    """
    Find the file in `directory` that corresponds to `modality`.
    Uses precedence-ordered regex matching to avoid t1 matching t1ce, etc.
    """
    modality = modality.lower()
    if modality not in _MODALITY_PATTERNS:
        raise ValueError(f"Unknown modality: {modality}")

    candidates = list(subject_dir.glob("*.nii.gz")) + list(subject_dir.glob("*.nii"))
    
    # First, exclude candidates that match a more-specific modality
    for other_mod, patterns in _MODALITY_PATTERNS.items():
        if other_mod == modality:
            continue
        # Skip if `other_mod` is less specific than `modality` 
        if _is_more_specific(modality, other_mod):
            continue
        candidates = [
            c for c in candidates
            if not any(re.search(p, c.name.lower()) for p in patterns)
        ]
    
    # Now match against the modality's own patterns
    for cand in candidates:
        if any(re.search(p, cand.name.lower()) for p in _MODALITY_PATTERNS[modality]):
            return cand
    return None


def validate_input(
    subject_dir: Path,
    required_modalities: list[str] | None = None,
    check_spacing: bool = True,
) -> dict[str, ValidationResult]:
    """Full input validation for a subject directory.

    Adapted from BrainLesion/BraTS input_sanity_check pattern.
    Validates all required modalities are present, loadable,
    have consistent shapes, and reasonable intensity distributions.

    Args:
        subject_dir: Directory containing NIfTI files for one subject.
        required_modalities: List of required modality names.
        check_spacing: Whether to check voxel spacing consistency.

    Returns:
        Dict mapping modality names to ValidationResult.
    """
    if required_modalities is None:
        required_modalities = REQUIRED_MODALITIES

    results = {}

    # 1. Find and validate each modality
    found_paths = {}
    for modality in required_modalities:
        result = ValidationResult(passed=True, modality=modality)

        path = find_modality_file(subject_dir, modality)
        if path is None:
            result.passed = False
            result.issues.append(f"Required modality '{modality}' not found in {subject_dir}")
            results[modality] = result
            continue

        found_paths[modality] = path

        # Validate loadable
        ok, msg = validate_nifti_loadable(path)
        if not ok:
            result.passed = False
            result.issues.append(msg)
            results[modality] = result
            continue

        # Load data for intensity checks
        try:
            data = nib.load(str(path)).get_fdata()
            result.stats["shape"] = list(data.shape)
            result.stats["dtype"] = str(data.dtype)
            result.stats["path"] = str(path)

            ok, issues, warnings = validate_intensity_distribution(data, modality)
            if not ok:
                result.passed = False
                result.issues.extend(issues)
            result.warnings.extend(warnings)

        except Exception as e:
            result.passed = False
            result.issues.append(f"Failed to process {path}: {e}")

        results[modality] = result

    # 2. Shape consistency across modalities
    volumes = {}
    for modality, path in found_paths.items():
        try:
            volumes[modality] = nib.load(str(path)).get_fdata()
        except Exception:
            pass

    if len(volumes) > 1:
        ok, issues = validate_shape_consistency(volumes)
        if not ok:
            for modality in results:
                if modality in volumes:
                    results[modality].passed = False
                    results[modality].issues.extend(issues)

    # 3. Voxel spacing consistency
    if check_spacing and len(found_paths) > 1:
        ok, issues, spacings = validate_voxel_spacing(found_paths)
        if not ok:
            for modality in results:
                if modality in spacings:
                    results[modality].stats["voxel_spacing_mm"] = list(spacings[modality])
                    results[modality].warnings.extend(issues)

    return results


def validate_input_tensor(
    tensor: np.ndarray,
    expected_channels: int = 4,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[bool, list[str]]:
    """Validate a preprocessed input tensor before model inference.

    Args:
        tensor: Input tensor (C, D, H, W) or (1, C, D, H, W).
        expected_channels: Expected number of channels.
        expected_shape: Optional expected spatial shape (D, H, W).

    Returns:
        (passed, issues) tuple.
    """
    issues = []

    # Remove batch dim if present
    if tensor.ndim == 5:
        tensor = tensor[0]

    if tensor.ndim != 4:
        issues.append(f"Expected 4D tensor (C, D, H, W), got {tensor.ndim}D")
        return False, issues

    c, d, h, w = tensor.shape
    if c != expected_channels:
        issues.append(f"Expected {expected_channels} channels, got {c}")

    if expected_shape is not None:
        if (d, h, w) != expected_shape:
            issues.append(f"Expected spatial shape {expected_shape}, got ({d}, {h}, {w})")

    # Check for NaN/Inf
    if np.any(np.isnan(tensor)):
        issues.append("Tensor contains NaN values")
    if np.any(np.isinf(tensor)):
        issues.append("Tensor contains Inf values")

    # Check for all-zero channels
    for ch in range(c):
        if np.count_nonzero(tensor[ch]) == 0:
            issues.append(f"Channel {ch} is entirely zero")

    return len(issues) == 0, issues
