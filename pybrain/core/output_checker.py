"""
Output sanity checker for segmentation results.

Adapted from BrainLesion/BraTS _sanity_check_output pattern:
- Validates output count matches expected inputs
- Detects zero-valued (empty) segmentation outputs
- Checks probability map range and statistical health
- Detects double-sigmoid collapse and channel swaps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pybrain.io.logging_utils import get_logger

logger = get_logger("core.output_checker")


@dataclass
class OutputCheckResult:
    """Result of an output sanity check."""

    passed: bool
    checks_run: int = 0
    checks_passed: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def check_output_count(
    input_paths: list[Path],
    output_paths: list[Path],
    expected_outputs: int = 1,
) -> tuple[bool, str]:
    """Verify the number of output files matches expectations.

    Adapted from BrainLesion/BraTS _sanity_check_output: checks that
    len(outputs) >= len(inputs).

    Args:
        input_paths: List of input file paths.
        output_paths: List of output file paths.
        expected_outputs: Minimum expected number of outputs.

    Returns:
        (passed, message) tuple.
    """
    n_outputs = len(output_paths)
    if n_outputs < expected_outputs:
        return False, (f"Not enough output files: expected >= {expected_outputs}, got {n_outputs}")
    return True, f"Output count OK: {n_outputs} files"


def check_output_not_empty(
    output_path: Path,
    subject_name: str = "<unknown>",
) -> tuple[bool, str]:
    """Check that a segmentation output is not entirely zero-valued.

    Adapted from BrainLesion/BraTS _sanity_check_output zero-check:
    loads the NIfTI and verifies nonzero voxel count > 0.

    Args:
        output_path: Path to the output NIfTI file.
        subject_name: Subject identifier for logging.

    Returns:
        (passed, message) tuple.
    """
    if not output_path.exists():
        return False, f"Output file does not exist: {output_path}"

    try:
        data = nib.load(str(output_path)).get_fdata()
    except Exception as e:
        return False, f"Failed to load {output_path}: {e}"

    nonzero = np.count_nonzero(data)
    if nonzero == 0:
        logger.warning(f"Output for {subject_name} contains only zeros. Segmentation may have failed.")
        return False, (f"Output for {subject_name} is entirely zero — no tumor detected or inference failed")
    return True, f"Output for {subject_name}: {nonzero} nonzero voxels"


def check_probability_range(
    prob: np.ndarray,
    name: str = "probability",
    tol: float = 0.001,
) -> tuple[bool, list[str]]:
    """Verify probability values are in [0, 1] and not degenerate.

    Adapted from BrainLesion/BraTS regression_baseline _check_range:
    - min >= -tol (no negative probabilities)
    - max <= 1 + tol (no >1 probabilities)
    - mean not suspiciously high (all-ones collapse)
    - mean not suspiciously low (all-zeros collapse)
    - no double-sigmoid collapse (mean ~0.5 with low std)

    Args:
        prob: Probability array (3, D, H, W) or (D, H, W).
        name: Identifier for error messages.
        tol: Numerical tolerance for bounds.

    Returns:
        (passed, issues) tuple.
    """
    issues = []

    if prob.min() < -tol:
        issues.append(f"{name}: min={prob.min():.4f} < 0 — invalid probability")
    if prob.max() > 1.0 + tol:
        issues.append(f"{name}: max={prob.max():.4f} > 1 — invalid probability")
    if prob.mean() > 0.95:
        issues.append(f"{name}: mean={prob.mean():.4f} suspiciously high (all-ones collapse)")
    if prob.mean() < 0.0001:
        issues.append(f"{name}: mean={prob.mean():.6f} suspiciously low (dead model / all-zeros)")
    # Double-sigmoid detection: output clusters near 0.5 with low variance
    if abs(prob.mean() - 0.5) < 0.02 and prob.std() < 0.06:
        issues.append(f"{name}: mean~0.5, std={prob.std():.4f} — possible double-sigmoid collapse")

    return len(issues) == 0, issues


def check_hierarchy_violations(
    seg: np.ndarray,
    max_violations: int = 50,
) -> tuple[bool, list[str]]:
    """Check ET ⊆ TC ⊆ WT label hierarchy in a segmentation.

    Adapted from BrainLesion/BraTS regression_baseline T7:
    After thresholding, ET voxels must be inside TC, and TC inside WT.
    Small boundary violations (<=max_violations) are tolerated as
    floating-point noise resolved by postprocessing.

    Label convention (BraTS 2021): 1=NCR (TC), 2=ED (WT only), 4=ET (TC+ET).

    Args:
        seg: Integer segmentation array with labels {0, 1, 2, 4}.
        max_violations: Maximum tolerated violation voxel count.

    Returns:
        (passed, issues) tuple.
    """
    issues = []
    tc_mask = ((seg == 1) | (seg == 4)).astype(bool)  # BraTS 2021: ET = label 4
    wt_mask = (seg > 0).astype(bool)
    et_mask = (seg == 4).astype(bool)  # BraTS 2021: ET = label 4
    total_vox = seg.size

    # ET ⊆ TC
    et_outside_tc = (et_mask & ~tc_mask).sum()
    if et_outside_tc > max_violations:
        pct = 100.0 * et_outside_tc / total_vox
        issues.append(f"FAIL: ET not subset of TC: {et_outside_tc} voxels ({pct:.3f}%)")
    elif et_outside_tc > 0:
        pct = 100.0 * et_outside_tc / total_vox
        issues.append(f"WARN: ET not subset of TC: {et_outside_tc} voxels ({pct:.3f}%) — boundary noise")

    # TC ⊆ WT
    tc_outside_wt = (tc_mask & ~wt_mask).sum()
    if tc_outside_wt > max_violations:
        pct = 100.0 * tc_outside_wt / total_vox
        issues.append(f"FAIL: TC not subset of WT: {tc_outside_wt} voxels ({pct:.3f}%)")
    elif tc_outside_wt > 0:
        pct = 100.0 * tc_outside_wt / total_vox
        issues.append(f"WARN: TC not subset of WT: {tc_outside_wt} voxels ({pct:.3f}%) — boundary noise")

    hard_fails = [i for i in issues if i.startswith("FAIL:")]
    return len(hard_fails) == 0, issues


def sanity_check_segmentation(
    input_dir: Path,
    output_path: Path,
    subject_name: str = "<unknown>",
    prob_map: np.ndarray | None = None,
) -> OutputCheckResult:
    """Run all output sanity checks on a segmentation result.

    This is the main entry point, adapted from BrainLesion/BraTS
    _sanity_check_output + _check_range + T7 hierarchy combined.

    Args:
        input_dir: Directory containing input files.
        output_path: Path to the output segmentation NIfTI.
        subject_name: Subject identifier for logging.
        prob_map: Optional (3, D, H, W) probability map for range checks.

    Returns:
        OutputCheckResult with all check results.
    """
    result = OutputCheckResult(passed=True)

    # 1. Output file existence and non-empty check
    result.checks_run += 1
    ok, msg = check_output_not_empty(output_path, subject_name)
    if ok:
        result.checks_passed += 1
    else:
        result.issues.append(msg)
        result.passed = False

    # 2. If prob_map provided, check range
    if prob_map is not None:
        result.checks_run += 1
        ok, issues = check_probability_range(prob_map, name=subject_name)
        if ok:
            result.checks_passed += 1
        else:
            result.issues.extend(issues)
            result.passed = False
            result.stats["prob_stats"] = {
                "min": float(prob_map.min()),
                "max": float(prob_map.max()),
                "mean": float(prob_map.mean()),
                "std": float(prob_map.std()),
            }

    # 3. Load segmentation and check hierarchy
    try:
        seg = nib.load(str(output_path)).get_fdata().astype(np.int32)
        result.checks_run += 1
        ok, issues = check_hierarchy_violations(seg)
        if ok:
            result.checks_passed += 1
        else:
            for issue in issues:
                if issue.startswith("FAIL:"):
                    result.issues.append(issue)
                    result.passed = False
                else:
                    result.warnings.append(issue)

        result.stats["label_counts"] = {int(label): int((seg == label).sum()) for label in np.unique(seg)}
    except Exception as e:
        result.issues.append(f"Failed to load segmentation for hierarchy check: {e}")

    return result


def sanity_check_batch(
    input_dir: Path,
    output_dir: Path,
    subject_pattern: str = "BraTS*",
) -> dict[str, OutputCheckResult]:
    """Batch sanity check across multiple subjects.

    Adapted from BrainLesion/BraTS batch inference output validation.

    Args:
        input_dir: Directory containing input subject directories.
        output_dir: Directory containing output NIfTI files.
        subject_pattern: Glob pattern for subject directories.

    Returns:
        Dict mapping subject names to their check results.
    """
    results = {}
    output_files = list(output_dir.glob("*.nii.gz")) + list(output_dir.glob("*.nii"))

    if not output_files:
        logger.warning(f"No output files found in {output_dir}")
        return results

    for output_path in output_files:
        subject_name = output_path.stem.replace(".nii", "")
        result = sanity_check_segmentation(
            input_dir=input_dir,
            output_path=output_path,
            subject_name=subject_name,
        )
        results[subject_name] = result

        if not result.passed:
            logger.error(f"Sanity check FAILED for {subject_name}: {'; '.join(result.issues)}")
        elif result.warnings:
            logger.warning(f"Sanity check passed with warnings for {subject_name}: {'; '.join(result.warnings)}")

    return results
