#!/usr/bin/env python3
"""
tests/verify_windsurf_changes.py
=================================
Quick smoke-test for the two critical Windsurf changes:

  1. brainmask.py has HD-BET integration
  2. preprocessing.py wires ref_nifti_path through

Usage:
    python3 tests/verify_windsurf_changes.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OK = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"

passed = 0
failed = 0


def check(desc: str, cond: bool):
    global passed, failed
    if cond:
        print(f"  {OK} {desc}")
        passed += 1
    else:
        print(f"  {FAIL} {desc}")
        failed += 1


bm = (PROJECT_ROOT / "pybrain/core/brainmask.py").read_text()
pp = (PROJECT_ROOT / "pybrain/core/preprocessing.py").read_text()

print("\n🔍 Verifying Windsurf changes\n")

print("── brainmask.py ──────────────────────────")
check("_hdbet_available() present", "_hdbet_available" in bm)
check("_strip_with_hdbet() present", "_strip_with_hdbet" in bm)
check("robust_brain_mask has ref_nifti_path", "ref_nifti_path" in bm)
check("robust_brain_mask has work_dir", "work_dir" in bm)
check("_is_already_skull_stripped() present", "_is_already_skull_stripped" in bm)
check("_morphological_mask() present", "_morphological_mask" in bm)
check("_mask_from_nonzero() present", "_mask_from_nonzero" in bm)
check("Volume guard 800-1700 cc", "800" in bm and "1700" in bm)

print("\n── preprocessing.py ──────────────────────")
check("ref_nifti_path wired to brainmask", "ref_nifti_path" in pp)
check("ref_seq in compute_and_apply_brain_mask", "ref_seq" in pp)
check("_count_slices() present", "_count_slices" in pp)
check("hdbet_work temp directory", "hdbet_work" in pp)
check("Brain mask volume warning (900-1700)", "900" in pp and "1700" in pp)

print("\n── import check ──────────────────────────")
try:
    from pybrain.core.brainmask import robust_brain_mask, _hdbet_available
    import inspect

    sig = inspect.signature(robust_brain_mask)
    check("robust_brain_mask importable", True)
    check("ref_nifti_path in signature", "ref_nifti_path" in sig.parameters)
    check("work_dir in signature", "work_dir" in sig.parameters)
except Exception as e:
    check(f"Import failed: {e}", False)

try:
    from pybrain.core.preprocessing import compute_and_apply_brain_mask, preprocess_mri
    import inspect

    sig = inspect.signature(compute_and_apply_brain_mask)
    check("compute_and_apply_brain_mask importable", True)
    check("ref_nifti_path in signature", "ref_nifti_path" in sig.parameters)
except Exception as e:
    check(f"Import failed: {e}", False)

print(f"\n  Results: {passed} passed  |  {failed} failed\n")
sys.exit(0 if failed == 0 else 1)
