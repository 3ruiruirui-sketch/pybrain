#!/usr/bin/env python3
"""
PY-BRAIN v2 — Complete Pipeline Code Audit
============================================
Verifies every module, function signature, safety guard, and integration
point across the entire pipeline.

Usage:
    python3 tests/audit_pipeline.py
    python3 tests/audit_pipeline.py --verbose
    python3 tests/audit_pipeline.py --fix-report   # saves audit_report.json
"""

import sys
import json
import inspect
import argparse
import importlib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []
verbose = False


def check(section: str, description: str, condition: bool, detail: str = "", warn_only: bool = False):
    if condition:
        status = PASS
    elif warn_only:
        status = WARN
    else:
        status = FAIL
    results.append(
        {
            "section": section,
            "description": description,
            "status": status,
            "detail": detail if not condition else "",
        }
    )
    line = f"  {status} {description}"
    if not condition and detail:
        line += f"\n       → {detail}"
    if verbose or not condition:
        print(line)
    elif condition:
        print(line)


def read_file(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("  PY-BRAIN v2 — Complete Pipeline Code Audit")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("═" * 70)


# ─── 1. PROJECT STRUCTURE ─────────────────────────────────────────────────────
print("\n📁 1. Project Structure")

required_files = [
    "cli.py",
    "pybrain/__init__.py",
    "pybrain/pipeline.py",
    "pybrain/core/__init__.py",
    "pybrain/core/brainmask.py",
    "pybrain/core/preprocessing.py",
    "pybrain/core/segmentation.py",
    "pybrain/core/labels.py",
    "pybrain/core/normalization.py",
    "pybrain/core/validation.py",
    "pybrain/models/__init__.py",
    "pybrain/models/segresnet.py",
    "pybrain/models/ensemble.py",
    "pybrain/models/mc_dropout.py",
    "pybrain/analysis/__init__.py",
    "pybrain/analysis/location.py",
    "pybrain/analysis/morphology.py",
    "pybrain/analysis/radiomics.py",
    "pybrain/io/__init__.py",
    "pybrain/io/nifti_io.py",
    "pybrain/io/logging_utils.py",
    "pybrain/report/__init__.py",
    "pybrain/report/pdf_report.py",
    "config/defaults.yaml",
    "tests/verify_windsurf_changes.py",
]

for f in required_files:
    path = PROJECT_ROOT / f
    check("structure", f"File exists: {f}", file_exists(path), f"Missing: {path}")


# ─── 2. DEPENDENCIES ──────────────────────────────────────────────────────────
print("\n📦 2. Dependencies")

import subprocess as _subprocess


def _import_with_timeout(pkg, timeout=30):
    """Try importing a package with a subprocess timeout to avoid hangs."""
    try:
        r = _subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except _subprocess.TimeoutExpired:
        return False


required_packages = [
    ("torch", "PyTorch"),
    ("monai", "MONAI"),
    ("nibabel", "NiBabel"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("SimpleITK", "SimpleITK"),
    ("yaml", "PyYAML"),
    ("HD_BET", "HD-BET skull stripper"),
    ("pydicom", "pyDICOM"),
    ("reportlab", "ReportLab PDF"),
    ("matplotlib", "Matplotlib"),
]

optional_packages = [
    ("radiomics", "PyRadiomics (optional)"),
    ("nnunet", "nnUNet (optional)"),
]
for pkg, name in required_packages:
    if _import_with_timeout(pkg):
        check("deps", f"{name} installed", True)
    else:
        check("deps", f"{name} installed", False, f"pip install {pkg.lower()}")

for pkg, name in optional_packages:
    if _import_with_timeout(pkg):
        check("deps", f"{name} installed", True)
    else:
        check("deps", f"{name} installed", True, warn_only=True)

# ─── 3. MODELS ────────────────────────────────────────────────────────────────
print("\n🤖 3. Model Weights")

model_files = [
    "models/brats_bundle/brats_mri_segmentation/models/model.pt",
]

optional_models = [
    "models/brainIAC",
    "models/brats_bundle/brats_mri_segmentation/models/fold1_swinunetr.pth",
]

for f in model_files:
    path = PROJECT_ROOT / f
    check(
        "models",
        f"Model weights: {Path(f).name}",
        path.exists() and path.stat().st_size > 1_000_000,
        f"Missing or too small: {path}",
    )

for f in optional_models:
    path = PROJECT_ROOT / f
    check("models", f"Optional model: {Path(f).name}", path.exists(), f"Not found (optional): {path}", warn_only=True)


# ─── 4. CONFIG ────────────────────────────────────────────────────────────────
print("\n⚙️  4. Configuration")

cfg = read_file(PROJECT_ROOT / "config/defaults.yaml")

check("config", "defaults.yaml not empty", len(cfg) > 100, "config/defaults.yaml is empty or missing")
check("config", "WT threshold = 0.40", "wt: 0.40" in cfg, "CRITICAL: wt threshold wrong — must be 0.40")
check("config", "TC threshold = 0.35", "tc: 0.35" in cfg, "CRITICAL: tc threshold wrong — must be 0.35")
check("config", "ET threshold = 0.35", "et: 0.35" in cfg, "CRITICAL: et threshold wrong — must be 0.35")
check(
    "config",
    "non_gbm_ct_calc_threshold >= 50cc",
    "non_gbm_ct_calc_threshold_cc: 50" in cfg or "non_gbm_ct_calc_threshold_cc: 200" in cfg,
    "CT calcification threshold too low",
)
check("config", "device: auto", "device:" in cfg, "Device not configured")
check(
    "config",
    "No external threshold file references",
    "optimal_thresholds.json" not in cfg and "platt_coefficients.json" not in cfg,
    "CRITICAL: external threshold file reference found — regression risk",
)


# ─── 5. brainmask.py ──────────────────────────────────────────────────────────
print("\n🧠 5. Brain Masking (brainmask.py)")

bm = read_file(PROJECT_ROOT / "pybrain/core/brainmask.py")

check("brainmask", "HD-BET availability check", "_hdbet_available" in bm, "Missing _hdbet_available()")
check("brainmask", "HD-BET skull stripping function", "_strip_with_hdbet" in bm, "Missing _strip_with_hdbet()")
check("brainmask", "ref_nifti_path parameter", "ref_nifti_path" in bm, "robust_brain_mask missing ref_nifti_path")
check("brainmask", "work_dir parameter", "work_dir" in bm, "robust_brain_mask missing work_dir")
check("brainmask", "BraTS pre-processed detection", "_is_already_skull_stripped" in bm, "BraTS detection missing")
check(
    "brainmask",
    "Robust BraTS detection (uses np.std)",
    "np.std" in bm,
    "BraTS detection not robust — should check std across sequences",
)
check("brainmask", "Morphological fallback", "_morphological_mask" in bm, "Morphological fallback missing")
check("brainmask", "Nonzero union fallback", "_mask_from_nonzero" in bm, "Nonzero fallback missing")
check("brainmask", "Volume guard (800-1700cc)", "800" in bm and "1700" in bm, "Missing volume sanity guard")
check(
    "brainmask",
    "No hardcoded Platt coefficients",
    "platt" not in bm.lower(),
    "Platt calibration found in brainmask — wrong file",
)

try:
    from pybrain.core.brainmask import robust_brain_mask, _hdbet_available

    sig = inspect.signature(robust_brain_mask)
    check("brainmask", "robust_brain_mask importable", True)
    check(
        "brainmask",
        "ref_nifti_path in signature",
        "ref_nifti_path" in sig.parameters,
        "Parameter missing from live function",
    )
    check("brainmask", "work_dir in signature", "work_dir" in sig.parameters, "Parameter missing from live function")
    check(
        "brainmask",
        "HD-BET available at runtime",
        _hdbet_available(),
        "HD-BET not importable at runtime",
        warn_only=not _hdbet_available(),
    )
except Exception as e:
    check("brainmask", "brainmask module importable", False, str(e))


# ─── 6. preprocessing.py ──────────────────────────────────────────────────────
print("\n⚙️  6. Preprocessing (preprocessing.py)")

pp = read_file(PROJECT_ROOT / "pybrain/core/preprocessing.py")

check(
    "preproc",
    "DICOM auto-detection (_is_dicom_dir)",
    "_is_dicom_dir" in pp,
    "DICOM detection missing — users must convert manually",
)
check("preproc", "dcm2niix auto-conversion", "_convert_dicom_dir_to_nifti" in pp, "DICOM conversion missing")
check(
    "preproc",
    "ref_nifti_path wired to brainmask",
    "ref_nifti_path" in pp,
    "ref_nifti_path not passed to robust_brain_mask",
)
check(
    "preproc",
    "ref_seq parameter in compute_and_apply_brain_mask",
    "ref_seq" in pp,
    "ref_seq missing — wrong sequence used for brain mask",
)
check(
    "preproc",
    "Slice-count reference selection (_count_slices)",
    "_count_slices" in pp,
    "No 3D/2D sequence detection — wrong reference for hospital DICOM",
)
check("preproc", "hdbet_work temp directory", "hdbet_work" in pp, "HD-BET work directory not configured")
check(
    "preproc",
    "Brain mask volume warning",
    "900" in pp and "1700" in pp,
    "No brain volume sanity check in preprocessing",
)
check("preproc", "LPS reorientation", "LPS" in pp, "LPS orientation not applied")
check(
    "preproc",
    "1mm isotropic resampling",
    "1.0, 1.0, 1.0" in pp or "[1.0, 1.0, 1.0]" in pp,
    "1mm isotropic resampling not configured",
)
check(
    "preproc",
    "Registration fallback (uses resampled if registration fails)",
    "registration failed" in pp.lower() or "fallback" in pp.lower(),
    "No registration fallback — crash if SimpleITK registration fails",
)
check(
    "preproc",
    "Intermediate file cleanup",
    "_raw_resampled" in pp or "unlink" in pp,
    "Intermediate files not cleaned up — disk space waste",
)

try:
    from pybrain.core.preprocessing import preprocess_mri

    sig = inspect.signature(preprocess_mri)
    check("preproc", "preprocess_mri importable", True)
    check(
        "preproc",
        "preprocess_mri has assignments param",
        "assignments" in sig.parameters,
        "Missing assignments parameter",
    )
    check(
        "preproc", "preprocess_mri has output_dir param", "output_dir" in sig.parameters, "Missing output_dir parameter"
    )
except Exception as e:
    check("preproc", "preprocess_mri importable", False, str(e))


# ─── 7. segmentation.py ───────────────────────────────────────────────────────
print("\n🔬 7. Segmentation (segmentation.py)")

seg = read_file(PROJECT_ROOT / "pybrain/core/segmentation.py")

check(
    "segmentation",
    "SegmentationConfig dataclass",
    "SegmentationConfig" in seg,
    "SegmentationConfig missing — config not encapsulated",
)
check(
    "segmentation",
    "SegmentationResult dataclass",
    "SegmentationResult" in seg,
    "SegmentationResult missing — results not structured",
)
check(
    "segmentation",
    "wt_threshold = 0.40 default",
    "wt_threshold: float = 0.40" in seg or "0.40" in seg,
    "WT threshold default wrong or missing",
)
check(
    "segmentation",
    "No external threshold file loading",
    "optimal_thresholds.json" not in seg,
    "CRITICAL: external threshold file — regression risk",
)
check(
    "segmentation",
    "No Platt calibration loading",
    "platt_coefficients.json" not in seg,
    "CRITICAL: Platt calibration — caused 0cc regression in v1",
)
check(
    "segmentation", "model_activated flag", "model_activated" in seg, "No model activation check — silent 0cc failure"
)
check("segmentation", "non_gbm_suspected flag", "non_gbm_suspected" in seg, "No non-GBM detection")
check("segmentation", "CT boost conditional (NMI validated)", "ct_boost" in seg.lower(), "CT boost not implemented")
check("segmentation", "MC-Dropout support", "mc_dropout" in seg.lower(), "MC-Dropout not implemented", warn_only=True)
check(
    "segmentation",
    "Memory cleanup (gc.collect)",
    "gc.collect" in seg,
    "No GPU memory cleanup — may OOM on large volumes",
    warn_only=True,
)
check(
    "segmentation",
    "Volume computation per sub-region",
    "wt_cc" in seg and "et_cc" in seg and "nc_cc" in seg,
    "Sub-region volumes not computed",
)

try:
    from pybrain.core.segmentation import segment, SegmentationConfig, SegmentationResult

    check("segmentation", "segment() importable", True)
    sig = inspect.signature(segment)
    check("segmentation", "segment() has volumes param", "volumes" in sig.parameters, "Missing volumes")
    check("segmentation", "segment() has brain_mask param", "brain_mask" in sig.parameters, "Missing brain_mask")
    check("segmentation", "segment() has config param", "config" in sig.parameters, "Missing config")
except Exception as e:
    check("segmentation", "segmentation module importable", False, str(e))


# ─── 8. pipeline.py ───────────────────────────────────────────────────────────
print("\n🔀 8. Pipeline Orchestrator (pipeline.py)")

pl = read_file(PROJECT_ROOT / "pybrain/pipeline.py")

check(
    "pipeline",
    "preproc_dir defined at top of run()",
    "preproc_dir" in pl,
    "CRITICAL: preproc_dir not defined — location/morphology/radiomics fail",
)
check(
    "pipeline",
    "skip_preprocessing flag supported",
    "skip_preprocessing" in pl,
    "Cannot use with pre-processed NIfTI — BraTS testing broken",
)
check(
    "pipeline", "GT validation support (gt_path)", "gt_path" in pl, "No ground truth validation — cannot measure Dice"
)
check(
    "pipeline",
    "load_config() with fallback defaults",
    "load_config" in pl,
    "No config loading — uses hardcoded values only",
)
check(
    "pipeline",
    "Device auto-detection (MPS > CUDA > CPU)",
    "mps" in pl.lower() and "cuda" in pl.lower(),
    "No device auto-detection",
)
check(
    "pipeline",
    "Stage toggles (run_location etc)",
    "run_location" in pl and "run_report" in pl,
    "No stage toggles — cannot skip stages",
)
check(
    "pipeline",
    "No session.json architecture",
    "session.json" not in pl,
    "CRITICAL: session.json found — v1 regression risk",
)
check(
    "pipeline",
    "No external threshold file loading",
    "optimal_thresholds.json" not in pl,
    "CRITICAL: external threshold override — regression risk",
)
check("pipeline", "segmentation_quality.json saved", "segmentation_quality.json" in pl, "Quality report not saved")
check("pipeline", "model_activated early exit", "model_activated" in pl, "No early exit on model non-activation")
check("pipeline", "elapsed time logged", "elapsed" in pl, "No timing — hard to profile performance", warn_only=True)

try:
    from pybrain.pipeline import run, load_config

    check("pipeline", "run() importable", True)
    sig = inspect.signature(run)
    for param in ["assignments", "output_dir", "skip_preprocessing", "gt_path", "run_report"]:
        check("pipeline", f"run() has {param} param", param in sig.parameters, f"Missing parameter: {param}")
except Exception as e:
    check("pipeline", "pipeline module importable", False, str(e))


# ─── 9. cli.py ────────────────────────────────────────────────────────────────
print("\n💻 9. CLI (cli.py)")

cli = read_file(PROJECT_ROOT / "cli.py")

check(
    "cli",
    "--t1 / --t1c / --t2 / --flair arguments",
    "--t1" in cli and "--flair" in cli,
    "MRI sequence arguments missing",
)
check("cli", "--ct argument", "--ct" in cli, "CT argument missing")
check("cli", "--output argument", "--output" in cli, "Output directory argument missing")
check(
    "cli",
    "--skip-preprocessing flag",
    "skip-preprocessing" in cli or "skip_preprocessing" in cli,
    "Cannot use with pre-processed data",
)
check("cli", "--gt flag for validation", "--gt" in cli, "No ground truth validation flag")
check("cli", "--no-report flag", "no-report" in cli or "no_report" in cli, "Cannot skip report generation")
check(
    "cli",
    "DICOM directory support mentioned in help",
    "DICOM" in cli or "dicom" in cli,
    "Help text does not mention DICOM support",
)
check(
    "cli",
    "Patient info args (--name --age --sex)",
    "--name" in cli and "--age" in cli and "--sex" in cli,
    "Patient info arguments missing",
)
check(
    "cli",
    "Exit code 2 for model non-activation",
    "sys.exit(2)" in cli or "exit(2)" in cli,
    "No distinct exit code for model non-activation warning",
    warn_only=True,
)
check(
    "cli",
    "No hardcoded patient values",
    "SOARES" not in cli and "630822" not in cli,
    "CRITICAL: hardcoded patient data in CLI — GDPR violation",
)


# ─── 10. Analysis modules ─────────────────────────────────────────────────────
print("\n📊 10. Analysis Modules")

for module in ["location", "morphology", "radiomics"]:
    content = read_file(PROJECT_ROOT / f"pybrain/analysis/{module}.py")
    check(
        "analysis",
        f"{module}.py: _to_serializable() present",
        "_to_serializable" in content,
        f"JSON serialization bug — bool_ not serializable",
    )
    check(
        "analysis",
        f"{module}.py: no session.json loading",
        "session.json" not in content,
        f"CRITICAL: session.json in {module} — v1 regression",
    )
    check(
        "analysis",
        f"{module}.py: no module-level execution",
        'if __name__ == "__main__"' not in content or content.count("if __name__") <= 1,
        f"{module}.py has module-level code — not testable",
    )


# ─── 11. Safety guards ────────────────────────────────────────────────────────
print("\n🛡️  11. Safety Guards")

all_code = " ".join(
    [
        read_file(PROJECT_ROOT / "pybrain/core/segmentation.py"),
        read_file(PROJECT_ROOT / "pybrain/pipeline.py"),
        read_file(PROJECT_ROOT / "pybrain/core/brainmask.py"),
    ]
)

check(
    "safety",
    "No hardcoded patient name",
    "CELESTE" not in all_code and "SOARES" not in all_code,
    "CRITICAL: hardcoded patient name — GDPR violation",
)
check(
    "safety",
    "No hardcoded reference volume (32cc)",
    '"32 cc"' not in all_code and '"32cc"' not in all_code,
    "CRITICAL: hardcoded reference volume from v1",
)
check(
    "safety",
    "No positive Platt A coefficient risk",
    "platt_coefficients.json" not in all_code,
    "CRITICAL: Platt file loading — caused 0cc regression in v1",
)
check(
    "safety",
    "No optimal_thresholds.json loading",
    "optimal_thresholds.json" not in all_code,
    "CRITICAL: external threshold override — caused 0cc regression in v1",
)
check(
    "safety",
    "Research disclaimer present",
    "RESEARCH" in all_code or "research" in all_code,
    "No research disclaimer — clinical misuse risk",
    warn_only=True,
)


# ─── 12. Integration test ─────────────────────────────────────────────────────
print("\n🔗 12. Integration (import chain)")

try:
    from pybrain.pipeline import run
    from pybrain.core.brainmask import robust_brain_mask
    from pybrain.core.preprocessing import preprocess_mri
    from pybrain.core.segmentation import segment, SegmentationConfig
    from pybrain.core.labels import canonical_labels

    check("integration", "Full import chain works", True)
except Exception as e:
    check("integration", "Full import chain works", False, str(e))

try:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    check("integration", f"Compute device available ({device})", True)
except Exception as e:
    check("integration", "Compute device check", False, str(e))

if _import_with_timeout("pybrain.models.segresnet", timeout=60):
    check("integration", "SegResNet loader importable", True)
else:
    check("integration", "SegResNet loader importable", False, "Could not import (timeout or missing)", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  AUDIT SUMMARY")
print("═" * 70)

passed = sum(1 for r in results if r["status"] == PASS)
failed = sum(1 for r in results if r["status"] == FAIL)
warned = sum(1 for r in results if r["status"] == WARN)
total = len(results)

# Group failures by section
if failed > 0:
    print(f"\n  ❌ FAILED CHECKS ({failed}):")
    sections_seen = {}
    for r in results:
        if r["status"] == FAIL:
            sec = r["section"]
            if sec not in sections_seen:
                sections_seen[sec] = []
            sections_seen[sec].append(r)
    for sec, items in sections_seen.items():
        print(f"\n    [{sec.upper()}]")
        for item in items:
            print(f"    • {item['description']}")
            if item["detail"]:
                print(f"      → {item['detail']}")

if warned > 0:
    print(f"\n  ⚠️  WARNINGS ({warned}):")
    for r in results:
        if r["status"] == WARN:
            print(f"    • [{r['section']}] {r['description']}")

print(f"\n  Results: {passed} passed  |  {failed} failed  |  {warned} warnings  |  {total} total")
print()

if failed == 0:
    print("  ✅ PIPELINE AUDIT PASSED — All critical checks green.")
    print()
    print("  Next step — test with real patient DICOM data:")
    print()
    print("    python3 cli.py \\")
    print("        --t1    /tmp/celeste_nifti/t1_mprage_sag_p2_iso_1.0.nii.gz \\")
    print("        --t1c   /tmp/celeste_nifti/t1_se_tra_civ.nii.gz \\")
    print("        --t2    /tmp/celeste_nifti/pd+t2_tse_tra_e2.nii.gz \\")
    print("        --flair /tmp/celeste_nifti/t2_tirm_tra_dark-fluid_320.nii.gz \\")
    print("        --output results/celeste_hdbet_test \\")
    print("        --name 'MARIA CELESTE COELHO CORREIA SOARES' \\")
    print("        --age 81 --sex F --no-report")
    print()
    print("  Expected: Brain mask 1100-1400cc | Whole tumour 25-45cc")
else:
    print(f"  ❌ AUDIT FAILED — {failed} critical issue(s) must be fixed.")
    print("  Fix the items above and re-run: python3 tests/audit_pipeline.py")

print("═" * 70 + "\n")

# Save JSON report if requested
if "--fix-report" in sys.argv:
    report_path = PROJECT_ROOT / "tests/audit_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "passed": passed,
                "failed": failed,
                "warned": warned,
                "total": total,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"  📄 Full report saved: {report_path}\n")

sys.exit(0 if failed == 0 else 1)
