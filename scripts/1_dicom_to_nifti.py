#!/usr/bin/env python3
"""
Stage 1 — DICOM → NIfTI Conversion + BraTS Preparation
=========================================================
Reads session from run_pipeline.py (PYBRAIN_SESSION env var).
Uses sequence assignments made interactively in the wizard.

Requirements:
  brew install dcm2niix
  pip install nibabel SimpleITK
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Load session (set by run_pipeline.py, or auto-find latest) ────────────
def load_session() -> dict:
    env_path = os.environ.get("PYBRAIN_SESSION")
    if env_path and Path(env_path).exists():
        with open(env_path) as f:
            return json.load(f)
    # fallback: most recent session by timestamp extraction (not mtime)
    import re

    results_dir = PROJECT_ROOT / "results"
    all_json = list(results_dir.glob("*/session.json")) if results_dir.exists() else []

    def _ts(p: Path):
        m = re.search(r"(\d{8}_\d{6})", p.parent.name)
        return m.group(1) if m else ""

    candidates = sorted([p for p in all_json if _ts(p)], key=_ts)
    if candidates:
        print(f"  ℹ️  Using latest session: {candidates[-1].parent.name}")
        with open(candidates[-1]) as f:
            return json.load(f)
    print("❌  No session found. Run:  python3 run_pipeline.py")
    sys.exit(1)


sess = load_session()
# DEBUG:
print(f"  DEBUG: Loaded session for {sess.get('patient', {}).get('name', '?')}")
print(f"  DEBUG: Assignments : {list(sess.get('assignments', {}).keys())}")

# ── Resolve paths from session ────────────────────────────────────────────
MONAI_DIR = Path(sess["monai_dir"])
EXTRA_DIR = Path(sess["extra_dir"])
NIFTI_DIR = Path(sess["nifti_dir"])

# Series paths and role assignments come from the wizard
# Format: {"T1": "folder_name", "T1c": "folder_name", ...}
ASSIGNMENTS = sess.get("assignments", {})
SERIES_PATHS = {k: Path(v) for k, v in sess.get("series_paths", {}).items()}

# ── Auto-fix broken series paths ─────────────────────────────────────────
# If session was created when project was in a different location
# (e.g. Downloads), series paths may point to wrong place.
# Try to find the DICOM folder automatically inside the project.
import typing


def _find_dicom_root() -> typing.Optional[Path]:
    """
    Look for DICOM folders in common locations relative to PROJECT_ROOT.
    Checks: DIRCOM/, Rm_Cranio/, tc_Cranio/, DICOM/, dicom/
    Returns the folder that contains the most series sub-folders.
    """
    candidates = [
        PROJECT_ROOT / "DIRCOM",
        PROJECT_ROOT / "Dircom",
        PROJECT_ROOT / "dircom",
        PROJECT_ROOT / "DICOM",
        PROJECT_ROOT / "dicom",
        PROJECT_ROOT / "Rm_Cranio",
        PROJECT_ROOT / "tc_Cranio",
        PROJECT_ROOT / "data" / "raw_dicom",
    ]
    for c in candidates:
        if c.exists() and any(True for _ in c.iterdir() if _.is_dir()):
            return c
    return None


def _fix_series_paths(series_paths: dict) -> dict:
    """
    If stored paths don't exist, try to remap them to the
    correct location by searching inside PROJECT_ROOT.
    """
    # Directories that are NEVER DICOM — skip during search
    _IGNORE_DIRS = {
        ".venv",
        "venv",
        ".env",
        "env",
        "__pycache__",
        ".git",
        ".svn",
        "node_modules",
        "models",
        "results",
        "scripts",
        "config",
        ".agents",
        ".gemini",
        "nifti",
        "monai_ready",
        "extra_sequences",
        "brats_bundle",
        "site-packages",
        "lib",
        "bin",
        "include",
    }

    # Check if any path is broken
    broken = [k for k, v in series_paths.items() if not Path(v).exists()]
    if not broken:
        return series_paths  # all good

    print(f"\n  ⚠️  {len(broken)} series path(s) from session not found.")
    print("  Searching for DICOM folders in project...")

    # Find all series folders, skipping non-DICOM trees
    found_folders = {}
    for item in PROJECT_ROOT.rglob("*"):
        if not item.is_dir():
            continue
        # Skip if any parent is in _IGNORE_DIRS
        skip = False
        for part in item.relative_to(PROJECT_ROOT).parts:
            if part.lower() in _IGNORE_DIRS or part.startswith("."):
                skip = True
                break
        if skip:
            continue
        # Check if this folder contains image files
        files = [f for f in item.iterdir() if f.is_file() and not f.name.startswith(".")]
        if files:
            found_folders[item.name] = item

    # Remap broken paths by folder name
    fixed = dict(series_paths)
    remapped: int = 0
    for key, old_path in series_paths.items():
        if not Path(old_path).exists():
            folder_name = Path(old_path).name
            if folder_name in found_folders:
                fixed[key] = str(found_folders[folder_name])
                print(f"  ✅ Remapped: {folder_name}")
                print(f"              {old_path}")
                print(f"           →  {fixed[key]}")
                remapped += 1
            else:
                print(f"  ❌ Could not find: {folder_name}")

    if remapped > 0:
        print(f"\n  Fixed {remapped}/{len(broken)} broken paths")
        # Save fixed paths back to the CURRENT session (not the most recent by mtime)
        current_sess_path = Path(os.environ.get("PYBRAIN_SESSION", ""))
        if not current_sess_path.exists():
            current_sess_path = PROJECT_ROOT / "results" / Path(sess["output_dir"]).name / "session.json"
        if current_sess_path.exists():
            with open(current_sess_path) as f:
                s = json.load(f)
            s["series_paths"] = fixed
            with open(current_sess_path, "w") as f:
                json.dump(s, f, indent=2)
            print(f"  Session updated: {current_sess_path}")
        else:
            print("  ⚠️  Could not find current session file to update")

    return fixed


SERIES_PATHS = _fix_series_paths(SERIES_PATHS)

# ── Role → output filename mapping ────────────────────────────────────────
ROLE_OUTPUT = {
    # BRATS required
    "T1": ("t1.nii.gz", "BRATS"),
    "T1c": ("t1c.nii.gz", "BRATS"),
    "T2": ("t2.nii.gz", "BRATS"),
    "FLAIR": ("flair.nii.gz", "BRATS"),
    # Optional extras
    "T2star": ("t2star.nii.gz", "EXTRA"),
    "DWI": ("dwi.nii.gz", "EXTRA"),
    "ADC": ("adc.nii.gz", "EXTRA"),
    "CT": ("ct_brain.nii.gz", "EXTRA"),
    "CT_bone": ("ct_bone.nii.gz", "EXTRA"),
}

# ─────────────────────────────────────────────────────────────────────────


def banner(t):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


def check_dcm2niix() -> str:
    """Find dcm2niix — checks Homebrew Apple Silicon path first."""
    for c in ["/opt/homebrew/bin/dcm2niix", "/usr/local/bin/dcm2niix", "dcm2niix"]:
        if shutil.which(c):
            return c
    print("❌ dcm2niix not found.")
    print("  Install:  brew install dcm2niix")
    sys.exit(1)


def convert_series(dcm2niix: str, dicom_dir: Path, out_dir: Path, out_name: str) -> bool:
    """Convert one DICOM series folder → NIfTI. Returns True on success."""
    stem = out_name.replace(".nii.gz", "")
    cmd = [
        dcm2niix,
        "-o",
        str(out_dir),
        "-f",
        stem,
        "-z",
        "y",  # gzip
        "-b",
        "y",  # JSON sidecar
        "-m",
        "y",  # merge 2D slices
        "-l",
        "y",  # scale to full dynamic range (fixes intensity normalization)
        "-v",
        "0",  # quiet
        str(dicom_dir),
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    candidates = list(out_dir.glob(f"{stem}*.nii.gz"))
    # Filter out trivially small files (< 10 KB = likely corrupt/empty)
    candidates = [c for c in candidates if c.stat().st_size > 10_000]
    if not candidates:
        return False
    best = max(candidates, key=lambda p: p.stat().st_size)
    target = out_dir / out_name
    if best != target:
        best.rename(target)
    for extra in out_dir.glob(f"{stem}*.nii.gz"):
        if extra != target:
            extra.unlink(missing_ok=True)
    return True


def safe_read_sitk(path: Path, sitk):
    """Read image and flatten 4D → 3D (DWI often has extra dim)."""
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if img.GetDimension() == 4:
        ex = sitk.ExtractImageFilter()
        sz = list(img.GetSize())
        sz[3] = 0
        ex.SetSize(sz)
        ex.SetIndex([0, 0, 0, 0])
        img = ex.Execute(img)
    return sitk.Cast(img, sitk.sitkFloat32)


def register(sitk, moving_path: Path, ref, out_path: Path):
    """Rigid registration of moving → ref space."""
    moving = safe_read_sitk(moving_path, sitk)
    try:
        init_tx = sitk.CenteredTransformInitializer(
            ref, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    except RuntimeError:
        init_tx = sitk.Euler3DTransform()

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init_tx, inPlace=False)

    try:
        final_tx = reg.Execute(ref, moving)
        metric = reg.GetMetricValue()
    except RuntimeError as e:
        print(f"⚠️  registration failed ({e}), using geometry resample")
        final_tx = sitk.Transform()
        metric = float("nan")

    resampled = sitk.Resample(moving, ref, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    sitk.WriteImage(resampled, str(out_path))
    return metric


# ═════════════════════════════════════════════════════════════════════════
# STEP 1 — VALIDATE
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 1 — VALIDATING")

# Check if session has assignments — if not, we cannot proceed
if not ASSIGNMENTS:
    print("❌  No sequence assignments found in session.")
    print("    Run:  python3 run_pipeline.py  and complete the wizard.")
    sys.exit(1)

# Check if any series paths actually exist
valid_paths = {k: v for k, v in SERIES_PATHS.items() if Path(v).exists()}
if not valid_paths:
    print("❌  No DICOM series folders found.")
    print("")
    print(f"  The session has {len(SERIES_PATHS)} series but none exist at stored paths.")
    print("  This happens when the project was moved after the wizard ran.")
    print("")
    print("  Your DICOM folder should be at:")
    print(f"    {PROJECT_ROOT}/DIRCOM/")
    print("")

    # Try to find DIRCOM automatically
    dicom_root = _find_dicom_root()
    if dicom_root:
        print(f"  Found DICOM folder at: {dicom_root}")
        ans = input("  Use this folder? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            # Rescan and rebuild series paths
            found_folders = {}
            for item in dicom_root.rglob("*"):
                if not item.is_dir():
                    continue
                files = [f for f in item.iterdir() if f.is_file() and not f.name.startswith(".")]
                if files:
                    found_folders[item.name] = item

            new_series_paths = {}
            new_assignments = {}
            for role, folder_name_obj in ASSIGNMENTS.items():
                folder_name = str(folder_name_obj)
                if folder_name in found_folders:
                    new_series_paths[folder_name] = str(found_folders[folder_name])
                    new_assignments[role] = folder_name

            SERIES_PATHS.update({k: Path(v) for k, v in new_series_paths.items()})
            ASSIGNMENTS.update(new_assignments)
            print(f"  ✅ Found {len(new_assignments)} series under {dicom_root}")
    else:
        print("  Could not auto-detect DICOM folder.")
        print("  Please re-run the wizard:  python3 run_pipeline.py")
        sys.exit(1)

dcm2niix_bin = check_dcm2niix()
print(f"  dcm2niix : {dcm2niix_bin}")

MONAI_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
print(f"  BraTS    → {MONAI_DIR}")
print(f"  Extra    → {EXTRA_DIR}")

# Show what will be converted
print(f"\n  Sequences to convert ({len(ASSIGNMENTS)}):")
for role, folder in ASSIGNMENTS.items():
    src_path = SERIES_PATHS.get(folder)
    exists = "✅" if src_path and src_path.exists() else "❌ NOT FOUND"
    out_file = ROLE_OUTPUT.get(role, (f"{role.lower()}.nii.gz", "EXTRA"))[0]
    print(f"  {role:10s} ← {folder:<40} {exists}")


# ═════════════════════════════════════════════════════════════════════════
# STEP 2 — CONVERT
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 2 — CONVERTING DICOM → NIfTI")

conv_results = {}

for role, folder in ASSIGNMENTS.items():
    if role not in ROLE_OUTPUT:
        continue

    out_file, out_role = ROLE_OUTPUT[role]
    dicom_dir = SERIES_PATHS.get(folder)

    if not isinstance(dicom_dir, Path):
        continue

    dicom_dir_path: Path = dicom_dir  # type: ignore

    # Count all image files recursively
    all_files = [f for f in dicom_dir_path.rglob("*") if f.is_file() and not f.name.startswith(".")]
    if not all_files:
        print(f"  ⚠️  {role:8s} — no files in {folder}")
        conv_results[role] = "empty"
        continue

    out_dir = MONAI_DIR if out_role == "BRATS" else EXTRA_DIR
    print(f"  Converting {role:8s} ({len(all_files)} files)…", end=" ", flush=True)

    ok = convert_series(dcm2niix_bin, dicom_dir_path, out_dir, out_file)
    if ok:
        size_mb = (out_dir / out_file).stat().st_size / 1024 / 1024
        print(f"✅  {out_file}  ({size_mb:.1f} MB)")
        conv_results[role] = "ok"
    else:
        print("❌  Failed")
        conv_results[role] = "failed"


# ═════════════════════════════════════════════════════════════════════════
# STEP 3 — REGISTER TO T1 SPACE
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 3 — REGISTERING TO T1 SPACE")

try:
    import SimpleITK as sitk  # type: ignore

    print("  SimpleITK ✅\n")

    t1_path = MONAI_DIR / "t1.nii.gz"
    if not t1_path.exists():
        print("  ❌ T1 not found — skipping registration")
    else:
        t1_ref = safe_read_sitk(t1_path, sitk)
        print(f"  T1 reference: {t1_ref.GetSize()}\n")

        # BraTS sequences — rigid registration
        for src, dst in [
            ("t1c.nii.gz", "t1c_resampled.nii.gz"),
            ("t2.nii.gz", "t2_resampled.nii.gz"),
            ("flair.nii.gz", "flair_resampled.nii.gz"),
        ]:
            src_p = MONAI_DIR / src
            dst_p = MONAI_DIR / dst
            if not src_p.exists():
                print(f"  ⚠️  {src} not found — skipping")
                continue
            print(f"  Registering {src} → {dst}…", end=" ", flush=True)
            metric = register(sitk, src_p, t1_ref, dst_p)
            print(f"✅  metric={metric:.4f}" if metric == metric else "✅  (geometry resample)")

        # Extra sequences — resample only (no registration)
        for src, dst in [
            ("t2star.nii.gz", "t2star_resampled.nii.gz"),
            ("dwi.nii.gz", "dwi_resampled.nii.gz"),
            ("adc.nii.gz", "adc_resampled.nii.gz"),
            ("ct_brain.nii.gz", "ct_brain_resampled.nii.gz"),
        ]:
            src_p = EXTRA_DIR / src
            dst_p = EXTRA_DIR / dst
            if not src_p.exists():
                continue
            print(f"  Resampling  {src}…", end=" ", flush=True)
            try:
                moving = safe_read_sitk(src_p, sitk)
                resampled = sitk.Resample(moving, t1_ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                sitk.WriteImage(resampled, str(dst_p))
                print("✅")
            except Exception as e:
                print(f"❌  {e}")

except ImportError:
    print("  ⚠️  SimpleITK not installed — copying without registration")
    print("      Install:  pip install SimpleITK\n")
    for src, dst in [
        ("t1c.nii.gz", "t1c_resampled.nii.gz"),
        ("t2.nii.gz", "t2_resampled.nii.gz"),
        ("flair.nii.gz", "flair_resampled.nii.gz"),
    ]:
        s, d = MONAI_DIR / src, MONAI_DIR / dst
        if s.exists() and not d.exists():
            shutil.copy(s, d)
            print(f"  Copied: {dst}")


# ═════════════════════════════════════════════════════════════════════════
# STEP 4 — VERIFY
# ═════════════════════════════════════════════════════════════════════════
banner("STEP 4 — VERIFYING OUTPUT")

try:
    import nibabel as nib  # type: ignore

    required = {
        "T1": MONAI_DIR / "t1.nii.gz",
        "T1c": MONAI_DIR / "t1c_resampled.nii.gz",
        "T2": MONAI_DIR / "t2_resampled.nii.gz",
        "FLAIR": MONAI_DIR / "flair_resampled.nii.gz",
    }

    all_ok = True
    shapes = {}
    print(f"\n  BraTS-ready files → {MONAI_DIR}\n")
    for name, path in required.items():
        if path.exists():
            img = nib.load(str(path))
            vox = img.header.get_zooms()[:3]
            shapes[name] = img.shape
            print(f"  ✅ {name:5s}: shape={img.shape}  voxel={tuple(round(float(v), 2) for v in vox)} mm")  # type: ignore
        else:
            print(f"  ❌ {name:5s}: MISSING")
            all_ok = False

    unique = set(shapes.values())
    if len(unique) > 1:
        print("\n  ⚠️  Shape mismatch — sequences not aligned:")
        for k, v in shapes.items():
            print(f"     {k}: {v}")
    elif all_ok:
        print("\n  ✅ All 4 BraTS sequences present and shape-matched!")

    print(f"\n  Extra sequences → {EXTRA_DIR}\n")
    for fname in ["t2star_resampled.nii.gz", "dwi_resampled.nii.gz", "adc_resampled.nii.gz"]:
        p = EXTRA_DIR / fname
        if p.exists():
            img = nib.load(str(p))
            print(f"  ✅ {fname:35s} shape={img.shape}")
        else:
            print(f"  ⚠️  {fname:35s} not available")

except ImportError:
    print("  pip install nibabel  to verify output")


# ═════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════
banner("SUMMARY")

ok_n = sum(1 for v in conv_results.values() if v == "ok")
skip_n = sum(1 for v in conv_results.values() if v in ("missing", "empty"))
fail_n = sum(1 for v in conv_results.values() if v == "failed")

print(f"\n  Conversion results:\n    ✅ Converted : {ok_n}")
if skip_n > 0:
    print(f"    ⚠️  Skipped  : {skip_n}")
if fail_n > 0:
    print(f"    ❌ Failed    : {fail_n}")

print(f"""
  Output:
  {NIFTI_DIR}/
  ├── monai_ready/
  │   ├── t1.nii.gz
  │   ├── t1c_resampled.nii.gz
  │   ├── t2_resampled.nii.gz
  │   └── flair_resampled.nii.gz
  └── extra_sequences/
      ├── t2star_resampled.nii.gz
      ├── dwi_resampled.nii.gz
      └── adc_resampled.nii.gz
""")
print("═" * 65)
print("  ✅  Stage 1 done  →  next: Stage 2 (ct_integration.py)")
print("═" * 65)
