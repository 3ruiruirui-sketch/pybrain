#!/usr/bin/env python3
"""
Stage 2 — CT Integration for Brain Tumor Analysis
===================================================
Converts CT DICOM → NIfTI, registers to MRI T1 space,
extracts calcification mask (HU > 130), haemorrhage mask,
tumour density map, and merges with BraTS MRI segmentation.

Reads paths from session (set by run_pipeline.py).

Standalone merge mode (Stage 2b):
    python 2_ct_integration.py --merge-only
Runs the CT→MRI merge step independently after Stage 3 completes,
reading the segmentation from OUTPUT_DIR and writing merged output there.
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

# ── Project imports ───────────────────────────────────────────────────────
import numpy as np  # type: ignore
import nibabel as nib  # type: ignore
import SimpleITK as sitk  # type: ignore
from scipy import ndimage  # type: ignore

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load session ──────────────────────────────────────────────────────────
from scripts.session_loader import get_session, get_paths  # type: ignore

sess = get_session()
_paths = get_paths(sess)

# ── All paths derived from session ────────────────────────────────────────
OUTPUT_DIR = _paths["output_dir"]
NIFTI_DIR = _paths["nifti_dir"]
MONAI_DIR = _paths["monai_dir"]
EXTRA_DIR = _paths["extra_dir"]
RESULTS_DIR = _paths["results_dir"]
WORK_DIR = NIFTI_DIR / "ct_work"

# CT DICOM source — from session assignments
ASSIGNMENTS = sess.get("assignments", {})
SERIES_PATHS = {k: Path(v) for k, v in sess.get("series_paths", {}).items()}


# Remap any stale paths by searching inside PROJECT_ROOT
def _remap_series_paths(series_paths: dict) -> dict:
    broken = [k for k, v in series_paths.items() if not Path(v).exists()]
    if not broken:
        return series_paths
    found = {}
    for depth in ["*", "*/*", "*/*/*"]:
        for item in PROJECT_ROOT.glob(depth):
            if item.is_dir() and item.name not in found:
                found[item.name] = item
    fixed = dict(series_paths)
    for key, old in series_paths.items():
        if not Path(old).exists():
            name = Path(old).name
            if name in found:
                fixed[key] = str(found[name])
                print(f"  ℹ️  Remapped: {name}")
    return fixed


SERIES_PATHS = _remap_series_paths(SERIES_PATHS)

# Build CT_DIR from the CT assignment in session
_ct_role = ASSIGNMENTS.get("CT") or ASSIGNMENTS.get("CT_bone")
_ct_path = SERIES_PATHS.get(_ct_role) if _ct_role else None
CT_DIR = Path(_ct_path).parent if _ct_path else None

# MRI reference
MRI_DIR = MONAI_DIR


# ─────────────────────────────────────────────────────────────────────────
# STANDALONE MERGE MODE — Stage 2b (Early Exit)
# ─────────────────────────────────────────────────────────────────────────
def _run_merge_only(sess, paths):
    """Stage 2b: Merge pre-computed CT masks into AI segmentation."""
    from scipy.ndimage import zoom as scipy_zoom

    def banner(msg):
        print("\n" + "═" * 65)
        print(f"  {msg}")
        print("═" * 65)

    banner("STAGE 2b — CT → MRI SEGMENTATION MERGE (Standalone)")

    _out_dir = Path(paths["output_dir"])
    _monai_dir = Path(paths["monai_dir"])

    # Load pre-computed CT masks from Stage 2 output
    ct_reg_path = _out_dir / "ct_brain_registered.nii.gz"
    calc_path = _out_dir / "ct_calcification.nii.gz"
    haem_path = _out_dir / "ct_haemorrhage.nii.gz"
    seg_names = ["segmentation_full.nii.gz", "segmentation_ensemble.nii.gz"]
    seg_path = None
    for n in seg_names:
        p = _out_dir / n
        if p.exists():
            seg_path = p
            break

    if not seg_path:
        print("  ❌ No AI segmentation found in output folder.")
        print("     Run Stage 3 first: python run_pipeline.py")
        sys.exit(1)
    if not ct_reg_path.exists() or not calc_path.exists():
        print("  ❌ CT masks from Stage 2 not found.")
        print("     Re-run Stage 2 before Stage 2b.")
        sys.exit(1)

    print(f"  Base segmentation : {seg_path.name}")
    print(f"  CT registration   : {ct_reg_path.name}")

    # Load CT mask arrays
    ct_itk = sitk.ReadImage(str(ct_reg_path), sitk.sitkFloat32)
    ct_arr = np.transpose(sitk.GetArrayFromImage(ct_itk), (2, 1, 0)).astype(np.float32)
    calc_arr = nib.load(str(calc_path)).get_fdata().astype(np.float32)
    haem_arr = nib.load(str(haem_path)).get_fdata().astype(np.float32)

    seg_nib = nib.load(str(seg_path))
    seg_arr = seg_nib.get_fdata().astype(np.uint8)
    affine = seg_nib.affine

    # Resample CT masks to segmentation space if needed
    if ct_arr.shape != seg_arr.shape:
        factors = np.array(seg_arr.shape) / np.array(ct_arr.shape)
        scipy_zoom(ct_arr, factors, order=1)
        calc_rs = scipy_zoom(calc_arr, factors, order=0)
        haem_rs = scipy_zoom(haem_arr, factors, order=0)
        print(f"  ℹ️  CT resampled to segmentation grid: {seg_arr.shape}")
    else:
        calc_rs = calc_arr
        haem_rs = haem_arr

    # Execute merge: CT calcification/hemorrhage overrides MRI segmentation
    merged = seg_arr.copy()
    calc_mask = (calc_rs > 0).astype(bool)
    haem_near = (haem_rs > 0) & ndimage.binary_dilation((seg_arr > 0), iterations=5)
    merged[calc_mask] = 1
    merged[haem_near] = 1

    m_path = _out_dir / "segmentation_ct_merged.nii.gz"
    nib.save(nib.Nifti1Image(merged.astype(np.uint8), affine), str(m_path))
    print(f"  ✅ Merge complete → {m_path.name}")
    banner("STAGE 2b COMPLETE")


if "--merge-only" in sys.argv or os.environ.get("PYBRAIN_MERGE_ONLY") == "1":
    _run_merge_only(sess, _paths)
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────
# MAIN STAGE 2 EXECUTION (CT Conversion + Registration)
# ─────────────────────────────────────────────────────────────────────────

# Create working directories
EXTRA_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# CT Hounsfield Unit thresholds
HU_CALCIFICATION_LOW = 130
HU_CALCIFICATION_HIGH = 1000
HU_HAEMORRHAGE_LOW = 50
HU_HAEMORRHAGE_HIGH = 90
HU_TUMOUR_LOW = 25
HU_TUMOUR_HIGH = 60

# ─────────────────────────────────────────────────────────────────────────


def banner(t):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


def check_dcm2niix():
    for c in ["dcm2niix", "/opt/homebrew/bin/dcm2niix", "/usr/local/bin/dcm2niix"]:
        if shutil.which(c):
            return c
    print("❌ dcm2niix not found. Run:  brew install dcm2niix")
    sys.exit(1)


def convert_dicom(dcm2niix, src_dir, out_dir, stem):
    """Convert a DICOM folder to NIfTI. Returns path to output file."""
    cmd = [dcm2niix, "-o", str(out_dir), "-f", stem, "-z", "y", "-b", "y", "-v", "0", str(src_dir)]
    subprocess.run(cmd, capture_output=True, text=True)
    candidates = sorted(out_dir.glob(f"{stem}*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        return None
    target = out_dir / f"{stem}.nii.gz"
    if candidates[0] != target:
        candidates[0].rename(target)
    return target


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — Convert CT DICOM → NIfTI
# ─────────────────────────────────────────────────────────────────────────
banner("STEP 1 — CONVERTING CT DICOM → NIfTI")

dcm2niix = check_dcm2niix()
ct_series = {}
for role in ("CT", "CT_bone"):
    folder_name = ASSIGNMENTS.get(role)
    if folder_name and folder_name in SERIES_PATHS:
        src_path = SERIES_PATHS[folder_name]
        if Path(src_path).exists():
            stem = "ct_brain" if role == "CT" else "ct_bone"
            ct_series[stem] = Path(src_path)
            print(f"  {role:8s} ← {folder_name}")

if not ct_series:
    print("❌  No CT series found in session.")
    sys.exit(1)

ct_nifti = {}
for stem, src in ct_series.items():
    print(f"  Converting {stem}…", end=" ", flush=True)
    out = convert_dicom(dcm2niix, src, WORK_DIR, stem)
    if out and out.exists():
        ct_nifti[stem] = out
        print(f"✅  {out.name}")
    else:
        print("❌  Failed")

# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — Register CT to MRI T1 space
# ─────────────────────────────────────────────────────────────────────────
banner("STEP 2 — REGISTERING CT → MRI T1 SPACE")

t1_path = MRI_DIR / "t1c_resampled.nii.gz"
if not t1_path.exists():
    t1_path = MRI_DIR / "t1.nii.gz"

if not t1_path.exists():
    print(f"  ❌ MRI Reference not found at {t1_path}")
    sys.exit(1)

print(f"  MRI Reference: {t1_path.name}")

t1_ref = sitk.ReadImage(str(t1_path), sitk.sitkFloat32)
ct_registered = {}

for stem, ct_path in ct_nifti.items():
    print(f"  Registering {stem}…")
    ct_raw = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    try:
        initial_tx = sitk.CenteredTransformInitializer(
            t1_ref, ct_raw, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    except RuntimeError:
        initial_tx = sitk.Euler3DTransform()

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)
    reg.SetInitialTransform(initial_tx)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    try:
        final_tx = reg.Execute(t1_ref, ct_raw)
        ct_resampled = sitk.Resample(ct_raw, t1_ref, final_tx, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
        out_path = WORK_DIR / f"{stem}_registered.nii.gz"
        sitk.WriteImage(ct_resampled, str(out_path))
        ct_registered[stem] = out_path
        print("    ✅ Registration complete")
    except Exception as e:
        print(f"    ❌ Registration failed for {stem}: {e}")
        # F8: Propagate flag to session so downstream stages know about the misalignment
        try:
            session_file = os.environ.get("PYBRAIN_SESSION")
            if session_file and Path(session_file).exists():
                with open(session_file, "r") as f:
                    session_data = json.load(f)

                # Add registration status to session
                registration_status = session_data.get("registration_status", {})
                registration_status[stem] = {"success": False, "error": str(e)}
                session_data["registration_status"] = registration_status

                with open(session_file, "w") as f:
                    json.dump(session_data, f, indent=2)
                print("    ℹ️  Registration failure flag saved to session.")
        except Exception as json_err:
            print(f"    ⚠️  Could not update session JSON: {json_err}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Extract CT-derived masks
# ─────────────────────────────────────────────────────────────────
banner("STEP 3 — EXTRACTING CT-DERIVED MASKS")

primary_ct_key = "ct_brain" if "ct_brain" in ct_registered else "ct_bone"
if primary_ct_key not in ct_registered:
    print("❌ No registered CT found for mask extraction.")
    sys.exit(1)

ct_img = sitk.ReadImage(str(ct_registered[primary_ct_key]), sitk.sitkFloat32)
ct_arr = np.transpose(sitk.GetArrayFromImage(ct_img), (2, 1, 0)).astype(np.float32)

t1_nib = nib.load(str(t1_path))
affine = t1_nib.affine


def save_mask(arr, fname, description):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(OUTPUT_DIR / fname))
    vol = (arr > 0).sum() / 1000.0
    print(f"  ✅ {description:35s} {vol:6.1f} cc  → {fname}")


calc_raw = ((ct_arr >= HU_CALCIFICATION_LOW) & (ct_arr <= HU_CALCIFICATION_HIGH)).astype(np.float32)

# Brain mask filtering (Stage 1b writes to monai_ready/; Stage 3 may also save to output root)
_brain_out = OUTPUT_DIR / "brain_mask.nii.gz"
_brain_monai = MONAI_DIR / "brain_mask.nii.gz"
brain_mask_path = _brain_out if _brain_out.exists() else _brain_monai
bm_arr = None
bm_for_ct = None  # brain mask resampled to CT grid (None = no mask to apply)
if brain_mask_path.exists():
    bm_arr = nib.load(str(brain_mask_path)).get_fdata().astype(np.float32)
    if bm_arr.shape == calc_raw.shape:
        print("  🧠 Applying brain mask to focus only on intracranial findings...")
        bm_for_ct = bm_arr
    else:
        # F10: Shape mismatch CT vs brain mask — add robust resampling
        print(f"  ⚠️  Shape mismatch caught in Stage 2: brain_mask {bm_arr.shape} ≠ CT {calc_raw.shape}")
        print("     Resampling brain mask to CT grid using scipy.ndimage.zoom...")
        try:
            from scipy.ndimage import zoom as scipy_zoom

            # Calculate zoom factors per dimension
            factors = np.array(calc_raw.shape) / np.array(bm_arr.shape)
            bm_for_ct = scipy_zoom(bm_arr, factors, order=0).clip(0, 1)  # nearest-neighbour for mask
            print(f"  ✅ Brain mask successfully resampled to {bm_for_ct.shape}")
        except Exception as e:
            print(f"  ❌ Resample failed ({e}) — skull-stripping SKIPPED for CT masks.")
else:
    print("  ⚠️  WARNING: brain_mask.nii.gz not found.")
    print("     Skull-stripping is inactive — calcification volumes may be inflated by bone/teeth.")

# Apply brain mask to calcification BEFORE cleanup
if bm_for_ct is not None:
    calc_raw *= bm_for_ct

# ── Spatial filtering: exclude physiological calcifications ──────────────────
# Load FLAIR to identify tumor/edema regions (high signal)
flair_path = MONAI_DIR / "flair.nii.gz"
tumor_region_mask = None
if flair_path.exists():
    try:
        flair_img = nib.load(str(flair_path))
        flair_data = flair_img.get_fdata()

        # Normalize FLAIR intensities
        p2, p98 = np.percentile(flair_data[flair_data > 0], [2, 98])
        flair_norm = np.clip((flair_data - p2) / (p98 - p2 + 1e-8), 0, 1)

        # High FLAIR signal = potential tumor/edema (threshold at 75th percentile)
        high_signal = flair_norm > 0.60

        # Dilate tumor region by 30mm to capture nearby calcifications
        # (30mm = ~30 voxels at 1mm resolution)
        tumor_region_dilated = ndimage.binary_dilation(high_signal, iterations=30)

        # Resample tumor mask to CT space if shapes don't match
        if tumor_region_dilated.shape != calc_raw.shape:
            from scipy.ndimage import zoom as scipy_zoom

            factors = np.array(calc_raw.shape) / np.array(tumor_region_dilated.shape)
            tumor_region_mask = scipy_zoom(tumor_region_dilated.astype(np.float32), factors, order=0) > 0.5
            print(f"  ℹ️  Tumor region mask resampled: {tumor_region_dilated.shape} → {tumor_region_mask.shape}")
        else:
            tumor_region_mask = tumor_region_dilated

        print("  ℹ️  Tumor region mask created from FLAIR (dilated 30mm)")
    except Exception as e:
        print(f"  ⚠️  Could not load FLAIR for spatial filtering: {e}")

# Cleanup calcification mask
labeled, n = ndimage.label(calc_raw)
if n > 0:
    sizes = ndimage.sum(calc_raw, labeled, range(1, n + 1))
    calc_clean = np.zeros_like(calc_raw)
    for i, sz in enumerate(sizes):
        if sz >= 5:
            component_mask = labeled == (i + 1)

            # If tumor region mask exists, only keep calcifications near tumor
            if tumor_region_mask is not None:
                overlap = np.sum(component_mask & tumor_region_mask)
                if overlap > 0:  # At least some overlap with tumor region
                    calc_clean[component_mask] = 1.0
                else:
                    print(f"  🗑️  Excluded calcification component {i + 1} (isolated, {sz:.0f} voxels)")
            else:
                # No FLAIR available — keep all calcifications (fallback)
                calc_clean[component_mask] = 1.0
else:
    calc_clean = calc_raw

save_mask(calc_clean, "ct_calcification.nii.gz", "Calcifications (HU 130–1000, tumor-proximal)")

# Note: thick-slice CT (>5mm) over-estimates calcification volume proportionally.
# A 25mm-slice CT showing 438cc may correspond to ~17cc on 1mm CT.
# The non_gbm_ct_calc_threshold in config must account for this.

haem_raw = ((ct_arr >= HU_HAEMORRHAGE_LOW) & (ct_arr <= HU_HAEMORRHAGE_HIGH)).astype(np.float32)
if bm_for_ct is not None:
    haem_raw *= bm_for_ct

labeled, n = ndimage.label(haem_raw)
if n > 0:
    sizes = ndimage.sum(haem_raw, labeled, range(1, n + 1))
    haem_clean = np.zeros_like(haem_raw)
    for i, sz in enumerate(sizes):
        if sz >= 10:
            haem_clean[labeled == (i + 1)] = 1.0
else:
    haem_clean = haem_raw

save_mask(haem_clean, "ct_haemorrhage.nii.gz", "Acute haemorrhage (HU 50–90)")

tumour_ct = ((ct_arr >= HU_TUMOUR_LOW) & (ct_arr <= HU_TUMOUR_HIGH)).astype(np.float32)
if bm_for_ct is not None:
    tumour_ct *= bm_for_ct
# Not loaded by Stage 3 — available for manual review
save_mask(tumour_ct, "ct_tumour_density.nii.gz", "Hyperdense tumour (HU 25–60)")
nib.save(nib.Nifti1Image(tumour_ct, affine), str(EXTRA_DIR / "ct_tumour_density.nii.gz"))

_ct_reg = OUTPUT_DIR / "ct_brain_registered.nii.gz"
nib.save(nib.Nifti1Image(ct_arr, affine), str(_ct_reg))
# Stage 3 looks in monai_ready/ for CT boost — keep a copy there too
shutil.copy2(str(_ct_reg), str(MONAI_DIR / "ct_brain_registered.nii.gz"))

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Merge CT + MRI Segmentation
# ─────────────────────────────────────────────────────────────────
banner("STEP 4 — MERGING CT + MRI SEGMENTATION")

# Load segmentation from current session output directory
target_names = ["segmentation_full.nii.gz", "segmentation_ensemble.nii.gz", "segmentation_ct_merged.nii.gz"]
latest_seg = None
for name in target_names:
    p = OUTPUT_DIR / name
    if p.exists():
        latest_seg = p
        break

if not latest_seg:
    print("  ℹ️  No AI segmentation in output folder yet.")
    print("     This is normal when Stage 2 runs before Stage 3 (default pipeline order).")
    print("     Merge is skipped — CT-derived masks are still saved. After Stage 3 finishes,")
    print("     re-run Stage 2 to write segmentation_ct_merged.nii.gz, or merge in an external tool.")
    merged = np.zeros_like(ct_arr, dtype=np.uint8)
else:
    print(f"  Using base segmentation: {latest_seg.name}")
    seg_nib = nib.load(str(latest_seg))
    seg_arr = seg_nib.get_fdata().astype(np.uint8)

    ct_for_merge = None
    merge_resampled = False

    if seg_arr.max() == 0:
        print("  ⚠️ MRI segmentation is empty — skipping merge")
        merged = seg_arr.copy()
    else:
        if ct_arr.shape != seg_arr.shape:
            print(f"  ⚠️  Shape mismatch: CT {ct_arr.shape} vs SEG {seg_arr.shape} — attempting resample")
            try:
                from scipy.ndimage import zoom as scipy_zoom

                factors = np.array(seg_arr.shape) / np.array(ct_arr.shape)
                ct_for_merge = scipy_zoom(ct_arr, factors, order=1)
                # Resample calc_clean and haem_clean too — they are on original ct_arr grid
                calc_for_merge = scipy_zoom(calc_clean, factors, order=0)  # nearest-neighbour
                haem_for_merge = scipy_zoom(haem_clean, factors, order=0)
                print(f"  ℹ️  CT resampled to {ct_for_merge.shape} to match segmentation")
                merge_resampled = True
            except Exception as e:
                print(f"  ⚠️  Resample failed ({e}) — merge SKIPPED, using MRI segmentation only")
                merged = seg_arr.copy()
                ct_for_merge = None
        else:
            ct_for_merge = ct_arr
            calc_for_merge = calc_clean
            haem_for_merge = haem_clean
            merge_resampled = False

        if ct_for_merge is not None:
            merged = seg_arr.copy()
            # SOBERANIA DO CT: Calcificações (>130 HU) são obrigatoriamente NÚCLEO (Label 1)
            calc_mask = (calc_for_merge > 0).astype(bool)
            merged[calc_mask] = 1
            # Hemorragia perto do tumor também é integrada no núcleo
            haem_near = (haem_for_merge > 0) & ndimage.binary_dilation((seg_arr > 0), iterations=5)
            merged[haem_near] = 1
            m_path = OUTPUT_DIR / "segmentation_ct_merged.nii.gz"
            nib.save(nib.Nifti1Image(merged, seg_nib.affine), str(m_path))
            note = " (CT resampled)" if merge_resampled else ""
            print(f"  ✅ Physical Priority Merge complete{note} -> {m_path.name}")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Density Analysis
# ─────────────────────────────────────────────────────────────────
banner("STEP 5 — CT DENSITY ANALYSIS")
if merged.any():
    ct_in_tumor = ct_arr[merged > 0]
    print(f"  Mean HU in tumour: {ct_in_tumor.mean():.1f} ± {ct_in_tumor.std():.1f}")
    print(f"  Max HU in tumour : {ct_in_tumor.max():.1f}")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
banner("SUMMARY")
print(f"""
  Files created in:
  {OUTPUT_DIR}/
  ├── ct_brain_registered.nii.gz     ← CT in MRI space
  ├── ct_calcification.nii.gz        ← HU 130-1000 (calcified)
  ├── ct_haemorrhage.nii.gz          ← HU 50-90 (haemorrhagic)
  └── segmentation_ct_merged.nii.gz    ← Final combined mask

  NEXT STEPS:
  The merged segmentation is ready for review in any medical image viewer.
""")
# ── Cleanup temporary files ──────────────────────────────────────
print("\n  🧹 Cleaning up temporary files...")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    print(f"  ✅ Removed {WORK_DIR.name}/")

# Also remove uncompressed NIfTI copies if Stage 1 left them
for p in NIFTI_DIR.rglob("*.nii"):
    if p.is_file():
        p.unlink()

print("═" * 65)
print("  ✅  Done!")
print("═" * 65)
