#!/usr/bin/env python3
"""
PY-BRAIN Pipeline Integration Test
====================================
Validates that 3_brain_tumor_analysis_v5_ensemble.py produced all
expected outputs and that 8_radiomics_analysis.py can consume them.

Run:
    python3 scripts/test_pipeline_integration.py

Pass: prints ✅ for each check, exits 0
Fail: prints ❌ for failed checks, exits 1
"""

import sys
import json
import numpy as np
from pathlib import Path

# ── Session ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.session_loader import get_session, get_paths

_sess = get_session()
_paths = get_paths(_sess)
OUTPUT_DIR = Path(_paths.get("output_dir", _paths["results_dir"]))
MONAI_DIR = Path(_paths["monai_dir"])

failures = []


def ok(msg):
    print(f"  ✅ {msg}")


def fail(msg):
    print(f"  ❌ {msg}")
    failures.append(msg)


def banner(t):
    print("\n" + "═" * 65 + f"\n  {t}\n" + "═" * 65)


# ── BLOCK 1: Ensemble Output Files ───────────────────────────────────
banner("BLOCK 1 — Ensemble Output Files")

REQUIRED_NIFTIS = [
    "segmentation_ensemble.nii.gz",
    "segmentation_full.nii.gz",
    "seg_necrotic.nii.gz",
    "seg_edema.nii.gz",
    "seg_enhancing.nii.gz",
    "ensemble_probability.nii.gz",
    "ensemble_uncertainty.nii.gz",
]

OPTIONAL_NIFTIS = [
    "segmentation_segresnet.nii.gz",
    "segmentation_tta4.nii.gz",
]

REQUIRED_IMAGES = [
    "uncertainty_analysis.png",
    "swinunetr_attention_maps.png",  # Placeholder (Transformers omitted)
    "view_axial.png",
    "view_coronal.png",
    "view_sagittal.png",
]

for f in REQUIRED_NIFTIS:
    p = OUTPUT_DIR / f
    if p.exists() and p.stat().st_size > 1000:
        ok(f"{f} ({p.stat().st_size // 1024} KB)")
    else:
        fail(f"{f} — missing or empty")

for f in OPTIONAL_NIFTIS:
    p = OUTPUT_DIR / f
    if p.exists():
        ok(f"{f} [secondary/TTA] ({p.stat().st_size // 1024} KB)")
    else:
        print(f"  ℹ️  {f} — not present (TTA or single-model run)")

for f in REQUIRED_IMAGES:
    p = OUTPUT_DIR / f
    if p.exists() and p.stat().st_size > 5000:
        ok(f"{f} ({p.stat().st_size // 1024} KB)")
    else:
        fail(f"{f} — missing or suspiciously small")


# ── BLOCK 2: JSON Schema Validation ──────────────────────────────────
banner("BLOCK 2 — tumor_stats_ensemble.json Schema")

json_path = OUTPUT_DIR / "tumor_stats_ensemble.json"
if not json_path.exists():
    fail("tumor_stats_ensemble.json — file not found")
else:
    try:
        with open(json_path) as f:
            stats = json.load(f)

        required_keys = {
            "segmentation_source": str,
            "volume_cc": dict,
            "voxel_geometry": list,
            "uncertainty_metrics": dict,
        }
        for key, expected_type in required_keys.items():
            if key not in stats:
                fail(f"JSON missing key: {key}")
            elif not isinstance(stats[key], expected_type):
                fail(f"JSON key '{key}' wrong type: got {type(stats[key])}")
            else:
                ok(f"JSON key '{key}' present and correct type")

        vol = stats.get("volume_cc", {})
        for sub in ["whole_tumor", "necrotic_core", "edema", "enhancing"]:
            if sub not in vol:
                fail(f"volume_cc missing sub-key: {sub}")
            elif vol[sub] < 0:
                fail(f"volume_cc.{sub} is negative: {vol[sub]}")
            else:
                ok(f"volume_cc.{sub} = {vol[sub]:.2f} cc")

        unc = stats.get("uncertainty_metrics", {})
        pct = unc.get("high_uncertainty_pct", -1)
        if pct < 0 or pct > 100:
            fail(f"high_uncertainty_pct out of range: {pct}")
        else:
            ok(f"high_uncertainty_pct = {pct:.1f}%")

    except json.JSONDecodeError as e:
        fail(f"tumor_stats_ensemble.json — invalid JSON: {e}")


# ── BLOCK 3: NIfTI Shape & Affine Consistency ─────────────────────────
banner("BLOCK 3 — NIfTI Shape & Affine Consistency")

try:
    import nibabel as nib

    ref_path = MONAI_DIR / "t1.nii.gz"
    if not ref_path.exists():
        fail(f"Reference T1 not found: {ref_path}")
    else:
        ref = nib.load(str(ref_path))
        ref_shape = ref.shape[:3]
        ref_affine = ref.affine

        for f in ["segmentation_ensemble.nii.gz", "ensemble_probability.nii.gz", "ensemble_uncertainty.nii.gz"]:
            p = OUTPUT_DIR / f
            if not p.exists():
                continue
            img = nib.load(str(p))
            if img.shape[:3] != ref_shape:
                fail(f"{f} shape {img.shape[:3]} != T1 shape {ref_shape}")
            elif not np.allclose(img.affine, ref_affine, atol=1e-3):
                fail(f"{f} affine mismatch vs T1")
            else:
                ok(f"{f} shape {img.shape[:3]} ✓ affine ✓")

except ImportError:
    print("  ℹ️  nibabel not available — skipping NIfTI checks")


# ── BLOCK 4: Segmentation Label Sanity ───────────────────────────────
banner("BLOCK 4 — Segmentation Label Sanity")

try:
    seg_path = OUTPUT_DIR / "segmentation_ensemble.nii.gz"
    if seg_path.exists():
        seg = nib.load(str(seg_path)).get_fdata().astype(np.uint8)
        labels = np.unique(seg)
        unexpected = [l for l in labels if l not in [0, 1, 2, 3]]
        if unexpected:
            fail(f"Unexpected label values: {unexpected}")
        else:
            ok(f"Labels in segmentation: {labels.tolist()}")

        wt_vox = int((seg > 0).sum())
        if wt_vox == 0:
            fail("Whole tumor is empty (0 voxels) — possible inference failure")
        else:
            ok(f"Whole tumor voxels: {wt_vox:,}")

        et_vox = int((seg == 4).sum())  # BraTS 2021: ET = label 4
        nc_vox = int((seg == 1).sum())
        ed_vox = int((seg == 2).sum())
        ok(f"Necrotic={nc_vox:,}  Edema={ed_vox:,}  Enhancing={et_vox:,}")

except Exception as e:
    fail(f"Segmentation sanity check error: {e}")


# ── BLOCK 5: Radiomics Input Availability ───────────────────────────
banner("BLOCK 5 — Radiomics Inputs (8_radiomics_analysis.py)")

RADIOMICS_INPUTS = {
    "segmentation_ensemble.nii.gz": "Primary mask (prefer)",
    "segmentation_full.nii.gz": "Fallback mask",
    "ensemble_probability.nii.gz": "Probability map",
    "tumor_stats_ensemble.json": "Volume reference",
}
for f, role in RADIOMICS_INPUTS.items():
    p = OUTPUT_DIR / f
    if p.exists():
        ok(f"{f} — {role}")
    else:
        fail(f"{f} — {role} — MISSING")

mri_files = {
    "t1.nii.gz": MONAI_DIR / "t1.nii.gz",
    "t1c_resampled.nii.gz": MONAI_DIR / "t1c_resampled.nii.gz",
    "t2_resampled.nii.gz": MONAI_DIR / "t2_resampled.nii.gz",
    "flair_resampled.nii.gz": MONAI_DIR / "flair_resampled.nii.gz",
}
for name, path in mri_files.items():
    if path.exists():
        ok(f"MRI input: {name}")
    else:
        fail(f"MRI input: {name} — MISSING at {path}")


# ── Final Report ─────────────────────────────────────────────────────
banner("INTEGRATION TEST RESULT")

total_checks = len(REQUIRED_NIFTIS) + len(REQUIRED_IMAGES) + 4 + 3 + 3 + len(RADIOMICS_INPUTS) + len(mri_files)

if failures:
    print(f"\n  ❌ FAILED — {len(failures)} issue(s) found:\n")
    for f in failures:
        print(f"     • {f}")
    print()
    sys.exit(1)
else:
    print("\n  ✅ ALL CHECKS PASSED — Pipeline ready for 8_radiomics_analysis.py\n")
    sys.exit(0)
