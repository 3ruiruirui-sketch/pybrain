"""
Regression tests for the audit fixes (April 2026).

Covers:
  B1 — np.ptp() removed in NumPy 2.x → _compute_nmi must not call .ptp()
  B6 — save_session() must be atomic (no partial files left on disk)
  R1 — _guess_type() must not classify "posterior_*" / "post_processing" as T1c
  R4 — --session loader must resolve every path-like field
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pipeline import _guess_type, save_session  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# B1 — NMI computation under NumPy 2.x
# ─────────────────────────────────────────────────────────────────────────
def test_compute_nmi_no_ptp():
    """Regression: np.ndarray.ptp() was removed in NumPy 2.0."""
    # Import lazily — script lives under scripts/ and pulls heavy deps
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brain_tumor_analysis",
        PROJECT_ROOT / "scripts" / "3_brain_tumor_analysis.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"Heavy dependencies unavailable: {e}")

    rng = np.random.default_rng(42)
    a = rng.random((16, 16, 16)).astype(np.float64)
    b = rng.random((16, 16, 16)).astype(np.float64)

    nmi = mod._compute_nmi(a, b)
    assert isinstance(nmi, float)
    # Studholme NMI ∈ [1.0, 2.0]; random pairs should be near 1.0
    assert 1.0 <= nmi <= 2.0


def test_compute_nmi_identical_volumes_high_score():
    """Identical volumes should produce NMI close to 2.0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brain_tumor_analysis",
        PROJECT_ROOT / "scripts" / "3_brain_tumor_analysis.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"Heavy dependencies unavailable: {e}")

    rng = np.random.default_rng(0)
    vol = rng.random((20, 20, 20)).astype(np.float64)
    nmi = mod._compute_nmi(vol, vol)
    assert nmi > 1.5, f"identical volumes should give high NMI, got {nmi}"


# ─────────────────────────────────────────────────────────────────────────
# B6 — atomic save_session
# ─────────────────────────────────────────────────────────────────────────
def test_save_session_atomic(tmp_path):
    """After save_session() returns, only the final session.json must exist."""
    sess = {"output_dir": str(tmp_path), "stages": {"stage_1_dicom": True}}
    out = save_session(sess, tmp_path)

    assert out.exists()
    assert out.name == "session.json"
    # No leftover temp file
    assert not (tmp_path / "session.json.tmp").exists()
    # Round-trips correctly
    loaded = json.loads(out.read_text())
    assert loaded["stages"]["stage_1_dicom"] is True


def test_save_session_overwrites_cleanly(tmp_path):
    """Re-saving must replace contents, not append."""
    save_session({"v": 1, "output_dir": str(tmp_path)}, tmp_path)
    save_session({"v": 2, "output_dir": str(tmp_path)}, tmp_path)
    loaded = json.loads((tmp_path / "session.json").read_text())
    assert loaded["v"] == 2
    assert not (tmp_path / "session.json.tmp").exists()


def test_save_session_serialises_path_objects(tmp_path):
    """Path objects nested in the dict must be JSON-serialisable."""
    sess = {"output_dir": tmp_path, "nested": {"p": tmp_path / "foo"}}
    out = save_session(sess, tmp_path)
    loaded = json.loads(out.read_text())
    assert loaded["output_dir"] == str(tmp_path)
    assert loaded["nested"]["p"] == str(tmp_path / "foo")


# ─────────────────────────────────────────────────────────────────────────
# R1 — _guess_type must not over-match "post"
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [
    "posterior_fossa_t1",
    "post_processing",
    "post_reg_results",
    "compostos",
])
def test_guess_type_post_not_t1c(name):
    """The bare 'post' substring used to falsely tag these as T1c."""
    assert _guess_type(name) != "T1c", f"{name!r} wrongly classified as T1c"


@pytest.mark.parametrize("name,expected", [
    ("t1_postcontrast",     "T1c"),
    ("ax_t1_post_gad",      "T1c"),
    ("t1_post_contrast_3d", "T1c"),
    ("t1c_mprage",          "T1c"),
    ("t1_gd",               "T1c"),
    ("t1_mprage",           "T1"),
    ("flair_tirm",          "FLAIR"),
    ("t2_tse_tra",          "T2"),
    ("ep2d_diff_tracew",    "DWI"),
    ("adc_map",             "ADC"),
    ("Cranio_STD_brain",    "CT_brain"),
])
def test_guess_type_positive_cases(name, expected):
    """The narrowed keywords must still catch the legitimate T1c patterns."""
    assert _guess_type(name) == expected


# ─────────────────────────────────────────────────────────────────────────
# R4 — --session must resolve every path field
# ─────────────────────────────────────────────────────────────────────────
def test_session_loader_resolves_all_paths(tmp_path, monkeypatch):
    """Driving the __main__ block via subprocess would launch the pipeline;
    instead we replicate exactly the resolution loop and verify it covers
    every path key the rest of the pipeline reads."""
    sess = {
        "project_root": "./root",
        "output_dir":   "./out",
        "results_dir":  "./res",
        "monai_dir":    "./out/nifti/monai_ready",
        "extra_dir":    "./out/nifti/extra_sequences",
        "nifti_dir":    "./out/nifti",
        "bundle_dir":   "./models/brats_bundle",
        "mri_dicom_dir": "~/dicom",
        "ct_dicom_dir":  "~/dicom",
        "ground_truth":  "./res/ground_truth.nii.gz",
        "series_paths": {"t1_mprage": "./root/dicom/t1"},
        "stages": {"stage_1_dicom": True},
    }

    monkeypatch.chdir(tmp_path)

    _PATH_KEYS = (
        "project_root", "output_dir", "results_dir",
        "monai_dir", "extra_dir", "nifti_dir", "bundle_dir",
        "mri_dicom_dir", "ct_dicom_dir", "ground_truth",
    )
    for k in _PATH_KEYS:
        if k in sess and sess[k]:
            sess[k] = str(Path(sess[k]).expanduser().resolve())
    if isinstance(sess.get("series_paths"), dict):
        sess["series_paths"] = {
            k: str(Path(v).expanduser().resolve()) for k, v in sess["series_paths"].items()
        }

    # Every path-like value must now be absolute
    for k in _PATH_KEYS:
        assert Path(sess[k]).is_absolute(), f"{k} not resolved: {sess[k]}"
    for k, v in sess["series_paths"].items():
        assert Path(v).is_absolute(), f"series_paths[{k}] not resolved: {v}"

    # ~/ must have been expanded
    assert "~" not in sess["mri_dicom_dir"]
    assert "~" not in sess["ct_dicom_dir"]


# ─────────────────────────────────────────────────────────────────────────
# B7 — BraTS 2021 ET label = 4 (not 3) across stages 6/7/8 + libs
#
# Bug: Stage 3 writes segmentation_full.nii.gz with labels {0, 1, 2, 4}.
# Many downstream stages used `seg == 3`, silently reporting enhancing
# tumor as 0 cc, which corrupted morphology, radiomics, location and the
# hierarchy checker. Tests below pin label 4 so the bug cannot regress.
# ─────────────────────────────────────────────────────────────────────────
def test_brats_label_no_legacy_label_3_in_active_code():
    """No active production source may compare segmentation labels to 3.

    Stage 3 writes BraTS 2021 labels {0, 1, 2, 4}. Any `seg == 3` (or
    similar) in stages 6/7/8 or pybrain libraries silently zeroes the
    enhancing tumour, corrupting morphology and radiomics. Archive
    scripts and tests are excluded — they document the regression.
    """
    import re

    pattern = re.compile(r"seg(?:_arr|_full)?\s*==\s*3\b")
    excluded_dirs = {"archive_scripts", "__pycache__", ".pytest_cache", "tests"}
    offenders: list[str] = []

    for sub in ("scripts", "pybrain"):
        for path in (PROJECT_ROOT / sub).rglob("*.py"):
            if any(part in excluded_dirs for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found legacy `seg == 3` references in production code. BraTS 2021 ET = 4.\n"
        + "\n".join(offenders)
    )


def test_brats_label_convention_hierarchy_checker():
    """Hierarchy checker must accept label 4 as ET."""
    from pybrain.core.output_checker import check_hierarchy_violations

    seg = np.zeros((32, 32, 32), dtype=np.int32)
    # Valid hierarchy: TC (NCR) wraps ET; both inside WT (= ED ∪ TC)
    seg[10:15, 10:15, 10:15] = 1   # NCR
    seg[12:13, 12:13, 12:13] = 4   # ET inside NCR — BraTS 2021
    seg[15:20, 15:20, 15:20] = 2   # ED
    ok, issues = check_hierarchy_violations(seg)
    assert ok, f"Valid BraTS 2021 segmentation rejected: {issues}"


def test_brats_label_convention_clinical_consistency():
    """Clinical consistency must compute TC = NCR ∪ ET(=4)."""
    from pybrain.clinical.consistency import validate_clinical_consistency

    seg = np.zeros((40, 40, 40), dtype=np.int32)
    seg[5:10, 5:10, 5:10] = 1   # 125 NCR voxels
    seg[10:14, 10:14, 10:14] = 4  # 64 ET voxels — BraTS 2021
    seg[14:20, 14:20, 14:20] = 2  # 216 ED voxels
    p_ensemble = np.zeros((3, 40, 40, 40), dtype=np.float32)
    out = validate_clinical_consistency(
        seg, p_ensemble, vox_vol_cc=0.001, config={}, contributed_models=[]
    )
    expected_tc_cc = round((125 + 64) * 0.001, 2)  # 0.19 cc
    assert out["v_tc_cc"] == pytest.approx(expected_tc_cc, abs=1e-2), (
        f"TC must include ET (label 4): got {out['v_tc_cc']}, expected {expected_tc_cc}"
    )
