# pybrain/io/session.py
"""
Session management and environment handling for PY-BRAIN.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_session() -> Dict[str, Any]:
    """
    Load the active session dictionary.
    Priority:
      1. PYBRAIN_SESSION env variable
      2. Most recent session.json in results/
    """
    # Option 1: env variable
    env_path = os.environ.get("PYBRAIN_SESSION")
    if env_path and Path(env_path).exists():
        with open(env_path) as f:
            sess = json.load(f)
        return _restore_paths(sess)

    # Option 2: most recent session in results/
    # Extract embedded timestamp from folder name to sort correctly.
    # Folders look like "SOARES_MARIA_CELESTE_20260323_031605" or "smoke_20260401_225837"
    import re
    results_dir = PROJECT_ROOT / "results"
    all_json   = list(results_dir.glob("*/session.json")) if results_dir.exists() else []

    def _ts(p: Path):
        m = re.search(r"(\d{8}_\d{6})", p.parent.name)
        return m.group(1) if m else ""

    candidates = sorted([p for p in all_json if _ts(p)], key=_ts)
    latest = candidates[-1] if candidates else None

    if latest is None and all_json:
        # Fallback: no recognisable timestamp — use most recently modified
        latest = sorted(all_json, key=lambda x: x.stat().st_mtime)[-1]

    if latest is None:
        raise RuntimeError("No active session found. Please run run_pipeline.py first.")

    with open(latest) as f:
        sess = json.load(f)
    return _restore_paths(sess)

def _restore_paths(sess: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string paths back to Path objects."""
    path_keys = {
        "project_root", "mri_dicom_dir", "ct_dicom_dir",
        "nifti_dir", "monai_dir", "extra_dir", "bundle_dir",
        "results_dir", "output_dir", "ground_truth",
    }
    for k, v in sess.items():
        if k in path_keys and isinstance(v, str):
            sess[k] = Path(v)
    return sess

def get_paths(sess: Dict[str, Any]) -> Dict[str, Path]:
    """Extract key path objects from session."""
    paths = {
        "project_root":  Path(sess.get("project_root", PROJECT_ROOT)),
        "monai_dir":     Path(sess.get("monai_dir", "")),
        "extra_dir":     Path(sess.get("extra_dir", "")),
        "bundle_dir":    Path(sess.get("bundle_dir", "")),
        "output_dir":    Path(sess.get("output_dir", "")),
        "results_dir":   Path(sess.get("results_dir", "")),
        "ground_truth":  Path(sess.get("ground_truth", "")),
        "mri_dicom_dir": Path(sess.get("mri_dicom_dir", "")),
        "ct_dicom_dir":  Path(sess.get("ct_dicom_dir", "")) if sess.get("ct_dicom_dir") else None,
        "nifti_dir":     Path(sess.get("nifti_dir", "")),
    }
    
    # Session-aware segmentation directory
    seg_session = os.environ.get("PYBRAIN_SEG_SESSION")
    if seg_session:
        results_dir = paths["results_dir"]
        seg_dir = results_dir / seg_session
        if seg_dir.exists():
            paths["seg_dir"] = seg_dir
        else:
            paths["seg_dir"] = paths["output_dir"]
    else:
        paths["seg_dir"] = paths["output_dir"]

    return paths

def get_patient(sess: Dict[str, Any]) -> Dict[str, Any]:
    """Return patient metadata."""
    return sess.get("patient", {})
