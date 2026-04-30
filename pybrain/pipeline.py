# pybrain/pipeline.py
"""
PY-BRAIN v2 — Pipeline Orchestrator
====================================
Single entry point for the full brain-tumour analysis pipeline.

Usage::

    from pybrain.pipeline import run, load_config

    config = load_config()
    result = run(
        assignments={"T1": "/path/to/t1.nii.gz", ...},
        output_dir=Path("results/patient_001"),
    )
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration with fallback defaults.

    Reads from ``pybrain/config/defaults.yaml`` (via ``get_config``)
    and merges with sensible defaults for every field.
    """
    from pybrain.io.config import get_config

    cfg = get_config()

    # Fallback defaults
    cfg.setdefault("thresholds", {"wt": 0.40, "tc": 0.35, "et": 0.35})
    cfg.setdefault("ensemble_weights", {"segresnet": 0.60, "tta4": 0.40})
    cfg.setdefault("ct_boost", {"enabled": False, "boost_factor": 0.15})
    cfg.setdefault("segmentation", {"non_gbm_ct_calc_threshold_cc": 200.0})
    cfg.setdefault("models", {})

    # Device auto-detection: MPS > CUDA > CPU
    try:
        import torch

        if torch.backends.mps.is_available():
            cfg.setdefault("hardware", {})["device"] = "mps"
        elif torch.cuda.is_available():
            cfg.setdefault("hardware", {})["device"] = "cuda"
        else:
            cfg.setdefault("hardware", {})["device"] = "cpu"
    except ImportError:
        cfg.setdefault("hardware", {})["device"] = "cpu"

    return cfg


def run(
    assignments: Dict[str, str],
    output_dir: Path,
    skip_preprocessing: bool = False,
    gt_path: Optional[Path] = None,
    run_location: bool = True,
    run_morphology: bool = True,
    run_radiomics: bool = True,
    run_report: bool = True,
    patient: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the full PY-BRAIN pipeline.

    Parameters
    ----------
    assignments : dict
        Sequence role → NIfTI path or DICOM folder.
    output_dir : Path
        Output directory.
    skip_preprocessing : bool
        If True, assume volumes are already preprocessed in *output_dir*.
    gt_path : Path, optional
        Ground-truth segmentation for Dice evaluation.
    run_location / run_morphology / run_radiomics / run_report : bool
        Stage toggles.
    patient : dict, optional
        Patient metadata (name, age, sex).
    config : dict, optional
        Pipeline configuration (from ``load_config``).

    Returns
    -------
    dict
        Summary with paths to all outputs.
    """
    t_start = time.time()
    config = config or load_config()
    patient = patient or {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir = output_dir  # preprocessed files land here

    result: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "preproc_dir": str(preproc_dir),
        "patient": patient,
        "model_activated": True,
        "non_gbm_suspected": False,
        "volumes": {},
    }

    # ── Preprocessing ─────────────────────────────────────────────────────
    if not skip_preprocessing:
        logger.info("Running preprocessing...")
        from pybrain.core.preprocessing import preprocess_mri

        preprocess_mri(assignments, output_dir)

    # Load preprocessed volumes
    volumes = {}
    ref_img = None
    for seq in ["t1", "t1c", "t2", "flair"]:
        p = preproc_dir / f"{seq}_resampled.nii.gz"
        if not p.exists():
            p = preproc_dir / f"{seq}.nii.gz"
        if p.exists():
            img = nib.load(str(p))
            volumes[seq.upper()] = img.get_fdata().astype(np.float32)
            if ref_img is None:
                ref_img = img

    if not volumes:
        logger.error("No preprocessed volumes found")
        return result

    # Brain mask
    mask_path = preproc_dir / "brain_mask.nii.gz"
    if mask_path.exists():
        brain_mask = nib.load(str(mask_path)).get_fdata().astype(np.float32)
    else:
        from pybrain.core.brainmask import robust_brain_mask

        zooms = ref_img.header.get_zooms()[:3]
        vox_vol_cc = float(np.prod(zooms)) / 1000.0
        t1_path = preproc_dir / "t1_resampled.nii.gz"
        brain_mask = robust_brain_mask(
            volumes,
            vox_vol_cc=vox_vol_cc,
            ref_nifti_path=t1_path if t1_path.exists() else None,
        )

    # ── Segmentation ──────────────────────────────────────────────────────
    logger.info("Running segmentation...")
    from pybrain.core.segmentation import segment, SegmentationConfig

    seg_cfg = SegmentationConfig(
        wt_threshold=config["thresholds"].get("wt", 0.40),
        tc_threshold=config["thresholds"].get("tc", 0.35),
        et_threshold=config["thresholds"].get("et", 0.35),
        ensemble_weights=config.get("ensemble_weights", {}),
        device=config.get("hardware", {}).get("device", "cpu"),
    )

    # CT data
    ct_data = None
    ct_path = preproc_dir / "ct_brain_registered.nii.gz"
    if ct_path.exists():
        ct_data = nib.load(str(ct_path)).get_fdata().astype(np.float32)
        seg_cfg.ct_boost["enabled"] = True

    seg_result = segment(volumes, brain_mask, seg_cfg, ct_data=ct_data)
    result["model_activated"] = seg_result.model_activated
    result["volumes"] = {
        "wt_cc": seg_result.wt_cc,
        "tc_cc": seg_result.tc_cc,
        "et_cc": seg_result.et_cc,
        "nc_cc": seg_result.nc_cc,
    }

    if not seg_result.model_activated:
        logger.warning("Model non-activation — max probability below threshold")

    # Save segmentation
    seg_path = output_dir / "segmentation_full.nii.gz"
    from pybrain.io.nifti_io import save_nifti

    save_nifti(seg_result.seg_full, seg_path, ref_img)

    # ── Quality report ────────────────────────────────────────────────────
    quality = {
        "model_activated": seg_result.model_activated,
        "non_gbm_suspected": seg_result.non_gbm_suspected,
        "volumes_cc": result["volumes"],
        "elapsed": round(time.time() - t_start, 1),
    }

    # CT calcification check
    calc_path = output_dir / "ct_calcification.nii.gz"
    if calc_path.exists():
        try:
            calc_img = nib.load(str(calc_path))
            calc_data = calc_img.get_fdata()
            zooms = calc_img.header.get_zooms()[:3]
            calc_cc = float((calc_data > 0).sum() * np.prod(zooms) / 1000)
            quality["ct_calcification_cc"] = calc_cc
            ct_thresh = config.get("segmentation", {}).get("non_gbm_ct_calc_threshold_cc", 200.0)
            quality["non_gbm_suspected"] = bool(not seg_result.model_activated and calc_cc > ct_thresh)
        except Exception:
            pass

    q_path = output_dir / "segmentation_quality.json"
    with open(q_path, "w") as f:
        json.dump(quality, f, indent=2)

    result["non_gbm_suspected"] = quality.get("non_gbm_suspected", False)
    result["elapsed"] = quality["elapsed"]

    # ── GT validation ─────────────────────────────────────────────────────
    if gt_path and gt_path.exists():
        logger.info("Running ground-truth validation...")
        try:
            from scripts.validate import validate_segmentation

            val = validate_segmentation(seg_path, gt_path)
            result["dice"] = val
        except Exception as exc:
            logger.warning(f"GT validation failed: {exc}")

    # ── Location analysis ─────────────────────────────────────────────────
    if run_location:
        try:
            from pybrain.analysis.location import analyse_location

            loc = analyse_location(seg_result.seg_full, brain_mask, ref_img)
            loc_path = output_dir / "tumour_location.json"
            with open(loc_path, "w") as f:
                json.dump(loc, f, indent=2, default=_to_serializable)
            result["location"] = str(loc_path)
        except Exception as exc:
            logger.warning(f"Location analysis failed: {exc}")

    # ── Morphology ────────────────────────────────────────────────────────
    if run_morphology:
        try:
            from pybrain.analysis.morphology import analyse_morphology

            morph = analyse_morphology(seg_result.seg_full, brain_mask, ref_img)
            morph_path = output_dir / "morphology.json"
            with open(morph_path, "w") as f:
                json.dump(morph, f, indent=2, default=_to_serializable)
            result["morphology"] = str(morph_path)
        except Exception as exc:
            logger.warning(f"Morphology analysis failed: {exc}")

    # ── Radiomics ─────────────────────────────────────────────────────────
    if run_radiomics:
        try:
            from pybrain.analysis.radiomics import extract_radiomics

            rad = extract_radiomics(volumes, seg_result.seg_full, brain_mask)
            rad_path = output_dir / "radiomics_features.json"
            with open(rad_path, "w") as f:
                json.dump(rad, f, indent=2, default=_to_serializable)
            result["radiomics"] = str(rad_path)
        except Exception as exc:
            logger.warning(f"Radiomics extraction failed: {exc}")

    # ── Stage 10: Molecular Status Prediction ────────────────────────────
    try:
        from pybrain.analysis.molecular import predict_molecular_status

        logger.info("Stage 10: Molecular Status Prediction (IDH/MGMT)")

        seg_quality_path = output_dir / "segmentation_quality.json"
        if seg_quality_path.exists():
            with open(seg_quality_path) as f:
                sq = json.load(f)
            volumes_cc = sq.get("volumes_cc", {})
        else:
            volumes_cc = result.get("volumes", {})

        wt = float(volumes_cc.get("wt_cc", 0))
        tc = float(volumes_cc.get("tc_cc", 0))
        et = float(volumes_cc.get("et_cc", 0))
        nc = float(volumes_cc.get("nc_cc", 0))
        ed = max(0, wt - tc)

        mol_volumes = {
            "whole_tumour": wt,
            "enhancing": et,
            "edema": ed,
            "necrotic_core": nc,
        }

        morph_data = None
        morph_path = output_dir / "morphology.json"
        if morph_path.exists():
            with open(morph_path) as f:
                morph_data = json.load(f)

        radio_data = None
        radio_path = output_dir / "radiomics_features.json"
        if radio_path.exists():
            with open(radio_path) as f:
                radio_data = json.load(f)

        molecular_result = predict_molecular_status(
            volumes_cc=mol_volumes,
            morphology=morph_data,
            radiomics=radio_data,
            patient_info=patient,
        )

        mol_path = output_dir / "molecular_prediction.json"
        with open(mol_path, "w") as f:
            json.dump(molecular_result, f, indent=2)

        result["molecular"] = molecular_result

        idh = molecular_result["idh"]
        mgmt = molecular_result["mgmt"]
        logger.info(f"  IDH:  {idh['prediction']} ({idh['probability']:.0%}, {idh['confidence_level']})")
        logger.info(f"  MGMT: {mgmt['prediction']} ({mgmt['probability']:.0%}, {mgmt['confidence_level']})")
    except Exception as exc:
        logger.warning(f"Molecular prediction failed: {exc}")

    gc.collect()
    logger.info(f"Pipeline complete — {result['elapsed']:.1f}s")
    return result


def _to_serializable(obj):
    """JSON serializer for numpy bool_, int_, float_ etc."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
