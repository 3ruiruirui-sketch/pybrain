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
from typing import Any, Dict, Optional, Literal

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    cfg.setdefault("bundle_dir", PROJECT_ROOT / "models" / "brats_bundle")

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
    export_dicom_seg: bool = False,
    source_dicom_dir: Optional[Path] = None,
    export_dicom_sr: bool = False,
    prior_session_dir: Optional[Path] = None,
    analysis_mode: Literal["glioma", "mets", "auto"] = "glioma",
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
    export_dicom_seg : bool
        If True, export segmentation as DICOM-SEG file.
    source_dicom_dir : Path, optional
        Source DICOM directory for DICOM-SEG export (required if export_dicom_seg=True).
    export_dicom_sr : bool
        If True, export measurements as DICOM-SR file.
    prior_session_dir : Path, optional
        Prior session directory for longitudinal comparison (contains T1c and segmentation).
    analysis_mode : str
        Analysis mode: "glioma" for single large tumor, "mets" for multiple small lesions,
        or "auto" to automatically classify based on lesion count.

    Returns
    -------
    dict
        Summary with paths to all outputs. Includes either "glioma_result" or "mets_result"
        depending on analysis_mode.
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

    # Add spacing to config for mets analysis
    zooms = ref_img.header.get_zooms()[:3]
    config["spacing"] = zooms

    # ── Analysis Mode Selection ─────────────────────────────────────────────
    if analysis_mode == "auto":
        logger.info("Auto-detecting analysis mode...")
        from pybrain.analysis.mets_pipeline import classify_analysis_mode

        determined_mode = classify_analysis_mode(
            volumes["T1C"],
            brain_mask,
            config,
        )
        logger.info(f"Auto-detected mode: {determined_mode}")
        analysis_mode = determined_mode

    # ── Segmentation ──────────────────────────────────────────────────────
    if analysis_mode == "mets":
        logger.info("Running mets analysis pipeline...")
        from pybrain.analysis.mets_pipeline import run_mets_analysis, generate_mets_report

        # Check if mets is enabled in config
        mets_config = config.get("mets", {})
        if not mets_config.get("enabled", False):
            logger.warning("Mets analysis requested but not enabled in config. Falling back to glioma.")
            analysis_mode = "glioma"
        else:
            # Run mets analysis
            mets_result = run_mets_analysis(
                t1c=volumes["T1C"],
                t1=volumes["T1"],
                t2=volumes["T2"],
                flair=volumes["FLAIR"],
                brain_mask=brain_mask,
                config=config,
                device=config.get("hardware", {}).get("device", "cpu"),
            )

            # Save mets segmentation
            mets_seg_path = output_dir / "segmentation_full.nii.gz"
            from pybrain.io.nifti_io import save_nifti
            save_nifti(mets_result.combined_segmentation, mets_seg_path, ref_img)

            # Save mets report
            mets_report = generate_mets_report(mets_result)
            mets_report_path = output_dir / "mets_report.json"
            with open(mets_report_path, "w") as f:
                json.dump(mets_report, f, indent=2)

            result["mets_result"] = {
                "total_lesion_count": mets_result.total_lesion_count,
                "total_lesion_volume_cc": mets_result.total_lesion_volume_cc,
                "detection_method": mets_result.detection_method,
                "segmentation_method": mets_result.segmentation_method,
                "report_path": str(mets_report_path),
            }
            result["analysis_mode"] = "mets"
            result["volumes"] = {
                "wt_cc": mets_result.total_lesion_volume_cc,  # All lesions as WT
                "tc_cc": 0.0,
                "et_cc": 0.0,
                "nc_cc": 0.0,
            }

            # Skip downstream glioma-specific analysis for mets
            logger.info("Mets analysis complete. Skipping glioma-specific analysis.")
            return result

    # Glioma analysis (default)
    logger.info("Running glioma segmentation...")
    from pybrain.core.segmentation import segment, SegmentationConfig

    seg_cfg = SegmentationConfig(
        wt_threshold=config["thresholds"].get("wt", 0.40),
        tc_threshold=config["thresholds"].get("tc", 0.35),
        et_threshold=config["thresholds"].get("et", 0.35),
        ensemble_weights=config.get("ensemble_weights", {}),
        device=config.get("hardware", {}).get("device", "cpu"),
        bundle_dir=Path(config.get("bundle_dir", PROJECT_ROOT / "models" / "brats_bundle")),
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

    # ── DICOM-SEG Export ───────────────────────────────────────────────────
    if export_dicom_seg:
        if source_dicom_dir is None:
            logger.warning("export_dicom_seg=True but source_dicom_dir not provided — skipping DICOM-SEG export")
        else:
            try:
                from pybrain.io.dicom_seg_writer import write_dicom_seg

                dicom_seg_path = output_dir / "segmentation.dcm"
                dicom_cfg = config.get("output", {}).get("dicom_seg", {})
                
                write_dicom_seg(
                    segmentation=seg_result.seg_full,
                    source_dicom_dir=Path(source_dicom_dir),
                    output_path=dicom_seg_path,
                    series_description=dicom_cfg.get(
                        "series_description",
                        "PY-BRAIN BraTS Segmentation (Research Only)",
                    ),
                    algorithm_name=dicom_cfg.get("algorithm_name", "PY-BRAIN v2"),
                    include_disclaimer=dicom_cfg.get("include_disclaimer", True),
                )
                result["dicom_seg_path"] = str(dicom_seg_path)
                logger.info(f"DICOM-SEG exported to {dicom_seg_path}")
            except Exception as exc:
                logger.warning(f"DICOM-SEG export failed: {exc}")
                result["dicom_seg_path"] = None

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

    # ── DICOM-SR Export (Final Stage) ────────────────────────────────────────
    if export_dicom_sr:
        if source_dicom_dir is None:
            logger.warning("export_dicom_sr=True but source_dicom_dir not provided — skipping DICOM-SR export")
        else:
            try:
                from pybrain.io.dicom_sr_writer import write_measurement_report

                dicom_sr_path = output_dir / "measurements.dcm"
                
                # Extract measurements from result
                measurements_dict = {}
                if "volumes" in result:
                    vol = result["volumes"]
                    measurements_dict["wt_volume_cc"] = vol.get("wt_cc", 0.0)
                    measurements_dict["tc_volume_cc"] = vol.get("tc_cc", 0.0)
                    measurements_dict["et_volume_cc"] = vol.get("et_cc", 0.0)
                    measurements_dict["nc_volume_cc"] = vol.get("nc_cc", 0.0)
                
                # Add uncertainty if available
                if "uncertainty_mean" in result:
                    measurements_dict["uncertainty_mean"] = result["uncertainty_mean"]
                
                # Write DICOM-SR (requires DICOM-SEG file)
                dicom_seg_path = output_dir / "segmentation.dcm"
                if not dicom_seg_path.exists():
                    logger.warning("DICOM-SEG file not found — DICOM-SR requires DICOM-SEG reference")
                    result["dicom_sr_path"] = None
                else:
                    write_measurement_report(
                        measurements=measurements_dict,
                        source_dicom_dir=Path(source_dicom_dir),
                        segmentation_dicom_path=dicom_seg_path,
                        output_path=dicom_sr_path,
                    )
                    result["dicom_sr_path"] = str(dicom_sr_path)
                    logger.info(f"DICOM-SR exported to {dicom_sr_path}")
            except Exception as exc:
                logger.warning(f"DICOM-SR export failed: {exc}")
                result["dicom_sr_path"] = None

    # ── Longitudinal Analysis (Final Stage) ────────────────────────────────────
    if prior_session_dir is not None:
        try:
            from pybrain.analysis.longitudinal import compare_timepoints

            longitudinal_output_dir = output_dir / "longitudinal"
            
            # Find required files in current session
            current_t1c = output_dir / "T1c.nii.gz"
            current_seg = output_dir / "segmentation_full.nii.gz"
            
            # Find required files in prior session
            prior_t1c = Path(prior_session_dir) / "T1c.nii.gz"
            prior_seg = Path(prior_session_dir) / "segmentation_full.nii.gz"
            
            # Validate files exist
            if not current_t1c.exists():
                logger.warning("Current T1c not found — skipping longitudinal analysis")
            elif not current_seg.exists():
                logger.warning("Current segmentation not found — skipping longitudinal analysis")
            elif not prior_t1c.exists():
                logger.warning("Prior T1c not found — skipping longitudinal analysis")
            elif not prior_seg.exists():
                logger.warning("Prior segmentation not found — skipping longitudinal analysis")
            else:
                longitudinal_config = config.get("longitudinal", {})
                longitudinal_result = compare_timepoints(
                    current_t1c=current_t1c,
                    current_seg=current_seg,
                    prior_t1c=prior_t1c,
                    prior_seg=prior_seg,
                    output_dir=longitudinal_output_dir,
                    config=longitudinal_config,
                )
                
                # Store result as dict for JSON serialization
                result["longitudinal"] = {
                    "registration_quality": longitudinal_result.registration_quality,
                    "rano_response": longitudinal_result.rano_response,
                    "volume_changes": {
                        region: {
                            "prior_cc": change.prior_cc,
                            "current_cc": change.current_cc,
                            "abs_change_cc": change.abs_change_cc,
                            "pct_change": change.pct_change,
                            "status": change.status,
                        }
                        for region, change in longitudinal_result.volume_changes.items()
                    },
                    "registered_prior_path": str(longitudinal_result.registered_prior_path),
                    "prior_seg_in_current_space_path": str(longitudinal_result.prior_seg_in_current_space_path),
                    "overlay_paths": {
                        orientation: str(path) for orientation, path in longitudinal_result.overlay_paths.items()
                    },
                }
                logger.info(f"Longitudinal analysis complete: {longitudinal_result.rano_response}")
        except Exception as exc:
            logger.warning(f"Longitudinal analysis failed: {exc}")
            result["longitudinal"] = None

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
