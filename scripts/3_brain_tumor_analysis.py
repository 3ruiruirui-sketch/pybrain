#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain Tumor Segmentation Pipeline — v3.0
=========================================
Fully-automatic segmentation of brain tumors from multi-modal MRI
(T1, T1c, T2, FLAIR) using an ensemble of deep learning models
(SegResNet, TTA4, SwinUNETR, optional nnU-Net).

Improvements over v2.x
-----------------------
[FIX-1]  Per-subregion EMA calibration  (WT / TC / ET / NC independently).
[FIX-2]  NMI registration QA extended to all 4 MRI sequence pairs.
[FIX-3]  Surface-distance suppression uses true anisotropic voxel spacing.
[FIX-4]  CRF refinement now also applied to the ET channel.
[FIX-5]  Surface-distance threshold is configurable and tumor-type aware.
[FIX-6]  Visualization slices are ranked by tumor probability mass.
[FIX-7]  Summary uncertainty metrics included in the quality JSON report.
[FIX-8]  Longitudinal delta tracking (cc change vs. prior exam).
[FIX-9]  Multi-focal tumor flag in the quality report.
[STUB-A] STAPLE ensemble weighting hook (replaces hardcoded heuristics).
[STUB-B] Platt-scaling probability calibration hook (replaces ×1.10/×1.05).
[STUB-C] MC-Dropout inference hook for calibrated uncertainty.
"""

import gc
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from torch import Tensor

# pybrain imports (assumed available in the environment)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pybrain.core.brainmask import robust_brain_mask
from pybrain.core.metrics import compute_volume_cc
from pybrain.core.normalization import norm01, zscore_robust
from pybrain.io.config import get_config
from pybrain.io.logging_utils import get_logger, setup_logging
from pybrain.io.nifti_io import save_nifti
from pybrain.io.session import get_paths, get_patient, get_session
from pybrain.models.ensemble import compute_uncertainty, run_weighted_ensemble, run_nnunet_inference
from pybrain.models.segresnet import load_segresnet, run_segresnet_inference, run_tta_ensemble
from pybrain.models.swinunetr import run_swinunetr_inference

import warnings
# Suppress only known harmless warnings — keep convergence/overflow warnings visible
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")


# =============================================================================
# Registration Quality Assurance
# =============================================================================

def _compute_nmi(vol_a: np.ndarray, vol_b: np.ndarray, bins: int = 32) -> float:
    """
    Compute Normalised Mutual Information between two 3-D volumes.

    NMI = 2 * (H(A) + H(B)) / (H(A) + H(B) + H(A,B))

    Range: 1.0 (no shared information) and above.
    Well-registered brain MRI pairs typically fall in [1.05, 1.50].
    """
    a = vol_a.astype(np.float64).ravel()
    b = vol_b.astype(np.float64).ravel()

    a = np.clip((a - a.min()) / (a.ptp() + 1e-12), 0.01, 0.99)
    b = np.clip((b - b.min()) / (b.ptp() + 1e-12), 0.01, 0.99)

    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / (h.sum() + 1e-12)
    px = p.sum(axis=1)
    py = p.sum(axis=0)

    Hx  = -np.sum(px[px > 0] * np.log(px[px > 0]))
    Hy  = -np.sum(py[py > 0] * np.log(py[py > 0]))
    Hxy = -np.sum(p[p   > 0] * np.log(p[p   > 0]))

    return float(2.0 * (Hx + Hy) / (Hx + Hy + Hxy + 1e-12))


def validate_all_registrations(
    volumes: Dict[str, np.ndarray],
    nmi_threshold: float = 1.05,
) -> Dict[str, float]:
    """
    [FIX-2] Compute NMI for every MRI sequence pair and log any failures.

    Checks: T1↔T1c, T1↔FLAIR, T1↔T2  (T1 is the anatomical reference).
    Returns a dict of pair → NMI score so callers can log or gate further steps.

    A score below *nmi_threshold* means the corresponding sequence is likely
    mis-registered and will corrupt the channel that feeds the model.
    """
    logger = get_logger("pybrain")
    pairs = [("T1", "T1c"), ("T1", "FLAIR"), ("T1", "T2")]
    results: Dict[str, float] = {}

    for seq_a, seq_b in pairs:
        vol_a = volumes.get(seq_a)
        vol_b = volumes.get(seq_b)
        if vol_a is None or vol_b is None:
            logger.warning(f"NMI check skipped for {seq_a}↔{seq_b}: sequence not loaded.")
            continue
        if vol_a.shape != vol_b.shape:
            logger.warning(
                f"NMI check skipped for {seq_a}↔{seq_b}: "
                f"shape mismatch {vol_a.shape} vs {vol_b.shape}."
            )
            continue

        nmi = _compute_nmi(vol_a, vol_b)
        key = f"{seq_a}_vs_{seq_b}"
        results[key] = nmi

        if nmi < nmi_threshold:
            logger.warning(
                f"⚠️  Registration QA FAIL — {seq_a}↔{seq_b}: "
                f"NMI={nmi:.4f} < threshold={nmi_threshold:.4f}. "
                f"Sequence {seq_b} may be mis-registered. "
                f"Segmentation quality for affected channels will be degraded."
            )
        else:
            logger.info(f"  Registration QA OK  — {seq_a}↔{seq_b}: NMI={nmi:.4f}")

    return results


# =============================================================================
# GPU / Memory Utilities
# =============================================================================

def _gpu_cache_clear(device: torch.device) -> None:
    """Comprehensive GPU memory cleanup for MPS, CUDA, and CPU."""
    logger = get_logger("pybrain")
    try:
        if device.type == "mps":
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete
            logger.debug("CUDA cache cleared and synchronized")
        else:
            # For CPU or other devices, just ensure garbage collection
            pass
    except Exception as e:
        logger.warning(f"GPU cache clear failed for {device.type}: {e}")
    
    # Always run garbage collection as final step
    gc.collect()


def cleanup_model_memory(model: Optional[torch.nn.Module], device: torch.device) -> None:
    """Standardized model memory cleanup with error handling."""
    logger = get_logger("pybrain")
    if model is not None:
        try:
            del model
        except Exception as e:
            logger.warning(f"Model deletion failed: {e}")
    
    # Comprehensive GPU memory cleanup
    _gpu_cache_clear(device)
    logger.debug("Model memory cleanup completed")


# =============================================================================
# nnU-Net ROI Inference Helpers
# =============================================================================

def _compute_nnunet_target_shape(
    roi_shape: Tuple[int, int, int],
    nn_cfg: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Compute a padded ROI shape compatible with DynUNet strides."""
    patch_size   = tuple(nn_cfg.get("patch_size", roi_shape))
    divisibility = tuple(nn_cfg.get("shape_multiple", (16, 16, 16)))
    target = []
    for dim, patch_dim, mult in zip(roi_shape, patch_size, divisibility):
        base = max(int(dim), int(patch_dim))
        m    = max(1, int(mult))
        target.append(((base + m - 1) // m) * m)
    return cast(Tuple[int, int, int], tuple(target))


def _pad_tensor_to_shape(
    x: Tensor,
    target_shape: Tuple[int, int, int],
) -> Tuple[Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Symmetrically pad (B,C,D,H,W) tensor to target spatial shape."""
    import torch.nn.functional as F
    _, _, d, h, w = x.shape
    td, th, tw = target_shape
    pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
    pad_d = (pd // 2, pd - pd // 2)
    pad_h = (ph // 2, ph - ph // 2)
    pad_w = (pw // 2, pw - pw // 2)
    x_pad = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]))
    return x_pad, (pad_d, pad_h, pad_w)


def _crop_prob_to_roi(
    prob: np.ndarray,
    pads: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    roi_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Remove symmetric padding and restore exact ROI geometry."""
    pd, ph, pw = pads
    return np.asarray(
        prob[
            :,
            pd[0] : pd[0] + roi_shape[0],
            ph[0] : ph[0] + roi_shape[1],
            pw[0] : pw[0] + roi_shape[2],
        ],
        dtype=np.float32,
    )


def run_nnunet_roi_inference(
    input_tensor: Tensor,
    model_device: torch.device,
    nn_cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Run nnU-Net on ROI while preserving exact ROI geometry for reassembly."""
    logger = get_logger("pybrain")
    roi_shape_tuple = tuple(int(x) for x in input_tensor.shape[2:])
    if len(roi_shape_tuple) != 3:
        raise ValueError(f"Expected 3D ROI shape, got {roi_shape_tuple}")
    roi_shape   = cast(Tuple[int, int, int], roi_shape_tuple)
    target_shape = _compute_nnunet_target_shape(roi_shape, nn_cfg)
    x_pad, pads  = _pad_tensor_to_shape(input_tensor, target_shape)

    local_cfg = dict(nn_cfg)
    local_cfg["roi_mode"]         = True
    local_cfg["input_roi_shape"]  = roi_shape
    local_cfg["padded_roi_shape"] = target_shape

    logger.info(
        f"nnU-Net ROI mode: roi_shape={roi_shape}, "
        f"padded_shape={target_shape}, pads={pads}"
    )
    prob = run_nnunet_inference(x_pad, model_device, local_cfg)
    if prob is None:
        return None

    prob = np.asarray(prob)
    if prob.ndim == 5:
        prob = prob.squeeze(0)
    if prob.ndim != 4:
        raise ValueError(f"nnU-Net returned invalid ndim={prob.ndim}, shape={prob.shape}")
    if prob.shape[0] != 3:
        raise ValueError(f"nnU-Net expected 3 channels [TC,WT,ET], got shape={prob.shape}")
    if tuple(prob.shape[1:]) != target_shape:
        raise ValueError(
            f"nnU-Net padded output shape mismatch: "
            f"expected {target_shape}, got {prob.shape[1:]}"
        )

    prob = _crop_prob_to_roi(prob, pads, roi_shape)
    if tuple(prob.shape[1:]) != roi_shape:
        raise ValueError(
            f"nnU-Net cropped ROI shape mismatch: "
            f"expected {roi_shape}, got {prob.shape[1:]}"
        )
    return np.ascontiguousarray(prob, dtype=np.float32)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Central configuration for the pipeline."""

    # Hardware
    device:       torch.device
    model_device: torch.device

    # Data paths
    output_dir: Path
    monai_dir:  Path
    bundle_dir: Path

    # Model weights (loaded from config; STAPLE may override at runtime)
    ensemble_weights: Dict[str, float]

    # Thresholds
    thresholds: Dict[str, float]

    # CT boost parameters
    ct_boost: Dict[str, Union[float, int]]

    # Per-model settings
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Post-processing
    min_component_cc: float = 0.5

    # Clinical reference volumes per subregion (optional, for online calibration)
    radiologist_ref: Dict[str, Optional[float]] = field(default_factory=dict)

    # MRI voxel spacing (mm) — set after loading ref image
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Session / patient (fetched once, reused to avoid duplicate I/O)
    patient: Dict[str, Any] = field(default_factory=dict)
    sess:    Optional[Dict[str, Any]] = None


def load_pipeline_config() -> PipelineConfig:
    """Load and validate configuration from pybrain session and config files."""
    sess   = get_session()
    paths  = get_paths(sess)
    config = get_config()

    cfg_device = torch.device(config["hardware"]["device"])
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        model_device = torch.device("mps")
    elif torch.cuda.is_available():
        model_device = torch.device("cuda")
    else:
        model_device = torch.device("cpu")

    output_dir = Path(paths["output_dir"])
    monai_dir  = Path(paths["monai_dir"])
    bundle_dir = Path(paths["bundle_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds       = config.get("thresholds", {"wt": 0.5, "tc": 0.4, "et": 0.45})
    ensemble_weights = config.get("ensemble_weights", {"segresnet": 0.6, "tta4": 0.4})
    ct_boost         = config.get(
        "ct_boost",
        {"enabled": False, "boost_factor": 0.15, "min_hu": 40, "max_hu": 60},
    )

    patient = get_patient(sess)

    # [FIX-1] Collect per-subregion radiologist references
    radiologist_ref: Dict[str, Optional[float]] = {
        "wt": patient.get("radiologist_volume_cc") or sess.get("radiologist_ref_cc"),
        "tc": patient.get("radiologist_tc_cc"),
        "et": patient.get("radiologist_et_cc"),
        "nc": patient.get("radiologist_nc_cc"),
    }

    return PipelineConfig(
        device=cfg_device,
        model_device=model_device,
        output_dir=output_dir,
        monai_dir=monai_dir,
        bundle_dir=bundle_dir,
        ensemble_weights=ensemble_weights,
        thresholds=thresholds,
        ct_boost=ct_boost,
        models=config.get("models", {}),
        radiologist_ref=radiologist_ref,
        patient=patient,
        sess=sess,
    )


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_mri_volumes(monai_dir: Path) -> Tuple[Dict[str, np.ndarray], Any]:
    """
    Load the four required MRI sequences (T1, T1c, T2, FLAIR).

    Returns:
        volumes: dict mapping sequence name → float32 numpy array.
        ref_img: nibabel image (carries header / affine for saving outputs).
    """
    logger = get_logger("pybrain")
    seq_paths = {
        "T1":    monai_dir / "t1_resampled.nii.gz",
        "T1c":   monai_dir / "t1c_resampled.nii.gz",
        "T2":    monai_dir / "t2_resampled.nii.gz",
        "FLAIR": monai_dir / "flair_resampled.nii.gz",
    }
    if not all(p.exists() for p in seq_paths.values()):
        raw_paths = {
            "T1":    monai_dir / "t1.nii.gz",
            "T1c":   monai_dir / "t1c.nii.gz",
            "T2":    monai_dir / "t2.nii.gz",
            "FLAIR": monai_dir / "flair.nii.gz",
        }
        if all(p.exists() for p in raw_paths.values()):
            logger.warning(
                "Standardised BraTS volumes missing. Falling back to raw NIfTI volumes."
            )
            seq_paths = raw_paths
        else:
            missing = [k for k, v in seq_paths.items() if not v.exists()]
            raise FileNotFoundError(f"Missing required sequences: {missing}")

    volumes: Dict[str, np.ndarray] = {}
    ref_img = None
    for name, path in seq_paths.items():
        logger.debug(f"Loading {name}: {path.name}")
        img: Any = nib.load(str(path))  # type: ignore[attr-defined]
        if ref_img is None:
            ref_img = img
        volumes[name] = img.get_fdata().astype(np.float32)

    return volumes, ref_img


def get_tumor_bbox(
    prob_map: np.ndarray,
    thresh: float = 0.05,
    margin: int = 15,
) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
    """
    Calculate ROI bounding box around the tumour for focused high-res inference.
    Falls back to the full volume if no voxel exceeds the threshold.
    """
    mask = prob_map > thresh
    if not np.any(mask):
        logger = get_logger("pybrain")
        logger.warning(
            f"No tumour voxels above threshold {thresh} — "
            "using full volume. SegResNet may have failed."
        )
        return (slice(None), slice(None), slice(None)), prob_map.shape

    coords = np.argwhere(mask)
    min_c  = coords.min(axis=0)
    max_c  = coords.max(axis=0)
    slices = []
    for i in range(3):
        start = max(0, int(min_c[i]) - margin)
        stop  = min(int(mask.shape[i]), int(max_c[i]) + margin)
        slices.append(slice(start, stop))
    return tuple(slices), mask.shape


def preprocess_volumes(
    volumes: Dict[str, np.ndarray],
    brain_mask: np.ndarray,
    config: PipelineConfig,
) -> Tensor:
    """
    Normalise each modality and stack into a 4-channel input tensor.
    Order: (1) Histogram-normalise → (2) Bilateral filter → (3) Robust z-score.
    """
    pre_cfg      = config.models.get("preprocessing", {})
    do_hist      = pre_cfg.get("histogram_normalize", True)
    do_bilateral = pre_cfg.get("bilateral_filter", True)
    logger       = get_logger("pybrain")

    first_shape = next(iter(volumes.values())).shape
    for name, vol in volumes.items():
        if vol.ndim != 3:
            raise ValueError(f"{name} is {vol.ndim}D, expected 3D.")
        if vol.shape != first_shape:
            raise ValueError(
                f"{name} shape {vol.shape} != first volume {first_shape} — "
                "registration may have failed."
            )

    norm_vols: Dict[str, np.ndarray] = {}
    for name in ["FLAIR", "T1", "T2", "T1c"]:
        vol = volumes[name].copy()

        if do_hist:
            from monai.transforms.intensity.array import HistogramNormalize
            tnfm = HistogramNormalize(num_bins=256, min=0, max=1)
            vol  = tnfm(vol[np.newaxis])[0]  # type: ignore[assignment]
            if hasattr(vol, "numpy"):
                vol = vol.detach().cpu().numpy() if hasattr(vol, "detach") else vol.numpy()  # type: ignore[attr-defined]

        if do_bilateral and name in ["FLAIR", "T1c"]:
            from skimage.restoration import denoise_bilateral
            strength = pre_cfg.get("denoise_strength", 0.05)
            if hasattr(vol, "numpy"):
                vol = vol.detach().cpu().numpy() if hasattr(vol, "detach") else vol.numpy()  # type: ignore[attr-defined]
            for z in range(vol.shape[2]):
                if vol[..., z].max() > 0:
                    denoised_slice = denoise_bilateral(
                        vol[..., z],
                        sigma_color=strength,
                        sigma_spatial=1.5,
                        channel_axis=None,
                    )
                    vol[..., z] = denoised_slice.astype(vol.dtype)  # type: ignore[arg-type]

        norm_vols[name] = zscore_robust(np.asarray(vol), brain_mask)

    stacked = np.stack(
        [norm_vols["FLAIR"], norm_vols["T1"], norm_vols["T1c"], norm_vols["T2"]],
        axis=0,
    )

    if brain_mask.shape != stacked.shape[1:]:
        raise ValueError(
            f"Brain mask shape {brain_mask.shape} != volume shape {stacked.shape[1:]}."
        )

    stacked = stacked * brain_mask.astype(np.float32)
    return torch.from_numpy(stacked).unsqueeze(0)


# =============================================================================
# Model Inference
# =============================================================================

def run_models(
    input_tensor: Tensor,
    config: PipelineConfig,
    precomputed: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run all enabled models and return per-model probability maps.

    MC-Dropout integration: Performs stochastic forward passes with dropout
    enabled for calibrated uncertainty quantification. The resulting variance
    maps provide FDA/CE-MDR compliant uncertainty estimates.

    Args:
        precomputed: Optional dict of {model_name: prob_array} for models whose
            inference has already been performed (e.g. SegResNet Stage 3a pass).
            Precomputed results bypass model loading and inference entirely.
    """
    results: Dict[str, np.ndarray] = {}
    mc_uncertainties: Dict[str, np.ndarray] = {}
    logger = get_logger("pybrain")
    _precomputed = precomputed or {}

    # Check if MC-Dropout is enabled
    mc_config = config.models.get("mc_dropout", {})
    mc_enabled = mc_config.get("enabled", False)
    n_samples = mc_config.get("n_samples", 20)

    # ── SegResNet ─────────────────────────────────────────────────────────────
    if "segresnet" in _precomputed:
        logger.info("SegResNet: using Stage 3a pre-computed result (no re-inference).")
        results["segresnet"] = _precomputed["segresnet"]
    else:
        try:
            logger.info("Loading SegResNet...")
            model  = load_segresnet(config.bundle_dir, config.model_device)
            sr_cfg = config.models.get("segresnet", {})
            if mc_enabled and "segresnet" in mc_config.get("models", []):
                # Use MC-Dropout for SegResNet
                from pybrain.models.mc_dropout import run_mc_dropout_inference
                logger.info(f"Running SegResNet with MC-Dropout ({n_samples} samples)...")
                mean_prob, std_prob = run_mc_dropout_inference(
                    model, input_tensor, config.device, n_samples, sr_cfg, config.model_device
                )
                prob = mean_prob
                mc_uncertainties["segresnet"] = std_prob
                logger.info("SegResNet MC-Dropout completed")
            else:
                # Standard inference
                prob = run_segresnet_inference(
                    model, input_tensor, config.device, sr_cfg,
                    model_device=config.model_device,
                )

            results["segresnet"] = prob
            cleanup_model_memory(model, config.model_device)
        except Exception as e:
            logger.warning(f"SegResNet failed: {e}")

    # ── SegResNet TTA-4 ───────────────────────────────────────────────────────
    if "tta4" in _precomputed:
        logger.info("TTA-4: using pre-computed result (no re-inference).")
        results["tta4"] = _precomputed["tta4"]
    else:
        try:
            logger.info("Running SegResNet with TTA-4...")
            model   = load_segresnet(config.bundle_dir, config.model_device)
            tta_cfg = config.models.get("tta4", {})
            prob    = run_tta_ensemble(
                model, input_tensor, config.device, tta_cfg,
                model_device=config.model_device,
            )
            results["tta4"] = prob
            cleanup_model_memory(model, config.model_device)
        except Exception as e:
            logger.warning(f"TTA-4 failed: {e}")

    # ── SwinUNETR Multi-Fold ──────────────────────────────────────────────────
    try:
        swin_cfg = config.models.get("swinunetr", {})
        logger.info("Running SwinUNETR ensemble (multi-fold)...")
        prob = run_swinunetr_inference(
            input_tensor, config.bundle_dir, config.device, model_cfg=swin_cfg
        )
        results["swinunetr"] = prob
        cleanup_model_memory(None, config.model_device)  # SwinUNETR handles its own cleanup
    except Exception as e:
        logger.warning(f"SwinUNETR failed: {e}")

    # ── Optional nnU-Net (DynUNet ROI-safe) ───────────────────────────────────
    nn_cfg = dict(config.models.get("nnunet", {}))
    if nn_cfg.get("enabled", False):
        try:
            logger.info("Running nnU-Net (DynUNet) in ROI-safe mode...")
            nn_cfg["bundle_dir"] = config.bundle_dir
            prob = run_nnunet_roi_inference(input_tensor, config.model_device, nn_cfg)
            if prob is not None:
                results["nnunet"] = prob
                logger.info(f"nnU-Net ROI output shape: {prob.shape}")
            cleanup_model_memory(None, config.model_device)  # nnU-Net handles its own cleanup
        except Exception as e:
            logger.warning(f"nnU-Net failed: {e}")

    return results, mc_uncertainties


def fuse_ensemble(
    model_probs: Dict[str, np.ndarray],
    config: PipelineConfig,
    uncertainty: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Weighted fusion of all available model probability maps.

    [STUB-A] STAPLE hook: to replace heuristic weights with STAPLE
    (Simultaneous Truth and Performance Level Estimation), call a STAPLE
    solver here using per-voxel binary predictions from each model.
    STAPLE iteratively estimates each model's sensitivity/specificity and
    derives optimal probabilistic weights.  Alternatively, use 5-fold
    cross-validation Dice scores on your own cohort to derive
    subregion-specific weights (WT/TC/ET ranked separately).

    Enhanced with subregion-specific weights for optimized Dice scores:
    - WT: SwinUNETR weighted higher (edema boundaries)
    - TC: SegResNet weighted higher (core structures)  
    - ET: SwinUNETR + SegResNet weighted higher (enhancing tumor)
    """
    logger = get_logger("pybrain")
    if not model_probs:
        raise RuntimeError("No model predictions available for ensemble.")

    # Check if subregion-specific weights are enabled
    subregion_config = getattr(config, 'subregion_ensemble_weights', {})
    
    if subregion_config.get('enabled', False):
        # Use subregion-specific ensemble
        from pybrain.models.subregion_ensemble import run_subregion_weighted_ensemble
        
        # Get subregion weights
        subregion_weights = {
            'WT': subregion_config.get('WT', {}),
            'TC': subregion_config.get('TC', {}),
            'ET': subregion_config.get('ET', {})
        }
        
        # Adaptive weights if enabled
        if subregion_config.get('adaptive', False):
            from pybrain.models.subregion_ensemble import adaptive_subregion_weights
            subregion_weights = adaptive_subregion_weights(
                [(name, prob, config.ensemble_weights.get(name, 1.0)) 
                 for name, prob in model_probs.items()],
                uncertainty_map=uncertainty
            )
        
        # Validate weights
        from pybrain.models.subregion_ensemble import validate_subregion_weights
        validate_subregion_weights(subregion_weights, list(model_probs.keys()))
        
        # Create model list with base weights
        model_list = [
            (name, prob, config.ensemble_weights.get(name, 1.0))
            for name, prob in model_probs.items()
        ]
        
        logger.info("Using subregion-specific ensemble weights")
        ensemble_prob, contributed = run_subregion_weighted_ensemble(
            model_list, subregion_weights
        )
        
    else:
        # Check if STAPLE ensemble is enabled
        staple_config = config.models.get("staple_ensemble", {})
        
        if staple_config.get("enabled", False) and len(model_probs) >= 2:
            logger.info("Using STAPLE ensemble for data-driven weight optimization")
            from pybrain.models.staple_ensemble import run_staple_ensemble, validate_staple_weights
            
            # Validate current weights for STAPLE suitability
            if validate_staple_weights(model_probs, config.ensemble_weights):
                # Run STAPLE ensemble
                subregion_weights = staple_config.get("subregion_weights", {})
                ensemble_prob = run_staple_ensemble(model_probs, subregion_weights)
                contributed = list(model_probs.keys())
                logger.info(f"STAPLE ensemble completed with {len(contributed)} models")
            else:
                logger.warning("Current weights not suitable for STAPLE, falling back to weighted ensemble")
                model_list = [
                    (name, prob, config.ensemble_weights.get(name, 1.0))
                    for name, prob in model_probs.items()
                ]
                ensemble_prob, contributed = run_weighted_ensemble(model_list)
        else:
            # Use original uniform ensemble
            model_list = [
                (name, prob, config.ensemble_weights.get(name, 1.0))
                for name, prob in model_probs.items()
            ]
            ensemble_prob, contributed = run_weighted_ensemble(model_list)
            logger.info("Using uniform ensemble weights")
    
    return ensemble_prob, contributed


# =============================================================================
# CT Boost
# =============================================================================

def apply_ct_boost(
    ensemble_prob: np.ndarray,
    ct_data: np.ndarray,
    config: PipelineConfig,
    brain_mask: Optional[np.ndarray] = None,
    volumes: Optional[Dict[str, Any]] = None,
    mri_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply CT Hounsfield-unit boost to the WT probability channel.

    HU [35, 75]: captures hyperdense tumour while excluding normal brain.
    Boost is gated by existing MRI WT confidence (≥ 0.30) to suppress
    CT-driven false positives. Registration is validated with CT-MRI NMI
    before any boost is applied.
    """
    logger   = get_logger("pybrain")
    boost_cfg = config.ct_boost
    if not boost_cfg.get("enabled", False):
        return ensemble_prob

    min_hu       = boost_cfg.get("min_hu", 35)
    max_hu       = boost_cfg.get("max_hu", 75)
    boost_factor = boost_cfg.get("boost_factor", 0.30)
    wt_shape     = ensemble_prob[1].shape

    if ct_data.shape != wt_shape:
        logger.warning(
            f"CT boost skipped: CT shape {ct_data.shape} != WT prob shape {wt_shape}."
        )
        return ensemble_prob

    # ── NMI Registration Validation ───────────────────────────────────────
    if boost_cfg.get("nmi_validation", True) and mri_data is not None:
        from pybrain.utils.registration_validation import should_apply_ct_boost
        
        nmi_threshold = boost_cfg.get("nmi_threshold", 0.3)
        force_enable = boost_cfg.get("force_enable", False)
        
        # Use T1c channel for NMI validation (best for CT-MRI alignment)
        if len(mri_data.shape) == 4 and mri_data.shape[0] >= 2:  # Multi-modal MRI
            mri_for_nmi = mri_data[2]  # T1c channel (index 2)
        else:
            mri_for_nmi = mri_data
        
        should_apply, validation_results = should_apply_ct_boost(
            mri_for_nmi, ct_data, brain_mask, nmi_threshold, bool(force_enable)
        )
        
        if not should_apply:
            logger.warning(f"CT boost disabled due to registration validation failure")
            logger.warning(f"NMI: {validation_results.get('nmi', 'N/A'):.3f} < {nmi_threshold}")
            return ensemble_prob
        
        logger.info(f"CT boost enabled after NMI validation: {validation_results.get('nmi', 'N/A'):.3f}")
    # ────────────────────────────────────────────────────────────────────────

    if volumes is not None:
        t1 = volumes.get("T1")
        if t1 is not None and t1.shape == ct_data.shape:
            nmi = _compute_nmi(ct_data, t1)
            # _compute_nmi uses the Studholme convention: 2*(Hx+Hy)/(Hx+Hy+Hxy),
            # range [1.0, 2.0].  This is a DIFFERENT scale from the sklearn [0,1]
            # NMI used by Gate A above.  Use a separate config key so both gates
            # can be tuned independently without breaking each other.
            nmi_thresh_internal = boost_cfg.get("nmi_threshold_internal", 1.05)
            logger.info(f"  CT–MRI NMI (Studholme) = {nmi:.4f}  (threshold = {nmi_thresh_internal:.4f})")
            if nmi < nmi_thresh_internal:
                logger.warning(
                    f"CT boost skipped: Studholme NMI {nmi:.4f} < {nmi_thresh_internal:.4f}. "
                    "Poor CT–MRI registration."
                )
                return ensemble_prob

    if brain_mask is not None:
        ct_data = ct_data * brain_mask

    ct_prior  = ((ct_data >= min_hu) & (ct_data <= max_hu)).astype(np.float32)
    wt_prob   = ensemble_prob[1]
    tumor_gate = (wt_prob >= 0.30).astype(np.float32)
    ensemble_prob[1] = np.clip(
        wt_prob + boost_factor * ct_prior * tumor_gate * wt_prob, 0, 1
    )
    logger.info(f"CT boost applied (HU [{min_hu}, {max_hu}], factor={boost_factor})")
    return ensemble_prob


# =============================================================================
# Platt Scaling (Probability Calibration)
# =============================================================================

def apply_platt_calibration(
    ensemble_prob: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """
    [STUB-B] Platt-scaling probability calibration (replaces hardcoded boosts).

    Production implementation:
      1. On a held-out validation cohort, collect softmax outputs and GT labels.
      2. Fit a logistic regression (sklearn.linear_model.LogisticRegression) or
         isotonic regression per subregion channel.
      3. Persist calibration coefficients to a JSON file referenced by config.
      4. At inference, apply: p_cal = sigmoid(A * logit(p) + B) per channel.

    Until calibration coefficients are available the function falls back to
    identity (no adjustment), which is safer than unvalidated empirical boosts.
    """
    logger   = get_logger("pybrain")
    cal_path = config.output_dir.parent / config.models.get(
        "platt_calibration", {}
    ).get("coefficients_file", "platt_coefficients.json")

    if cal_path.exists():
        try:
            with open(cal_path) as f:
                coeffs = json.load(f)
            for ch_idx, subregion in enumerate(["tc", "wt", "et"]):
                A = coeffs.get(subregion, {}).get("A")
                B = coeffs.get(subregion, {}).get("B")
                if A is not None and B is not None:
                    logit = np.log(
                        ensemble_prob[ch_idx] / (1.0 - ensemble_prob[ch_idx] + 1e-8) + 1e-8
                    )
                    ensemble_prob[ch_idx] = np.clip(
                        1.0 / (1.0 + np.exp(-(A * logit + B))), 0.0, 1.0
                    )
            logger.info("Platt scaling applied from saved coefficients.")
            return ensemble_prob
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(f"Platt calibration file invalid ({exc}). Falling back.")

    # ── Fallback: identity (no boost) until validated on a held-out cohort ────
    logger.warning(
        "Platt calibration coefficients not found. "
        "No probability adjustment applied (identity fallback). "
        "Train Platt/isotonic calibration on a validation cohort before clinical use."
    )
    # Previously: TC×1.10, ET×1.05 — removed because unvalidated multipliers
    # systematically inflate volumes without evidence.  Identity is safer.
    return ensemble_prob


# =============================================================================
# Post-processing
# =============================================================================

def postprocess_segmentation(
    ensemble_prob: np.ndarray,
    brain_mask: np.ndarray,
    vox_vol_cc: float,
    config: PipelineConfig,
    volumes: Dict[str, np.ndarray],
    model_probs_list: Optional[List[np.ndarray]] = None,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Convert ensemble probability maps to a binary BraTS segmentation.

    Steps:
      1. Probability calibration (Platt scaling or empirical fallback).
      2. Threshold → binary masks with hierarchical consistency (ET⊂TC⊂WT).
      3. Shape-based component filtering.
      4. CRF boundary refinement (WT, TC, and ET channels). [FIX-4]
      5. Anatomical surface-distance constraints. [FIX-3, FIX-5]
      6. BraTS label derivation (necrosis, edema, enhancing).
      7. Small isolated component removal.
    """
    logger   = get_logger("pybrain")
    post_cfg = config.models.get("postprocessing", {})

    # ── 1. Probability Calibration ────────────────────────────────────────────
    ensemble_prob = apply_platt_calibration(ensemble_prob, config)

    tc_prob = ensemble_prob[0]
    wt_prob = ensemble_prob[1]
    et_prob = ensemble_prob[2]

    # Initialize thresholds (will be updated by statistical optimization if enabled)
    # Fallbacks are 0.35 for all subregions — consistent with defaults.yaml.
    # Previously WT fallback was 0.45 (inconsistent with config and all other thresholds).
    final_thresholds = {
        'tc': config.thresholds.get("tc", 0.35),
        'wt': config.thresholds.get("wt", 0.35),
        'et': config.thresholds.get("et", 0.35)
    }

    wt_thresh = final_thresholds.get("wt", 0.35)
    tc_thresh = final_thresholds.get("tc", 0.35)
    et_thresh = final_thresholds.get("et", 0.35)
    logger.info(f"Thresholds applied: WT={wt_thresh}  TC={tc_thresh}  ET={et_thresh}")

    # ── 2. Threshold + hierarchical consistency ───────────────────────────────
    wt_bin = (wt_prob > wt_thresh).astype(np.float32) * brain_mask
    tc_bin = (tc_prob > tc_thresh).astype(np.float32) * brain_mask * wt_bin
    et_bin = (et_prob > et_thresh).astype(np.float32) * brain_mask * tc_bin

    # ── 3. Shape-based component filtering ────────────────────────────────────
    if post_cfg.get("shape_filtering", True):
        from skimage.measure import label, regionprops
        labels    = label(wt_bin > 0)
        props     = regionprops(labels)
        max_ecc   = post_cfg.get("max_eccentricity", 0.98)
        min_sol   = post_cfg.get("min_solidity", 0.55)
        keep_mask = np.zeros_like(wt_bin, dtype=bool)
        for p in props:
            is_good = (
                p.solidity >= min_sol
                and getattr(p, "extent", 1.0) >= 0.005
                and getattr(p, "eccentricity", 0.0) <= max_ecc
            )
            if is_good or (p.area * vox_vol_cc > 1.0):
                keep_mask[labels == p.label] = True
    else:
        keep_mask = np.ones_like(wt_bin, dtype=bool)

    wt_bin = wt_bin * keep_mask
    tc_bin = tc_bin * keep_mask
    et_bin = et_bin * keep_mask

    # ── 4. CRF refinement (WT, TC, ET) ───────────────────────────────────────
    # [FIX-4] ET is now also refined, which is critical for RANO 2.0 metrics.
    if post_cfg.get("crf_refinement", False):
        wt_bin = apply_3d_crf(wt_bin, volumes["T1c"], config)
        tc_bin = apply_3d_crf(tc_bin, volumes["T1c"], config)
        et_bin = apply_3d_crf(et_bin, volumes["T1c"], config)
        # Re-enforce consistency after CRF (CRF may independently expand boundaries)
        tc_bin = tc_bin * wt_bin
        et_bin = et_bin * tc_bin

    # ── 5. Anatomical constraints ─────────────────────────────────────────────
    if post_cfg.get("anatomical_constraints", True):
        tumor_type      = config.patient.get("tumor_type", "")
        surface_thresh  = _resolve_surface_threshold(tumor_type, config)
        wt_bin = apply_anatomical_constraints(
            wt_bin, brain_mask, voxel_spacing, surface_thresh
        )
        tc_bin = tc_bin * (wt_bin > 0)
        et_bin = et_bin * (wt_bin > 0)

    # ── 6. BraTS label derivation ─────────────────────────────────────────────
    enhancing = et_bin
    necrotic  = np.clip(tc_bin - enhancing, 0, 1)
    edema     = np.clip(wt_bin - tc_bin,    0, 1)

    seg_full = np.zeros_like(necrotic, dtype=np.uint8)
    seg_full[edema    > 0] = 2
    seg_full[necrotic > 0] = 1
    seg_full[enhancing > 0] = 3

    # ── 7. Small component removal ────────────────────────────────────────────
    min_voxels = max(1, int(config.min_component_cc / vox_vol_cc))
    labeled, n_comp = ndi.label(seg_full > 0)  # type: ignore[arg-type]
    if n_comp > 1:
        sizes  = ndi.sum(seg_full > 0, labeled, list(range(1, n_comp + 1)))
        largest = int(np.argmax(sizes)) + 1
        keep    = np.zeros(seg_full.shape, dtype=bool)
        for i, sz in enumerate(sizes, start=1):
            if sz >= min_voxels or i == largest:
                keep[labeled == i] = True
        seg_full[~keep] = 0

    return seg_full, necrotic, edema, enhancing, final_thresholds


def _resolve_surface_threshold(tumor_type: str, config: PipelineConfig) -> float:
    """
    [FIX-5] Return a surface-distance suppression threshold (mm) based on
    tumour type from session metadata.

    Tumour-type rationale
    ---------------------
    cortical_glioma      : grade-2 glioma often involves the cortical ribbon —
                           must use a large threshold (10 mm) to avoid pruning
                           real tumour adjacent to the cortical surface.
    skull_base / thalamic: tumour centroid may be genuinely close to anatomical
                           edges; use 8 mm to reduce false suppression.
    leptomeningeal       : surface-adherent by definition — disable suppression.
    default              : 3 mm retains the original conservative behaviour for
                           typical parenchymal glioblastoma.
    """
    ttype = (tumor_type or "").lower()
    mapping = config.models.get("surface_thresholds", {})
    if mapping:
        for key, val in mapping.items():
            if key in ttype:
                return float(val)

    if any(t in ttype for t in ("cortical",)):
        return 10.0
    if any(t in ttype for t in ("skull_base", "thalamic", "hypothalamic")):
        return 8.0
    if "leptomeningeal" in ttype:
        return 999.0  # effectively disabled
    return 3.0


def apply_anatomical_constraints(
    mask: np.ndarray,
    brain_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    surface_thresh_mm: float = 3.0,
) -> np.ndarray:
    """
    Two-stage anatomical constraint for brain tumour segmentation.

    Stage 1 — Hard limit to brain parenchyma.
    Stage 2 — Surface-distance suppression using a 3-D Euclidean distance
               transform with **true anisotropic voxel spacing**.

    [FIX-3] The original implementation used the cube-root of voxel volume
    as a scalar conversion factor, which is only correct for isotropic voxels.
    This version passes *sampling=voxel_spacing* directly to
    distance_transform_edt so that each axis is handled independently.
    """
    from scipy.ndimage import distance_transform_edt

    # Stage 1: hard brain parenchyma limit
    mask = mask * (brain_mask > 0)

    if surface_thresh_mm >= 900:          # leptomeningeal — skip suppression
        return mask.astype(np.float32)

    # Stage 2: distance from brain surface using true voxel spacing
    # distance_transform_edt: for each NON-ZERO voxel, computes Euclidean
    # distance to the nearest ZERO voxel.  brain_interior is 1 inside the
    # brain and 0 outside, so each brain voxel gets its depth from the
    # brain surface (nearest exterior voxel).
    brain_interior = np.where(brain_mask > 0, 1, 0).astype(np.float32)
    dist_from_surface = np.asarray(distance_transform_edt(
        brain_interior,
        sampling=voxel_spacing,           # (sz, sy, sx) in mm — true anisotropic
    ), dtype=np.float32)

    labeled, n_comp = ndi.label(mask > 0)  # type: ignore[assignment]
    n_comp = int(n_comp)
    if n_comp == 0:
        return mask

    suppress_mask = np.zeros_like(mask, dtype=bool)

    for comp_id in range(1, n_comp + 1):
        comp_mask = (labeled == comp_id)
        # Use the MAXIMUM distance (deepest voxel) in this component.
        # If even the deepest voxel is within the threshold, the entire
        # component is superficial noise and should be suppressed.
        # This avoids pruning elongated tumours whose centroid happens
        # to be near the surface but that extend deep into parenchyma.
        max_depth_mm = float(dist_from_surface[comp_mask].max())
        if max_depth_mm < surface_thresh_mm:
            suppress_mask[comp_mask] = True

    return np.where(suppress_mask, 0, mask).astype(np.float32)


def apply_3d_crf(
    mask: np.ndarray,
    image: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Refine segmentation boundaries using Dense CRF.

    Currently applied slice-wise (2D) for computational feasibility.
    Upgrade path: pass full 3D volume to pydensecrf.densecrf.DenseCRF with a
    3-D Gaussian pairwise potential, or replace entirely with SAM2
    prompt-based boundary correction (see Roadmap item #5).
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels

        logger = get_logger("pybrain")
        logger.info("Applying CRF refinement (2D slice-wise)...")

        refined = np.zeros_like(mask)
        for z in range(mask.shape[2]):
            m_slice = mask[..., z]
            i_slice = image[..., z]
            if m_slice.max() == 0:
                continue
            labels = (m_slice > 0.5).astype(np.int32)
            unary  = unary_from_labels(labels, 2, gt_prob=0.8)
            d      = dcrf.DenseCRF2D(m_slice.shape[1], m_slice.shape[0], 2)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            img_c   = (norm01(i_slice) * 255).astype(np.uint8)
            img_3ch = np.ascontiguousarray(np.stack([img_c, img_c, img_c], axis=-1))
            d.addPairwiseBilateral(sxy=5, srgb=5, rgbim=img_3ch, compat=10)
            q = d.inference(5)
            refined[..., z] = np.argmax(q, axis=0).reshape(m_slice.shape)

        return refined.astype(np.float32)
    except ImportError:
        return mask


# =============================================================================
# Per-Subregion EMA Calibration
# =============================================================================

_SUBREGION_KEYS = ("wt", "tc", "et", "nc")


def apply_calibration(
    volumes_cc: Dict[str, float],
    config: PipelineConfig,
) -> Dict[str, float]:
    """
    [FIX-1] Apply per-subregion historical calibration factors.

    Reads separate EMA factors for WT / TC / ET / NC from the calibration
    JSON file and scales each subregion volume independently.
    Returns a dict of calibrated volumes.
    """
    cal_cfg     = config.models.get("calibration", {})
    if not cal_cfg.get("enabled", True):
        return dict(volumes_cc)

    factor_path = config.output_dir.parent / cal_cfg.get(
        "factor_file", "calibration_factors.json"
    )

    factors: Dict[str, float] = {k: 1.0 for k in _SUBREGION_KEYS}
    if factor_path.exists():
        try:
            with open(factor_path) as f:
                stored = json.load(f)
            for key in _SUBREGION_KEYS:
                factors[key] = float(stored.get(key, {}).get("factor", 1.0))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    return {k: volumes_cc.get(k, 0.0) * factors[k] for k in _SUBREGION_KEYS}


def update_calibration_ema(
    volumes_cc: Dict[str, float],
    config: PipelineConfig,
) -> None:
    """
    [FIX-1] Update per-subregion EMA calibration factors from radiologist references.

    Iterates over WT / TC / ET / NC.  Any subregion without a radiologist
    reference for the current exam is skipped — its EMA factor is unchanged.
    """
    logger   = get_logger("pybrain")
    cal_cfg  = config.models.get("calibration", {})
    alpha    = cal_cfg.get("ema_alpha", 0.2)
    factor_path = config.output_dir.parent / cal_cfg.get(
        "factor_file", "calibration_factors.json"
    )

    # Guard: only update EMA when at least one radiologist reference is present.
    # If radiologist_ref is absent or all values are None, the AI's own predictions
    # would become the reference — accumulating and compounding over successive runs.
    has_any_ref = any(
        config.radiologist_ref.get(k) is not None for k in _SUBREGION_KEYS
    )
    if not has_any_ref:
        logger.info(
            "EMA calibration update skipped: no radiologist reference values present. "
            "Provide radiologist_ref in session metadata to enable calibration updates."
        )
        return

    # Load current state
    stored: Dict[str, Any] = {}
    if factor_path.exists():
        try:
            with open(factor_path) as f:
                stored = json.load(f)
        except (json.JSONDecodeError, ValueError):
            stored = {}

    updated = False
    for key in _SUBREGION_KEYS:
        ref_val = config.radiologist_ref.get(key)
        ai_vol  = volumes_cc.get(key, 0.0)
        if ref_val is None or ai_vol < 1e-6:
            continue

        current_ratio = float(ref_val) / ai_vol
        prev          = stored.get(key, {})
        prev_factor   = float(prev.get("factor", current_ratio))
        prev_samples  = int(prev.get("samples", 0))

        new_factor  = current_ratio if prev_samples == 0 else (
            alpha * current_ratio + (1 - alpha) * prev_factor
        )
        stored[key] = {"factor": new_factor, "samples": prev_samples + 1}

        logger.info(
            f"  Calibration [{key.upper()}]: "
            f"ref={ref_val:.2f} cc | AI={ai_vol:.2f} cc | "
            f"ratio={current_ratio:.4f} | new_factor={new_factor:.4f} "
            f"(EMA α={alpha}, n={prev_samples + 1})"
        )
        updated = True

    if updated:
        with open(factor_path, "w") as f:
            json.dump(stored, f, indent=2)


# =============================================================================
# Longitudinal Tracking
# =============================================================================

def compute_longitudinal_delta(
    volumes_cc: Dict[str, float],
    config: PipelineConfig,
) -> Dict[str, Optional[float]]:
    """
    [FIX-8] Compare current subregion volumes against the most recent prior exam.

    Looks for a 'prior_volumes.json' in the patient-level directory (one level
    above the current output_dir).  If found, computes cc deltas and percentage
    change per subregion.  The calling function appends these to the quality
    report JSON.

    Returns a dict of {subregion: delta_cc} or {subregion: None} if no prior.
    """
    logger    = get_logger("pybrain")
    prior_path = config.output_dir.parent / "prior_volumes.json"

    if not prior_path.exists():
        logger.info("No prior volumes found — longitudinal delta not computed.")
        return {k: None for k in _SUBREGION_KEYS}

    try:
        with open(prior_path) as f:
            prior = json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"Could not read prior volumes: {exc}")
        return {k: None for k in _SUBREGION_KEYS}

    deltas: Dict[str, Optional[float]] = {}
    for key in _SUBREGION_KEYS:
        prior_val = prior.get(key)
        curr_val  = volumes_cc.get(key, 0.0)
        if prior_val is not None:
            delta = curr_val - float(prior_val)
            pct   = 100.0 * delta / (float(prior_val) + 1e-8)
            deltas[key] = round(delta, 3)
            logger.info(
                f"  Longitudinal [{key.upper()}]: "
                f"prior={prior_val:.2f} cc | current={curr_val:.2f} cc | "
                f"Δ={delta:+.2f} cc ({pct:+.1f}%)"
            )
        else:
            deltas[key] = None

    return deltas


def save_current_volumes_as_prior(
    volumes_cc: Dict[str, float],
    config: PipelineConfig,
) -> None:
    """Persist the current subregion volumes so the next exam can diff against them."""
    prior_path = config.output_dir.parent / "prior_volumes.json"
    with open(prior_path, "w") as f:
        json.dump({k: round(volumes_cc.get(k, 0.0), 4) for k in _SUBREGION_KEYS}, f, indent=2)


# =============================================================================
# Visualization
# =============================================================================

def _select_tumor_slices(
    wt_prob: np.ndarray,
    axis: int,
    n_slices: int = 8,
) -> np.ndarray:
    """
    [FIX-6] Select the N slices along *axis* with the highest whole-tumour
    probability mass.

    The original implementation used np.linspace (equidistant), which often
    returned slices entirely outside the tumour.  This version sums the WT
    probability along the chosen axis and picks the top-N indices.
    """
    sums      = wt_prob.sum(axis=tuple(i for i in range(3) if i != axis))
    top_idx   = np.argsort(sums)[::-1][:n_slices]
    return np.sort(top_idx)


def generate_visualization(
    volumes: Dict[str, np.ndarray],
    seg_components: Dict[str, np.ndarray],
    output_dir: Path,
    ref_img: Any,
    wt_prob: Optional[np.ndarray] = None,
) -> None:
    """
    Create three orthogonal views (axial, coronal, sagittal) with tumour overlays.
    Slices are ranked by WT probability mass rather than chosen equidistantly. [FIX-6]
    """
    norm_vols   = {k: norm01(volumes[k]) for k in ["T1", "T1c", "FLAIR"]}
    comp_colors = {"necrotic": "#4499ff", "edema": "#44ee44", "enhancing": "#ff4444"}

    # Fallback: use edema mask as a proxy when no prob map is provided
    proxy_prob = wt_prob if wt_prob is not None else (
        seg_components.get("edema", np.zeros_like(volumes["T1"]))
    )

    def get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
        if axis == 0:
            return arr[idx, :, :].T
        if axis == 1:
            return arr[:, idx, :].T
        return arr[:, :, idx].T

    view_specs = [
        (2, "axial"),
        (1, "coronal"),
        (0, "sagittal"),
    ]

    for axis, name in view_specs:
        slices = _select_tumor_slices(proxy_prob, axis=axis, n_slices=8)
        fig    = plt.figure(figsize=(22, 11), facecolor="#0a0a0a")
        gs     = gridspec.GridSpec(4, 8, figure=fig, hspace=0.04, wspace=0.04)

        for r, mod in enumerate(["T1", "T1c", "FLAIR", "Overlay"]):
            for c, sl in enumerate(slices):
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")
                if mod == "Overlay":
                    ax.imshow(get_slice(norm_vols["T1c"], axis, sl), cmap="gray", vmin=0, vmax=1)
                    for comp_name, mask in seg_components.items():
                        s = get_slice(mask, axis, sl)
                        if s.max() > 0:
                            rgba = np.zeros((*s.shape, 4))
                            rgba[..., :3] = mcolors.to_rgb(comp_colors[comp_name])
                            rgba[...,  3] = s * 0.65
                            ax.imshow(rgba)
                else:
                    ax.imshow(get_slice(norm_vols[mod], axis, sl), cmap="gray", vmin=0, vmax=1)

        plt.savefig(
            output_dir / f"view_{name}.png",
            bbox_inches="tight",
            facecolor="#0a0a0a",
            dpi=160,
        )
        plt.close()


# =============================================================================
# Uncertainty Summarisation
# =============================================================================

def summarise_uncertainty(
    uncertainty: np.ndarray,
    seg_full: np.ndarray,
) -> Dict[str, Any]:
    """
    [FIX-7] Compute summary uncertainty statistics within the tumour mask.

    Returns mean, median, and 95th-percentile uncertainty, both globally
    and restricted to tumour voxels.  These values are included in the
    quality JSON so downstream reporting tools don't need to load the NIfTI.
    """
    tumour_vox = uncertainty[seg_full > 0]
    if tumour_vox.size == 0:
        return {
            "mean_global":    float(uncertainty.mean()),
            "mean_tumour":    None,
            "p95_tumour":     None,
        }
    return {
        "mean_global": round(float(uncertainty.mean()), 5),
        "mean_tumour": round(float(tumour_vox.mean()), 5),
        "p95_tumour":  round(float(np.percentile(tumour_vox, 95)), 5),
    }


def flag_high_uncertainty_regions(
    uncertainty: np.ndarray,
    seg_full: np.ndarray,
    p95_threshold: float = 0.25,
) -> bool:
    """
    Return True if the 95th-percentile uncertainty within tumour voxels exceeds
    a threshold that warrants radiologist review before clinical use.
    """
    tumour_vox = uncertainty[seg_full > 0]
    if tumour_vox.size == 0:
        return False
    return bool(np.percentile(tumour_vox, 95) > p95_threshold)


# =============================================================================
# Multi-Focal Detection
# =============================================================================

def detect_multifocal(
    seg_full: np.ndarray,
    vox_vol_cc: float,
    min_satellite_cc: float = 0.5,
) -> Tuple[bool, int]:
    """
    [FIX-9] Detect whether the tumour is multi-focal.

    A tumour is considered multi-focal if there are ≥2 connected components
    each larger than *min_satellite_cc* cubic centimetres.

    Returns (is_multifocal, n_foci).
    """
    labeled, n_comp = ndi.label(seg_full > 0)  # type: ignore[assignment]
    n_comp = int(n_comp)
    if n_comp <= 1:
        return False, n_comp

    min_vox   = max(1, int(min_satellite_cc / vox_vol_cc))
    valid_foci = sum(
        1
        for i in range(1, n_comp + 1)
        if int(ndi.sum((seg_full > 0).astype(np.uint8), labeled, i)) >= min_vox
    )
    return valid_foci >= 2, valid_foci


# =============================================================================
# Output Saving
# =============================================================================

def save_all_outputs(
    seg_full: np.ndarray,
    seg_components: Dict[str, np.ndarray],
    ensemble_prob: np.ndarray,
    uncertainty: np.ndarray,
    brain_mask: np.ndarray,
    ref_img: Any,
    config: PipelineConfig,
    quality_report: Dict[str, Any],
    model_probs_list: Optional[List[np.ndarray]] = None,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    applied_thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Save NIfTI segmentations, probability maps, and JSON reports."""
    logger     = get_logger("pybrain")
    output_dir = config.output_dir

    for p in output_dir.glob("segmentation_*.nii.gz"):
        try:
            p.unlink()
        except OSError as exc:
            logger.warning(f"Could not delete stale file {p}: {exc}")

    logger.debug(f"Saving segmentation shape: {seg_full.shape}")
    save_nifti(seg_full,                   output_dir / "segmentation_ensemble.nii.gz", ref_img)
    save_nifti(seg_full,                   output_dir / "segmentation_full.nii.gz",     ref_img)
    save_nifti(seg_components["necrotic"], output_dir / "seg_necrotic.nii.gz",          ref_img)
    save_nifti(seg_components["edema"],    output_dir / "seg_edema.nii.gz",             ref_img)
    save_nifti(seg_components["enhancing"],output_dir / "seg_enhancing.nii.gz",         ref_img)
    
    # Statistical Threshold Optimization (BraTS QU-inspired)
    # Initialize final_thresholds for statistical optimization
    # Fallback 0.35 for all — consistent with defaults.yaml and postprocess_segmentation.
    final_thresholds = {
        'tc': config.thresholds.get("tc", 0.35),
        'wt': config.thresholds.get("wt", 0.35),
        'et': config.thresholds.get("et", 0.35)
    }
    
    # statistical_thresholds is disabled (enabled=false in defaults.yaml).
    # The optimizer block ran after NIfTIs were written — a silent no-op.
    # It is kept here for future re-enablement once the pipeline order is
    # restructured, but it must remain disabled until then.
    # The uncertainty parameter passed in is used directly; no recomputation.

    # Save ensemble probabilities and uncertainty
    save_nifti(
        ensemble_prob,
        output_dir / "ensemble_probability.nii.gz",
        ref_img
    )
    save_nifti(
        uncertainty,
        output_dir / "ensemble_uncertainty.nii.gz",
        ref_img
    )

    stats = {
        "segmentation_source": "ensemble",
        "volume_cc": {
            "brain":       quality_report["brain_vol_cc"],
            "whole_tumor": quality_report["v_wt_cc"],
            "core":        quality_report["v_tc_cc"],
            "enhancing":   quality_report["v_et_cc"],
            "necrotic":    quality_report["v_nc_cc"],
            "edema":       quality_report["v_ed_cc"],
        },
        "calibrated_volume_cc": quality_report.get("calibrated", {}),
        "tumor_pct_brain": quality_report["tumor_pct_brain"],
        "thresholds":       applied_thresholds or config.thresholds,
        "vox_vol_cc":       quality_report["vox_vol_cc"],
    }
    with open(output_dir / "tumor_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "segmentation_quality.json", "w") as f:
        json.dump(
            {
                "patient":              quality_report.get("patient", "unknown"),
                "exam_date":            quality_report.get("exam_date", ""),
                "engine":               "pybrain v3.0",
                "quality":              quality_report,
                "tumour_inside_brain":  not bool(
                    np.any((seg_full > 0) & (brain_mask == 0))
                ),
                "uncertainty_summary":  quality_report.get("uncertainty_summary", {}),
                "high_uncertainty_flag":quality_report.get("high_uncertainty_flag", False),
                "multifocal":           quality_report.get("multifocal", False),
                "n_foci":               quality_report.get("n_foci", 1),
                "longitudinal_delta_cc":quality_report.get("longitudinal_delta_cc", {}),
                "registration_nmi":     quality_report.get("registration_nmi", {}),
                "roi_localisation_failed": quality_report.get("roi_localisation_failed", False),
            },
            f,
            indent=2,
        )


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Brain Tumor Segmentation Pipeline  v3.0")
    logger.info("=" * 60)

    try:
        # ── Configuration ─────────────────────────────────────────────────────
        config = load_pipeline_config()
        logger.info(f"Device: {config.device} | Model device: {config.model_device}")

        # ── Load MRI volumes ──────────────────────────────────────────────────
        logger.info("Loading MRI volumes...")
        volumes, ref_img = load_mri_volumes(config.monai_dir)

        # Extract true voxel spacing for anisotropic distance transforms [FIX-3]
        zooms = ref_img.header.get_zooms()[:3]
        voxel_spacing: Tuple[float, float, float] = (
            float(zooms[0]), float(zooms[1]), float(zooms[2])
        )
        config.voxel_spacing = voxel_spacing
        vox_vol_cc = float(np.prod(zooms)) / 1000.0
        logger.info(
            f"Voxel spacing: {voxel_spacing} mm | Voxel volume: {vox_vol_cc:.6f} cc"
        )

        # ── Stage 1b registration warning passthrough ─────────────────────────
        reg_flag_path = config.monai_dir / "registration_warnings.json"
        if reg_flag_path.exists():
            try:
                with open(reg_flag_path) as f:
                    reg_info = json.load(f)
                failed = reg_info.get("failed_sequences", [])
                if failed:
                    logger.warning(
                        f"⚠️  Stage 1b registration warning: "
                        f"{', '.join(failed)} may be mis-aligned."
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        # ── [FIX-2] NMI quality assurance across all MRI sequence pairs ───────
        logger.info("Running MRI registration QA (all sequence pairs)...")
        nmi_cfg     = config.models.get("registration_qc", {})
        nmi_threshold = nmi_cfg.get("nmi_threshold", 1.05)
        registration_nmi = validate_all_registrations(volumes, nmi_threshold)

        # ── Brain mask ────────────────────────────────────────────────────────
        logger.info("Acquiring brain mask...")
        mask_path = config.monai_dir / "brain_mask.nii.gz"
        if mask_path.exists():
            logger.info("Using pre-computed brain mask from Stage 1b.")
            brain_mask = nib.load(str(mask_path)).get_fdata()  # type: ignore[attr-defined]
        else:
            logger.info("Computing multi-contrast brain mask...")
            brain_mask = robust_brain_mask(volumes, vox_vol_cc=vox_vol_cc)
            if config.models.get("save_brain_mask", True):
                save_nifti(brain_mask, config.output_dir / "brain_mask.nii.gz", ref_img)

        brain_vol_cc = compute_volume_cc(brain_mask, vox_vol_cc)
        logger.info(f"Brain volume: {brain_vol_cc:.1f} cc")

        # ── CT boost data ─────────────────────────────────────────────────────
        ct_data = None
        ct_path = config.monai_dir / "ct_brain_registered.nii.gz"
        if ct_path.exists():
            logger.info("Loading registered CT for boost...")
            ct_data = nib.load(str(ct_path)).get_fdata().astype(np.float32)  # type: ignore[attr-defined]
            config.ct_boost["enabled"] = True
        else:
            logger.info("No registered CT found — CT boost disabled.")

        # ── Preprocessing ─────────────────────────────────────────────────────
        input_tensor = preprocess_volumes(volumes, brain_mask, config)
        input_tensor = input_tensor.to(config.model_device)

        # ── Stage 3a: Fast ROI localisation (SegResNet pass 1) ───────────────
        logger.info("Stage 3a: ROI localisation (SegResNet)...")
        sr_model = load_segresnet(config.bundle_dir, config.model_device)
        sr_prob  = run_segresnet_inference(
            sr_model, input_tensor, config.device,
            config.models.get("segresnet", {}),
            model_device=config.model_device,
        )
        roi_slices, orig_shape = get_tumor_bbox(sr_prob[1])
        roi_localisation_failed = all(
            isinstance(s, slice) and s.start is None for s in roi_slices
        )
        if roi_localisation_failed:
            logger.warning(
                "ROI localisation fallback: SegResNet found no tumour voxels above "
                "threshold. Ensemble will run on the full volume."
            )
        logger.info(f"Tumour ROI: {roi_slices}")
        del sr_model
        gc.collect()
        _gpu_cache_clear(config.model_device)

        # ── Stage 3b: Focused ensemble on ROI ─────────────────────────────────
        logger.info("Stage 3b: Focused ensemble inference (ROI)...")
        roi_input = input_tensor[:, :, roi_slices[0], roi_slices[1], roi_slices[2]].clone()
        del input_tensor
        gc.collect()
        _gpu_cache_clear(config.model_device)

        # Crop the Stage 3a SegResNet full-volume probability to the ROI instead
        # of re-running SegResNet on the undersized ROI crop.  The full-volume
        # pass (window = 240×240×160, matching the trained size) is higher quality
        # than a second pass that would be padded to fit a ~128³ input.
        sr_prob_roi = sr_prob[:, roi_slices[0], roi_slices[1], roi_slices[2]]
        del sr_prob
        gc.collect()

        results_roi, mc_uncertainties_roi = run_models(
            roi_input, config,
            precomputed={"segresnet": sr_prob_roi},
        )

        # Reassemble ROI → full volume
        model_probs: Dict[str, np.ndarray] = {}
        for name, prob_roi in results_roi.items():
            full_prob = np.zeros((3,) + orig_shape, dtype=np.float32)
            full_prob[:, roi_slices[0], roi_slices[1], roi_slices[2]] = prob_roi
            model_probs[name] = full_prob

        del roi_input, results_roi, mc_uncertainties_roi
        gc.collect()
        _gpu_cache_clear(config.model_device)

        # ── Ensemble fusion ───────────────────────────────────────────────────
        # uncertainty is computed correctly after ensemble fusion (line ~1755).
        # A premature call here used prob_list[0] as a fake ensemble, which was
        # always overwritten.  Pass None; fuse_ensemble only uses it when
        # adaptive subregion weights are enabled (default: False).
        ensemble_prob, contributed = fuse_ensemble(model_probs, config, uncertainty=None)
        logger.info(f"Ensemble fusion using: {', '.join(contributed)}")

        # ── CT boost ──────────────────────────────────────────────────────────
        if ct_data is not None and config.ct_boost.get("enabled", False):
            # Create MRI data array for NMI validation (use T1c channel if available)
            mri_for_nmi = None
            if 'T1c' in volumes:
                mri_for_nmi = volumes['T1c']
            elif 'FLAIR' in volumes:
                mri_for_nmi = volumes['FLAIR']
            
            ensemble_prob = apply_ct_boost(
                    ensemble_prob, ct_data, config,
                    brain_mask=brain_mask, volumes=volumes, mri_data=mri_for_nmi,
                )

        # ── Post-processing ───────────────────────────────────────────────────
        seg_full, necrotic, edema, enhancing, applied_thresholds = postprocess_segmentation(
            ensemble_prob, brain_mask, vox_vol_cc, config,
            volumes=volumes, voxel_spacing=voxel_spacing,
        )

        # ── Sub-region volumes ────────────────────────────────────────────────
        raw_volumes_cc: Dict[str, float] = {
            "wt": compute_volume_cc(seg_full > 0,          vox_vol_cc),
            "tc": compute_volume_cc((necrotic + enhancing) > 0, vox_vol_cc),
            "et": compute_volume_cc(enhancing,             vox_vol_cc),
            "nc": compute_volume_cc(necrotic,              vox_vol_cc),
        }
        ed_vol = compute_volume_cc(edema, vox_vol_cc)

        # ── [FIX-1] Per-subregion EMA calibration update ─────────────────────
        logger.info("Stage 3c: Per-subregion calibration update...")
        update_calibration_ema(raw_volumes_cc, config)
        calibrated_volumes = apply_calibration(raw_volumes_cc, config)

        # ── [FIX-8] Longitudinal delta ────────────────────────────────────────
        longitudinal_delta = compute_longitudinal_delta(raw_volumes_cc, config)
        save_current_volumes_as_prior(raw_volumes_cc, config)

        # ── Uncertainty ───────────────────────────────────────────────────────
        prob_list   = [model_probs[n] for n in contributed if n in model_probs]
        model_probs_list = prob_list  # Store for later use in save_all_outputs
        uncertainty = compute_uncertainty(ensemble_prob, prob_list)
        del model_probs
        gc.collect()
        _gpu_cache_clear(config.model_device)

        uncertainty_summary  = summarise_uncertainty(uncertainty, seg_full)
        high_uncertainty_flag = flag_high_uncertainty_regions(uncertainty, seg_full)
        if high_uncertainty_flag:
            logger.warning(
                "⚠️  High uncertainty detected in tumour region. "
                "Radiologist review recommended before clinical use."
            )

        # ── [FIX-9] Multi-focal detection ─────────────────────────────────────
        is_multifocal, n_foci = detect_multifocal(seg_full, vox_vol_cc)
        if is_multifocal:
            logger.info(
                f"  Multi-focal tumour detected: {n_foci} distinct foci. "
                "Smallest lesions may not be included in the primary component."
            )

        # ── Quality report ────────────────────────────────────────────────────
        quality_report: Dict[str, Any] = {
            "patient":              config.patient.get("name", "unknown"),
            "exam_date":            config.patient.get("exam_date", ""),
            "v_wt_cc":              raw_volumes_cc["wt"],
            "v_tc_cc":              raw_volumes_cc["tc"],
            "v_et_cc":              raw_volumes_cc["et"],
            "v_nc_cc":              raw_volumes_cc["nc"],
            "v_ed_cc":              ed_vol,
            "brain_vol_cc":         brain_vol_cc,
            "tumor_pct_brain":      100 * raw_volumes_cc["wt"] / (brain_vol_cc + 1e-8),
            "vox_vol_cc":           vox_vol_cc,
            "calibrated":           calibrated_volumes,
            "longitudinal_delta_cc":longitudinal_delta,
            "uncertainty_summary":  uncertainty_summary,
            "high_uncertainty_flag":high_uncertainty_flag,
            "multifocal":           is_multifocal,
            "n_foci":               n_foci,
            "registration_nmi":     registration_nmi,
            "roi_localisation_failed": roi_localisation_failed,
        }

        # ── Visualizations ────────────────────────────────────────────────────
        logger.info("Generating visualizations...")
        seg_components = {"necrotic": necrotic, "edema": edema, "enhancing": enhancing}
        generate_visualization(
            volumes, seg_components, config.output_dir, ref_img,
            wt_prob=ensemble_prob[1],   # [FIX-6] pass prob map for tumour-aware slices
        )

        # ── Save all outputs ──────────────────────────────────────────────────
        logger.info("Saving outputs...")
        save_all_outputs(
            seg_full, seg_components, ensemble_prob, uncertainty,
            brain_mask, ref_img, config, quality_report,
            model_probs_list, voxel_spacing,
            applied_thresholds=applied_thresholds
        )

        # ── Optional ground-truth validation ──────────────────────────────────
        gt_path = get_paths(config.sess).get("ground_truth") if config.sess else None
        if gt_path and Path(gt_path).exists():
            logger.info("Ground truth found — evaluating segmentation...")
            val_script = Path(__file__).parent / "5_validate_segmentation.py"
            pred_path  = config.output_dir / "segmentation_full.nii.gz"
            if val_script.exists():
                try:
                    result = subprocess.run(
                        [sys.executable, str(val_script),
                         "--pred", str(pred_path), "--gt", str(gt_path)],
                        capture_output=True, text=True, timeout=300,
                    )
                    for line in result.stdout.split("\n"):
                        if line.strip():
                            logger.info(f"Validation: {line.strip()}")
                    if result.stderr:
                        logger.warning(f"Validation stderr: {result.stderr.strip()}")
                except subprocess.TimeoutExpired:
                    logger.warning("Validation script timed out (300 s) — skipping.")
                except Exception as exc:
                    logger.warning(f"Could not run validation script: {exc}")
            else:
                logger.warning("Validation script not found — skipping.")

        logger.info("=" * 60)
        logger.info("Pipeline finished successfully.")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Fatal error during pipeline execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
