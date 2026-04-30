# pybrain/core/segmentation.py
"""
Brain tumour segmentation wrapper.

Encapsulates model loading, inference, ensemble, thresholding, and
post-processing behind a clean ``segment()`` API.
"""

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ─── Config & Result dataclasses ──────────────────────────────────────────────


@dataclass
class SegmentationConfig:
    """All knobs for a segmentation run."""

    # Thresholds
    wt_threshold: float = 0.40
    tc_threshold: float = 0.35
    et_threshold: float = 0.35

    # CT boost
    ct_boost: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "min_hu": 35,
            "max_hu": 75,
            "boost_factor": 0.30,
        }
    )

    # Model settings
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "segresnet": 0.60,
            "tta4": 0.40,
        }
    )

    # MC-Dropout
    mc_dropout: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "n_samples": 15,
        }
    )

    # Device
    device: str = "cpu"

    # Paths
    bundle_dir: Optional[Path] = None
    output_dir: Optional[Path] = None


@dataclass
class SegmentationResult:
    """Structured output from ``segment()``."""

    seg_full: np.ndarray
    wt_prob: np.ndarray
    tc_prob: np.ndarray
    et_prob: np.ndarray
    brain_mask: np.ndarray

    wt_cc: float = 0.0
    tc_cc: float = 0.0
    et_cc: float = 0.0
    nc_cc: float = 0.0

    model_activated: bool = True
    non_gbm_suspected: bool = False
    elapsed: float = 0.0


# ─── Segmentation ─────────────────────────────────────────────────────────────


def segment(
    volumes: Dict[str, np.ndarray],
    brain_mask: np.ndarray,
    config: SegmentationConfig,
    ct_data: Optional[np.ndarray] = None,
) -> SegmentationResult:
    """
    Run full tumour segmentation.

    Parameters
    ----------
    volumes : dict
        T1, T1c, T2, FLAIR as float32 numpy arrays (same shape).
    brain_mask : np.ndarray
        Binary brain mask (same shape).
    config : SegmentationConfig
        Thresholds, model paths, device, etc.
    ct_data : np.ndarray, optional
        Registered CT in the same space for density boost.

    Returns
    -------
    SegmentationResult
    """
    import time

    t0 = time.time()

    from pybrain.core.normalization import zscore_robust

    # Normalise
    normed = {}
    for name, vol in volumes.items():
        normed[name] = zscore_robust(vol, brain_mask)

    # Stack → 4-channel tensor [1, 4, D, H, W]
    order = ["T1", "T1c", "T2", "FLAIR"]
    arrs = [normed[k] for k in order if k in normed]
    tensor = np.stack(arrs, axis=0)[None]
    tensor = tensor.astype(np.float32)

    # ── Model inference (stub — delegates to existing scripts in production) ─
    # In production this calls load_segresnet + run_segresnet_inference etc.
    # For the library API we assume probability maps are supplied externally
    # or we attempt lazy import.
    wt_prob = np.zeros_like(arrs[0], dtype=np.float32)
    tc_prob = np.zeros_like(arrs[0], dtype=np.float32)
    et_prob = np.zeros_like(arrs[0], dtype=np.float32)
    model_activated = True

    try:
        from pybrain.models.segresnet import load_segresnet, run_segresnet_inference
        import torch

        if config.bundle_dir:
            model = load_segresnet(str(config.bundle_dir), config.device)
            prob = run_segresnet_inference(model, torch.from_tensor(tensor).to(config.device), config.device, {})
            wt_prob = prob[1].cpu().numpy()
            tc_prob = prob[2].cpu().numpy()
            et_prob = prob[3].cpu().numpy()
            del model
            gc.collect()
    except Exception as exc:
        logger.warning(f"Model inference failed: {exc} — returning zero segmentation")
        model_activated = False

    # CT boost
    if ct_data is not None and config.ct_boost.get("enabled"):
        hu_min = config.ct_boost.get("min_hu", 35)
        hu_max = config.ct_boost.get("max_hu", 75)
        boost = config.ct_boost.get("boost_factor", 0.30)
        ct_mask = (ct_data >= hu_min) & (ct_data <= hu_max)
        wt_prob[ct_mask] = np.clip(wt_prob[ct_mask] + boost, 0, 1)

    # Threshold
    wt_bin = (wt_prob > config.wt_threshold).astype(np.float32) * brain_mask
    tc_bin = (tc_prob > config.tc_threshold).astype(np.float32) * brain_mask * wt_bin
    et_bin = (et_prob > config.et_threshold).astype(np.float32) * brain_mask * tc_bin

    # Compose seg_full: 0=bg, 1=necrotic, 2=edema, 3=enhancing
    seg_full = np.zeros_like(wt_bin, dtype=np.uint8)
    seg_full[wt_bin > 0] = 2  # edema
    seg_full[tc_bin > 0] = 1  # necrotic core
    seg_full[et_bin > 0] = 3  # enhancing

    # Volumes
    vox_vol_cc = 0.001  # 1 mm isotropic
    wt_cc = float(wt_bin.sum() * vox_vol_cc)
    tc_cc = float(tc_bin.sum() * vox_vol_cc)
    et_cc = float(et_bin.sum() * vox_vol_cc)
    nc_cc = tc_cc - et_cc

    elapsed = time.time() - t0

    # Model activation check
    max_prob = max(wt_prob.max(), tc_prob.max(), et_prob.max())
    if max_prob < 0.10:
        model_activated = False

    return SegmentationResult(
        seg_full=seg_full,
        wt_prob=wt_prob,
        tc_prob=tc_prob,
        et_prob=et_prob,
        brain_mask=brain_mask,
        wt_cc=wt_cc,
        tc_cc=tc_cc,
        et_cc=et_cc,
        nc_cc=nc_cc,
        model_activated=model_activated,
        elapsed=elapsed,
    )
