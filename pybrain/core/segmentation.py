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
    Run full tumour segmentation via the production ensemble.

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
    from pybrain.core.normalization import zscore_robust
    from pybrain.core.inference import run_ensemble_inference

    t0 = time.time()

    # Normalize each modality
    normed = {name: zscore_robust(vol, brain_mask) for name, vol in volumes.items()}

    # Resolve bundle_dir — must be set somewhere. Try config, then default.
    bundle_dir = config.bundle_dir
    if bundle_dir is None:
        # Fall back to repo default
        bundle_dir = Path(__file__).resolve().parent.parent.parent / "models" / "brats_bundle"
        if not bundle_dir.exists():
            raise FileNotFoundError(
                f"bundle_dir not set on SegmentationConfig and default {bundle_dir} "
                f"does not exist. Set config.bundle_dir to the model checkpoint directory."
            )

    # Run the ensemble (single source of truth)
    ensemble = run_ensemble_inference(
        volumes=normed,
        brain_mask=brain_mask,
        config={
            "ensemble_weights": config.ensemble_weights,
            "swinunetr": getattr(config, "swinunetr_cfg", {}),
            "sliding_window": getattr(config, "sliding_window_cfg", {}),
            "calibration": {"enabled": True},
        },
        bundle_dir=bundle_dir,
        device=config.device,
    )

    wt_prob = ensemble.wt_prob
    tc_prob = ensemble.tc_prob
    et_prob = ensemble.et_prob

    # CT boost (unchanged)
    if ct_data is not None and config.ct_boost.get("enabled"):
        hu_min = config.ct_boost.get("min_hu", 35)
        hu_max = config.ct_boost.get("max_hu", 75)
        boost = config.ct_boost.get("boost_factor", 0.30)
        ct_mask = (ct_data >= hu_min) & (ct_data <= hu_max)
        wt_prob[ct_mask] = np.clip(wt_prob[ct_mask] + boost, 0, 1)

    # Threshold + nested constraint (TC ⊆ WT, ET ⊆ TC)
    wt_bin = (wt_prob > config.wt_threshold).astype(np.float32) * brain_mask
    tc_bin = (tc_prob > config.tc_threshold).astype(np.float32) * wt_bin
    et_bin = (et_prob > config.et_threshold).astype(np.float32) * tc_bin

    # Compose: 0=bg, 1=necrotic, 2=edema, 3=enhancing
    seg_full = np.zeros_like(wt_bin, dtype=np.uint8)
    seg_full[wt_bin > 0] = 2
    seg_full[tc_bin > 0] = 1
    seg_full[et_bin > 0] = 3

    vox_vol_cc = 0.001
    wt_cc = float(wt_bin.sum() * vox_vol_cc)
    tc_cc = float(tc_bin.sum() * vox_vol_cc)
    et_cc = float(et_bin.sum() * vox_vol_cc)
    nc_cc = tc_cc - et_cc

    # Activation check (now meaningful — actual probabilities computed)
    max_prob = max(wt_prob.max(), tc_prob.max(), et_prob.max())
    model_activated = bool(max_prob >= 0.10)

    return SegmentationResult(
        seg_full=seg_full,
        wt_prob=wt_prob, tc_prob=tc_prob, et_prob=et_prob,
        brain_mask=brain_mask,
        wt_cc=wt_cc, tc_cc=tc_cc, et_cc=et_cc, nc_cc=nc_cc,
        model_activated=model_activated,
        elapsed=time.time() - t0,
    )
