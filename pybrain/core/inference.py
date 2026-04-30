"""Production-grade ensemble inference engine.

This module is the single source of truth for tumor segmentation inference.
Both the legacy script (scripts/3_brain_tumor_analysis.py) and the new
library API (pybrain.pipeline.run -> core.segmentation.segment) call into
this module so they cannot diverge.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pybrain.io.logging_utils import get_logger


logger = get_logger("core.inference")


@dataclass
class EnsembleProbabilities:
    """Output of run_ensemble_inference — already calibrated probability maps."""
    wt_prob: np.ndarray   # whole tumor [D, H, W]
    tc_prob: np.ndarray   # tumor core
    et_prob: np.ndarray   # enhancing tumor
    per_model_dice: dict[str, float] | None = None


def run_ensemble_inference(
    volumes: dict[str, np.ndarray],
    brain_mask: np.ndarray,
    config: dict[str, Any],
    bundle_dir: Path,
    device: str | torch.device = "cpu",
) -> EnsembleProbabilities:
    """
    Run the full SegResNet + TTA-4 + SwinUNETr ensemble.

    Parameters
    ----------
    volumes : dict
        Z-score-normalized volumes keyed by 'T1', 'T1c', 'T2', 'FLAIR'.
        All same shape, float32.
    brain_mask : np.ndarray
        Binary brain mask, same shape as volumes.
    config : dict
        Pipeline config (ensemble_weights, thresholds, swin/segresnet sub-configs).
    bundle_dir : Path
        Path to models/brats_bundle/ containing checkpoints.
    device : str | torch.device

    Returns
    -------
    EnsembleProbabilities with calibrated wt/tc/et probability maps.
    """
    from pybrain.models.segresnet import (
        load_segresnet, run_segresnet_inference, run_tta_ensemble,
    )
    from pybrain.models.swinunetr import run_swinunetr_inference

    device = torch.device(device) if isinstance(device, str) else device

    # ── Stack input in the order the rest of the pipeline expects ────────────
    # Pipeline convention: [FLAIR, T1, T1c, T2]
    # run_segresnet_inference handles its own permutation to [T1c, T1, T2, FLAIR]
    order = ["FLAIR", "T1", "T1c", "T2"]
    
    # Handle case-insensitive key lookup with fallback
    volume_map = {k.upper(): v for k, v in volumes.items()}
    logger.debug(f"Available volume keys: {list(volumes.keys())}")
    logger.debug(f"Volume map keys: {list(volume_map.keys())}")
    
    try:
        arrs = [volume_map[k.upper()].astype(np.float32) for k in order]
    except KeyError as e:
        logger.error(f"Missing volume key: {e}. Available keys: {list(volume_map.keys())}")
        raise
    stacked = np.stack(arrs, axis=0)  # [4, D, H, W]
    input_tensor = torch.from_numpy(stacked).unsqueeze(0)  # [1, 4, D, H, W]

    weights = config.get("ensemble_weights", {})
    ensemble_components: list[tuple[str, np.ndarray, float]] = []

    # ── SegResNet ────────────────────────────────────────────────────────────
    w_seg = float(weights.get("segresnet", 0.33))
    if w_seg > 0:
        logger.info(f"Running SegResNet (weight={w_seg:.2f})")
        try:
            model = load_segresnet(bundle_dir, device)
            sw_cfg = config.get("sliding_window", {})
            prob_seg = run_segresnet_inference(model, input_tensor, device, sw_cfg)
            ensemble_components.append(("segresnet", prob_seg, w_seg))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        except Exception as e:
            logger.error(f"SegResNet failed: {e}", exc_info=True)
            raise

    # ── TTA-4 ────────────────────────────────────────────────────────────────
    w_tta = float(weights.get("tta4", 0.33))
    if w_tta > 0:
        logger.info(f"Running TTA-4 (weight={w_tta:.2f})")
        try:
            model = load_segresnet(bundle_dir, device)
            sw_cfg = config.get("sliding_window", {})
            prob_tta = run_tta_ensemble(model, input_tensor, device, sw_cfg)
            ensemble_components.append(("tta4", prob_tta, w_tta))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        except Exception as e:
            logger.error(f"TTA-4 failed: {e}", exc_info=True)
            raise

    # ── SwinUNETR ────────────────────────────────────────────────────────────
    w_swin = float(weights.get("swinunetr", 0.34))
    if w_swin > 0:
        logger.info(f"Running SwinUNETr (weight={w_swin:.2f})")
        try:
            swin_cfg = config.get("swinunetr", {})
            prob_swin = run_swinunetr_inference(
                input_tensor, bundle_dir, device, model_cfg=swin_cfg
            )
            ensemble_components.append(("swinunetr", prob_swin, w_swin))
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        except Exception as e:
            logger.error(f"SwinUNETr failed: {e}", exc_info=True)
            raise

    if not ensemble_components:
        raise RuntimeError(
            "No ensemble component ran. Check ensemble_weights — at least one "
            "of segresnet/tta4/swinunetr must have weight > 0."
        )

    # ── Weighted average ─────────────────────────────────────────────────────
    total_w = sum(w for _, _, w in ensemble_components)
    if total_w <= 0:
        raise RuntimeError(f"Total ensemble weight is {total_w}; must be > 0.")
    
    weighted = sum(p * (w / total_w) for _, p, w in ensemble_components)
    # weighted shape: [3, D, H, W] for (TC, WT, ET) per segresnet convention
    
    tc_prob = weighted[0].astype(np.float32)
    wt_prob = weighted[1].astype(np.float32)
    et_prob = weighted[2].astype(np.float32)

    # ── Platt calibration ────────────────────────────────────────────────────
    cal_cfg = config.get("calibration", {})
    if cal_cfg.get("enabled", True):
        try:
            from pybrain.models.calibration import apply_platt_calibration
            wt_prob = apply_platt_calibration(wt_prob, "wt")
            tc_prob = apply_platt_calibration(tc_prob, "tc")
            et_prob = apply_platt_calibration(et_prob, "et")
            logger.info("Applied Platt calibration to ensemble probabilities")
        except ImportError:
            logger.warning("Platt calibration module not found; using raw probabilities")
        except Exception as e:
            logger.warning(f"Platt calibration failed: {e}; using raw probabilities")

    # ── Mask to brain ────────────────────────────────────────────────────────
    bm = brain_mask.astype(np.float32)
    wt_prob = wt_prob * bm
    tc_prob = tc_prob * bm
    et_prob = et_prob * bm

    return EnsembleProbabilities(wt_prob=wt_prob, tc_prob=tc_prob, et_prob=et_prob)
