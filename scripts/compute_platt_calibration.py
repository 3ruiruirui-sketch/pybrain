#!/usr/bin/env python3
"""
Platt calibration + evidence-based ensemble weights from BraTS 2021.

Solves Problems A and B from the pipeline reliability audit:
  A) Fit per-channel Platt scaling coefficients from real GT comparisons.
  B) Measure per-model Dice scores and compute optimal ensemble weights.

Usage:
    python scripts/compute_platt_calibration.py \\
        --brats_dir  data/datasets/BraTS2021/raw/BraTS2021_Training_Data \\
        --bundle_dir models/brats_bundle \\
        --n_cases    50 \\
        --device     cpu \\
        --out_dir    models/calibration

Outputs:
    models/calibration/platt_coefficients.json   → used by apply_platt_calibration()
    models/calibration/ensemble_weights.json     → review and copy to defaults.yaml
    models/calibration/per_model_dice.json       → per-case, per-model, per-subregion Dice
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# ── Resolve project root so we can import pybrain ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pybrain.io.logging_utils import setup_logging, get_logger
from pybrain.models.segresnet import load_segresnet, run_segresnet_inference
from pybrain.models.swinunetr import run_swinunetr_inference
from pybrain.models.ensemble import run_weighted_ensemble
from pybrain.core.normalization import zscore_robust


# =============================================================================
# BraTS label → subregion binary mask
# =============================================================================

def brats_seg_to_channels(seg: np.ndarray) -> np.ndarray:
    """
    Convert a BraTS 2021 label map to 3-channel binary ground truth.

    BraTS labels:
        0 = background
        1 = necrotic core   (NCR)
        2 = peritumoral edema (ED)
        4 = enhancing tumor  (ET)   — note: label 3 absent in BraTS 2021

    Channel order matches the pipeline: [TC, WT, ET]
        TC (Tumor Core)      = labels 1 + 4
        WT (Whole Tumor)     = labels 1 + 2 + 4
        ET (Enhancing Tumor) = label 4
    """
    tc = ((seg == 1) | (seg == 4)).astype(np.float32)
    wt = ((seg == 1) | (seg == 2) | (seg == 4)).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([tc, wt, et], axis=0)   # (3, D, H, W)


# =============================================================================
# Preprocessing — mirrors preprocess_volumes() in the pipeline
# =============================================================================

def make_brain_mask(volumes: Dict[str, np.ndarray]) -> np.ndarray:
    """Non-zero union across all modalities."""
    mask = np.zeros(next(iter(volumes.values())).shape, dtype=bool)
    for v in volumes.values():
        mask |= (v > 0)
    return mask.astype(np.float32)


def preprocess_brats_case(case_dir: Path) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Load one BraTS case and return:
        input_tensor : (1, 4, D, H, W)  float32
        brain_mask   : (D, H, W)         float32
        gt_channels  : (3, D, H, W)      float32  [TC, WT, ET]
    """
    cid = case_dir.name
    volumes = {
        "FLAIR": nib.load(str(case_dir / f"{cid}_flair.nii.gz")).get_fdata().astype(np.float32),  # type: ignore[union-attr]
        "T1":    nib.load(str(case_dir / f"{cid}_t1.nii.gz")).get_fdata().astype(np.float32),    # type: ignore[union-attr]
        "T1c":   nib.load(str(case_dir / f"{cid}_t1ce.nii.gz")).get_fdata().astype(np.float32),  # type: ignore[union-attr]
        "T2":    nib.load(str(case_dir / f"{cid}_t2.nii.gz")).get_fdata().astype(np.float32),    # type: ignore[union-attr]
    }
    seg = nib.load(str(case_dir / f"{cid}_seg.nii.gz")).get_fdata().astype(np.int32)  # type: ignore[union-attr]

    brain_mask = make_brain_mask(volumes)

    norm_vols = {}
    for name, vol in volumes.items():
        norm_vols[name] = zscore_robust(vol, brain_mask)

    stacked = np.stack(
        [norm_vols["FLAIR"], norm_vols["T1"], norm_vols["T1c"], norm_vols["T2"]],
        axis=0,
    )
    stacked = stacked * brain_mask.astype(np.float32)
    input_tensor = torch.from_numpy(stacked).unsqueeze(0)

    gt_channels = brats_seg_to_channels(seg)
    return input_tensor, brain_mask, gt_channels


# =============================================================================
# Dice score
# =============================================================================

def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Binary Dice coefficient. Returns 1.0 if both are empty."""
    intersection = float((pred_bin * gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    if denom < 1e-6:
        return 1.0
    return 2.0 * intersection / denom


# =============================================================================
# Per-model inference (no ensemble)
# =============================================================================

def run_single_model(
    model_name: str,
    input_tensor: torch.Tensor,
    model_device: torch.device,
    bundle_dir: Path,
    model_cfg: dict,
) -> Optional[np.ndarray]:
    """
    Run one model and return prob map (3, D, H, W) or None on failure.
    """
    logger = get_logger("calibration")
    try:
        if model_name == "segresnet":
            model = load_segresnet(bundle_dir, model_device)
            prob = run_segresnet_inference(
                model, input_tensor, model_device, model_cfg, model_device=model_device
            )
            del model
            gc.collect()
            return prob

        if model_name == "swinunetr":
            prob = run_swinunetr_inference(
                input_tensor, bundle_dir, model_device, model_cfg=model_cfg
            )
            return prob

    except Exception as exc:
        logger.warning(f"  {model_name} failed: {exc}")
        return None

    return None


# =============================================================================
# Main calibration loop
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Platt calibration + ensemble weights")
    parser.add_argument("--brats_dir",  required=True,  type=Path)
    parser.add_argument("--bundle_dir", required=True,  type=Path)
    parser.add_argument("--n_cases",    default=50,     type=int,
                        help="Number of cases to process (max 1251). 50 is sufficient.")
    parser.add_argument("--device",     default="cpu",  type=str)
    parser.add_argument("--out_dir",    default="models/calibration", type=Path)
    parser.add_argument("--seed",       default=42,     type=int)
    parser.add_argument("--skip_swinunetr", action="store_true",
                        help="Skip SwinUNETR inference (much faster; Platt fitted on SegResNet only). "
                             "Use when SwinUNETR Dice on full volumes is low or inference is too slow.")
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("calibration")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Collect all case directories ──────────────────────────────────────────
    all_cases = sorted([
        p for p in args.brats_dir.iterdir()
        if p.is_dir() and (p / f"{p.name}_seg.nii.gz").exists()
    ])
    logger.info(f"Found {len(all_cases)} valid BraTS cases")

    random.seed(args.seed)
    cases = random.sample(all_cases, min(args.n_cases, len(all_cases)))
    logger.info(f"Selected {len(cases)} cases (seed={args.seed})")

    # ── Model configs ─────────────────────────────────────────────────────────
    sr_cfg = {"roi_size": [240, 240, 160], "overlap": 0.5}
    sw_cfg = {
        "weights": "fold1_swin_unetr.pth",
        "roi_size": [128, 128, 128],
        "overlap": 0.5,
        "bundle_dir": str(args.bundle_dir),
    }
    ensemble_weights = {"segresnet": 0.4, "swinunetr": 0.3, "tta4": 0.3}
    channel_names = ["tc", "wt", "et"]

    # Accumulators for Platt fitting
    # For each channel: list of (flat_prob, flat_gt) tuples across all cases
    all_probs: Dict[str, List[np.ndarray]] = {c: [] for c in channel_names}
    all_gts:   Dict[str, List[np.ndarray]] = {c: [] for c in channel_names}

    # Accumulators for per-model Dice
    dice_records: List[dict] = []
    active_models = ["segresnet", "ensemble"] if args.skip_swinunetr else ["segresnet", "swinunetr", "ensemble"]
    per_model_dice: Dict[str, Dict[str, List[float]]] = {
        m: {c: [] for c in channel_names}
        for m in active_models
    }
    if args.skip_swinunetr:
        logger.info("--skip_swinunetr: SwinUNETR inference will be skipped. "
                    "Platt coefficients fitted on SegResNet ensemble only.")

    # ── Per-case loop ─────────────────────────────────────────────────────────
    for case_idx, case_dir in enumerate(cases):
        cid = case_dir.name
        logger.info(f"\n[{case_idx+1}/{len(cases)}] {cid}")

        try:
            input_tensor, brain_mask, gt_channels = preprocess_brats_case(case_dir)
        except Exception as exc:
            logger.warning(f"  Preprocessing failed: {exc} — skipping")
            continue

        input_tensor = input_tensor.to(device)
        model_probs: Dict[str, np.ndarray] = {}
        sw_prob = None

        # ── SegResNet ─────────────────────────────────────────────────────────
        logger.info("  Running SegResNet...")
        sr_prob = run_single_model("segresnet", input_tensor, device, args.bundle_dir, sr_cfg)
        if sr_prob is not None:
            model_probs["segresnet"] = sr_prob

        # ── SwinUNETR ─────────────────────────────────────────────────────────
        if not args.skip_swinunetr:
            logger.info("  Running SwinUNETR...")
            sw_prob = run_single_model("swinunetr", input_tensor, device, args.bundle_dir, sw_cfg)
            if sw_prob is not None:
                model_probs["swinunetr"] = sw_prob
        else:
            logger.info("  SwinUNETR skipped (--skip_swinunetr)")

        if not model_probs:
            logger.warning("  No models succeeded — skipping case")
            continue

        # ── Ensemble ──────────────────────────────────────────────────────────
        model_list = [
            (name, prob, ensemble_weights.get(name, 1.0))
            for name, prob in model_probs.items()
        ]
        ensemble_prob, _ = run_weighted_ensemble(model_list)

        # ── Measure Dice per model per channel ────────────────────────────────
        case_dice: dict = {"case": cid}
        for ch_idx, ch_name in enumerate(channel_names):
            gt_bin = gt_channels[ch_idx]
            for model_name, prob_map in list(model_probs.items()) + [("ensemble", ensemble_prob)]:
                prob_ch = prob_map[ch_idx]
                pred_bin = (prob_ch > 0.5).astype(np.float32)
                d = dice_score(pred_bin, gt_bin)
                per_model_dice[model_name][ch_name].append(d)
                case_dice[f"{model_name}_{ch_name}_dice"] = round(d, 4)

        dice_records.append(case_dice)
        logger.info(
            f"  Dice — SR: WT={case_dice.get('segresnet_wt_dice','?'):.3f} "
            f"TC={case_dice.get('segresnet_tc_dice','?'):.3f} "
            f"ET={case_dice.get('segresnet_et_dice','?'):.3f} | "
            f"ENS: WT={case_dice.get('ensemble_wt_dice','?'):.3f} "
            f"TC={case_dice.get('ensemble_tc_dice','?'):.3f} "
            f"ET={case_dice.get('ensemble_et_dice','?'):.3f}"
        )

        # ── Collect probability + GT for Platt fitting ────────────────────────
        # Sample at most 50k voxels per case to keep memory manageable
        flat_mask = brain_mask.astype(bool).ravel()
        n_voxels  = flat_mask.sum()
        max_vox   = 50_000
        if n_voxels > max_vox:
            idx = np.where(flat_mask)[0]
            chosen = np.random.choice(idx, size=max_vox, replace=False)
        else:
            chosen = np.where(flat_mask)[0]

        for ch_idx, ch_name in enumerate(channel_names):
            p_flat  = ensemble_prob[ch_idx].ravel()[chosen]
            gt_flat = gt_channels[ch_idx].ravel()[chosen]
            all_probs[ch_name].append(p_flat)
            all_gts[ch_name].append(gt_flat)

        # Free GPU memory
        del input_tensor, sr_prob, sw_prob, ensemble_prob
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Fit Platt calibration ─────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Fitting Platt calibration...")
    platt_coefficients: dict = {}

    for ch_name in channel_names:
        p_all  = np.concatenate(all_probs[ch_name])
        gt_all = np.concatenate(all_gts[ch_name])

        # Need both positive and negative examples
        n_pos = int(gt_all.sum())
        n_neg = int((1 - gt_all).sum())
        logger.info(f"  {ch_name.upper()}: {len(p_all)} voxels  pos={n_pos}  neg={n_neg}")

        if n_pos < 100 or n_neg < 100:
            logger.warning(f"  {ch_name.upper()}: Too few examples — skipping calibration for this channel")
            platt_coefficients[ch_name] = {"A": 1.0, "B": 0.0, "note": "insufficient_data"}
            continue

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(p_all.reshape(-1, 1), gt_all)
        A = float(clf.coef_[0][0])  # type: ignore[index]
        B = float(clf.intercept_[0])
        platt_coefficients[ch_name] = {"A": A, "B": B}
        logger.info(f"  {ch_name.upper()}: A={A:.4f}  B={B:.4f}")

    # ── Compute optimal ensemble weights from Dice scores ────────────────────
    logger.info("\nComputing ensemble weights from Dice scores...")
    mean_dice: Dict[str, Dict[str, float]] = {}
    for model_name in per_model_dice:
        mean_dice[model_name] = {}
        for ch_name in channel_names:
            scores = per_model_dice[model_name][ch_name]
            if scores:
                mean_dice[model_name][ch_name] = float(np.mean(scores))
                logger.info(
                    f"  {model_name:12s} {ch_name.upper()}: "
                    f"mean Dice = {mean_dice[model_name][ch_name]:.4f}  "
                    f"(n={len(scores)})"
                )

    # Overall weight = mean Dice across all 3 channels
    model_overall: Dict[str, float] = {}
    measured_models = ["segresnet"] if args.skip_swinunetr else ["segresnet", "swinunetr"]
    for model_name in measured_models:
        if model_name in mean_dice:
            scores_all = [mean_dice[model_name][c] for c in channel_names if c in mean_dice[model_name]]
            model_overall[model_name] = float(np.mean(scores_all)) if scores_all else 0.0

    total = sum(model_overall.values()) + 1e-8
    normalised_weights = {m: round(v / total, 4) for m, v in model_overall.items()}
    # TTA-4 not run independently here — assign same weight as SegResNet as a safe default
    normalised_weights["tta4"] = normalised_weights.get("segresnet", 0.33)
    if args.skip_swinunetr:
        # SwinUNETR not measured — set to 0.0 pending a full measurement run
        normalised_weights["swinunetr"] = 0.0
        logger.info("swinunetr weight set to 0.0 (not measured — rerun without --skip_swinunetr to measure)")
    # Re-normalise
    total2 = sum(normalised_weights.values()) + 1e-8
    normalised_weights = {m: round(v / total2, 4) for m, v in normalised_weights.items()}
    normalised_weights["nnunet"] = 0.0   # Always 0 until weights provided

    logger.info(f"\nRecommended ensemble_weights: {normalised_weights}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    platt_path = args.out_dir / "platt_coefficients.json"
    with open(platt_path, "w") as f:
        json.dump(platt_coefficients, f, indent=2)
    logger.info(f"\nPlatt coefficients saved → {platt_path}")
    logger.info(f"  Copy to: {args.bundle_dir.parent}/platt_coefficients.json")
    logger.info(f"  (apply_platt_calibration reads from output_dir.parent/platt_coefficients.json)")

    weights_path = args.out_dir / "ensemble_weights.json"
    with open(weights_path, "w") as f:
        json.dump({
            "recommended_ensemble_weights": normalised_weights,
            "mean_dice_per_model": mean_dice,
            "n_cases": len(dice_records),
        }, f, indent=2)
    logger.info(f"Ensemble weights saved → {weights_path}")

    dice_path = args.out_dir / "per_model_dice.json"
    with open(dice_path, "w") as f:
        json.dump(dice_records, f, indent=2)
    logger.info(f"Per-case Dice records saved → {dice_path}")

    # ── Print final summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"\nCases processed: {len(dice_records)}")
    print("\nPlatt coefficients:")
    for ch, coeff in platt_coefficients.items():
        print(f"  {ch.upper()}: A={coeff.get('A','-'):.4f}  B={coeff.get('B','-'):.4f}")
    print("\nRecommended ensemble_weights (copy to defaults.yaml):")
    for m, w in normalised_weights.items():
        print(f"  {m}: {w}")
    print("\nNext steps:")
    print(f"  1. Copy {platt_path}")
    print(f"     → {args.bundle_dir.parent}/platt_coefficients.json")
    print(f"  2. Update ensemble_weights in config/defaults.yaml")
    print(f"  3. Re-run regression suite: python tests/regression_baseline.py --device cpu")
    print("="*60)


if __name__ == "__main__":
    main()
