#!/usr/bin/env python3
"""
Validate BraTS-style segmentation against ground truth using Dice, HD95, ASD.
Can run in two modes:
  1. Session mode (default): reads pred/gt from current session paths,
     outputs validation_metrics.json
  2. CLI mode: --pred PATH --gt PATH (explicit file paths)

Usage (session mode):
    python3 scripts/5_validate_segmentation.py

Usage (CLI mode):
    python3 scripts/5_validate_segmentation.py --pred segmentation_full.nii.gz --gt ground_truth.nii.gz
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.core.labels import canonical_labels


# ─── Metrics ───────────────────────────────────────────────────────────────────


def dice_score(pred, gt, label):
    p = (pred == label).astype(np.float32)
    g = (gt == label).astype(np.float32)
    intersection = (p * g).sum()
    return float((2.0 * intersection) / (p.sum() + g.sum() + 1e-8))


def dice_per_label(pred, gt):
    """Dice for each BraTS label."""
    return {
        "necrotic": dice_score(pred, gt, 1),
        "edema": dice_score(pred, gt, 2),
        "enhancing": dice_score(pred, gt, 3),
    }


def dice_wt(pred, gt):
    """Whole tumour Dice (any label > 0)."""
    p = (pred > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    intersection = (p * g).sum()
    return float((2.0 * intersection) / (p.sum() + g.sum() + 1e-8))


def hd95(pred, gt, spacing=None):
    try:
        from monai.metrics import compute_hausdorff_distance

        if not pred.any() or not gt.any():
            return float("nan")
        p = pred.astype(np.float32)[None, None, ...]
        g = gt.astype(np.float32)[None, None, ...]
        return float(compute_hausdorff_distance(p, g, percentile=95).item())
    except Exception:
        return float("nan")


def asd(pred, gt, spacing=None):
    try:
        from monai.metrics import compute_average_surface_distance

        if not pred.any() or not gt.any():
            return float("nan")
        p = pred.astype(np.float32)[None, None, ...]
        g = gt.astype(np.float32)[None, None, ...]
        return float(compute_average_surface_distance(p, g).item())
    except Exception:
        return float("nan")


# ─── Session helpers ────────────────────────────────────────────────────────────


def _load_session():
    from pybrain.io.session import get_session, get_paths as _get_paths
    import os

    env_path = os.environ.get("PYBRAIN_SESSION")
    if env_path and Path(env_path).exists():
        with open(env_path) as f:
            json.load(f)  # warm cache
    sess = get_session()
    return sess, _get_paths


def _guess_gt(output_dir: Path) -> Optional[Path]:
    """Look for ground truth in standard locations."""
    candidates = [
        output_dir / "ground_truth.nii.gz",
        PROJECT_ROOT / "results" / "ground_truth.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ─── Core validation ───────────────────────────────────────────────────────────


def validate(pred_path: Path, gt_path: Path, out_dir: Optional[Path] = None) -> dict:
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))

    pred_img = pred_nii.get_fdata()
    gt_img = gt_nii.get_fdata()

    if pred_img.shape != gt_img.shape:
        raise ValueError(f"Shape mismatch: pred={pred_img.shape} gt={gt_img.shape}")

    # Normalise label conventions: BraTS 2021 GT uses ET=4; pipeline output uses ET=3.
    # Remap both to the pipeline convention (ET=3) before any metric computation.
    # Without this, ET Dice is always 0 when comparing pipeline output vs BraTS 2021 GT.
    pred_img = canonical_labels(pred_img)
    gt_img = canonical_labels(gt_img)

    # Voxel spacing for HD95/ASD (physical units in mm)
    spacing = tuple(float(x) for x in pred_nii.header.get_zooms()[:3])

    dice_lab = dice_per_label(pred_img, gt_img)
    d_wt = float(dice_wt(pred_img, gt_img))
    h_wt = float(hd95(pred_img > 0, gt_img > 0, spacing))
    a_wt = float(asd(pred_img > 0, gt_img > 0, spacing))

    # Volume comparison (cc)
    vox_cc = float(np.prod(spacing) / 1000.0)
    pred_vol = float((pred_img > 0).sum() * vox_cc)
    gt_vol = float((gt_img > 0).sum() * vox_cc)
    vol_diff_pct = float(abs(pred_vol - gt_vol) / (gt_vol + 1e-8) * 100)

    report = {
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "vox_size_mm": list(spacing),
        "vox_vol_cc": float(vox_cc),
        "dice": {
            "enhancing_tumour": float(dice_lab["enhancing"]),
            "necrotic_core": float(dice_lab["necrotic"]),
            "edema": float(dice_lab["edema"]),
            "whole_tumour": float(d_wt),
        },
        "hausdorff95_mm": round(float(h_wt), 2),
        "avg_surface_mm": round(float(a_wt), 2),
        "volume_cc": {
            "pred": round(float(pred_vol), 2),
            "gt": round(float(gt_vol), 2),
            "diff_pct": round(float(vol_diff_pct), 2),
        },
    }

    # Print report to console
    _print_report(report)
    return report


def _print_report(r: dict):
    dice = r["dice"]
    vol = r["volume_cc"]
    print(f"\n{'=' * 50}")
    print(f"{'SEGMENTATION VALIDATION REPORT':^50}")
    print(f"{'=' * 50}")
    print(f"  Pred : {r['pred_path']}")
    print(f"  GT   : {r['gt_path']}")
    print(f"  Voxel size : {r['vox_size_mm']} mm  ({r['vox_vol_cc']:.4f} cc/voxel)")
    print()
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 35}")
    print(f"  {'Dice ET (enhancing)':<25} {dice['enhancing_tumour']:>10.4f}")
    print(f"  {'Dice TC (necrotic core)':<25} {dice['necrotic_core']:>10.4f}")
    print(f"  {'Dice ED (edema)':<25} {dice['edema']:>10.4f}")
    print(f"  {'Dice WT (whole tumour)':<25} {dice['whole_tumour']:>10.4f}")
    print(f"  {'HD95 (whole tumour)':<25} {r['hausdorff95_mm']:>10.2f} mm")
    print(f"  {'ASD (avg surface dist)':<25} {r['avg_surface_mm']:>10.2f} mm")
    print()
    print(f"  Volume : pred={vol['pred']:.1f} cc  gt={vol['gt']:.1f} cc")
    print(f"            diff={vol['diff_pct']:.1f}%")
    print(f"{'=' * 50}")

    grade = dice["whole_tumour"]
    if grade >= 0.8:
        quality = "EXCELLENT"
    elif grade >= 0.6:
        quality = "GOOD"
    elif grade >= 0.4:
        quality = "FAIR"
    else:
        quality = "POOR"
    print(f"  Overall quality: {quality} (Dice WT = {grade:.4f})")
    print(f"{'=' * 50}\n")


# ─── CLI / session mode ─────────────────────────────────────────────────────────


def main():
    import os

    has_env_session = bool(os.environ.get("PYBRAIN_SESSION"))

    parser = argparse.ArgumentParser(description="BraTS Segmentation Validator")
    parser.add_argument("--pred", default=None, help="Predicted segmentation NIfTI path")
    parser.add_argument("--gt", default=None, help="Ground truth NIfTI path")
    args = parser.parse_args()

    output_dir = None
    pred_path = None
    gt_path = None

    # ── Session mode ─────────────────────────────────────────────────────────────
    if args.pred is None and args.gt is None and has_env_session:
        sess, get_paths = _load_session()
        paths = get_paths(sess)
        output_dir = Path(paths["output_dir"])

        pred_path = output_dir / "segmentation_full.nii.gz"
        gt_path = _guess_gt(output_dir)

        if not pred_path.exists():
            alt = output_dir / "segmentation_ensemble.nii.gz"
            if alt.exists():
                pred_path = alt

        if not pred_path.exists():
            print(f"❌ Predicted segmentation not found: {pred_path}")
            sys.exit(1)

        if gt_path is None:
            print("⚠️  Ground truth not found. Skipping validation.")
            print("   To enable: save your corrected mask as ground_truth.nii.gz")
            print(f"   Expected at: {output_dir / 'ground_truth.nii.gz'}")
            sys.exit(0)

        print(f"[INFO] Pred: {pred_path}")
        print(f"[INFO] GT  : {gt_path}")
        report = validate(pred_path, gt_path, out_dir=output_dir)

    # ── CLI explicit mode ───────────────────────────────────────────────────────
    elif args.pred is not None and args.gt is not None:
        pred_path = Path(args.pred)
        gt_path = Path(args.gt)
        if not pred_path.exists():
            print(f"❌ Predicted not found: {pred_path}")
            sys.exit(1)
        if not gt_path.exists():
            print(f"❌ Ground truth not found: {gt_path}")
            sys.exit(1)
        report = validate(pred_path, gt_path)

    else:
        print("❌ Provide both --pred and --gt, or run with PYBRAIN_SESSION set.")
        print("   Session mode: python3 scripts/5_validate_segmentation.py")
        print("   CLI mode:     python3 scripts/5_validate_segmentation.py --pred X.nii.gz --gt Y.nii.gz")
        sys.exit(1)

    # ── Save JSON report ──────────────────────────────────────────────────────────
    out_dir = output_dir if output_dir else pred_path.parent
    out_path = out_dir / "validation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Report saved → {out_path}")


if __name__ == "__main__":
    main()
