#!/usr/bin/env python3
"""
Segmentation Validation Metrics
=================================
Computes Dice, HD95, Sensitivity, Specificity, and FP Volume
by comparing AI segmentation against a manual ground truth.

HOW TO CREATE THE GROUND TRUTH IN 3D SLICER:
  1. Open 3D Slicer
  2. Run load_in_slicer.py in the Python console to load everything
  3. In the left panel click on "BraTS Tumor Segmentation"
  4. Go to Modules → Segment Editor
  5. Select the "Paint" or "Draw" tool
  6. Paint over any regions the AI missed (radiologist described ~32 cc)
     Focus on:
       - The full 4.3 cm left fronto-parietal lesion
       - The calcified/dark regions CT confirmed
       - The ~1 cm right frontal lesion
  7. File → Save Data → save as:
       /path/to/PY-BRAIN/results/ground_truth.nii.gz

  Then run this script:
       python3 validate_segmentation.py

Requirements:
  pip install nibabel numpy scipy scikit-image
"""

import sys
import json
from pathlib import Path
# ── PY-BRAIN session loader ──────────────────────────────────────────
import sys as _sys
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.session_loader import get_session, get_paths, get_patient
try:
    _sess = get_session()
    _paths = get_paths(_sess)
    PATIENT       = get_patient(_sess)
    DICOM_MRI_DIR = _paths["mri_dicom_dir"]
    DICOM_CT_DIR  = _paths.get("ct_dicom_dir")
    NIFTI_DIR     = _paths.get("nifti_dir", _paths["monai_dir"].parent)
    MONAI_DIR     = _paths["monai_dir"]
    EXTRA_DIR     = _paths["extra_dir"]
    BUNDLE_DIR    = _paths["bundle_dir"]
    RESULTS_DIR   = _paths["results_dir"]
    GROUND_TRUTH  = _paths["ground_truth"]
    OUTPUT_DIR    = _paths.get("output_dir", RESULTS_DIR)
    
    DEVICE = "cpu"
    MODEL_DEVICE = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            MODEL_DEVICE = torch.device("cpu")
        elif torch.cuda.is_available():
            DEVICE = MODEL_DEVICE = torch.device("cuda")
        else:
            DEVICE = MODEL_DEVICE = torch.device("cpu")
    except ImportError:
        pass

    WT_THRESH, TC_THRESH, ET_THRESH = 0.30, 0.35, 0.40
    HU_CALCIFICATION_LOW, HU_CALCIFICATION_HIGH = 130, 1000
    HU_HAEMORRHAGE_LOW, HU_HAEMORRHAGE_HIGH = 50, 90
    HU_TUMOUR_LOW, HU_TUMOUR_HIGH = 25, 60
    LABEL_NAMES = {1: "Necrotic core", 2: "Edema", 3: "Enhancing tumor"}
    LABEL_COLORS = {1: "Blues", 2: "Greens", 3: "Reds"}
    LABEL_HEX = {1: "#4488ff", 2: "#44cc44", 3: "#ff4444"}
    LABEL_RGB = {1: [0.27, 0.53, 1.00], 2: [0.27, 0.87, 0.27], 3: [1.00, 0.27, 0.27]}
    MRI_SEQUENCE_MAP = {}
    CT_SEQUENCE_MAP = {}
except SystemExit:
    raise
except Exception as e:
    print(f"❌ Failed to load session: {e}")
    _sys.exit(1)
# ─────────────────────────────────────────────────────────────────────

from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────
# ⚙️  PATHS — update RESULTS_DIR to your latest run
# ─────────────────────────────────────────────────────────────────────────

RESULTS_BASE = RESULTS_DIR
GROUND_TRUTH = RESULTS_BASE / "ground_truth.nii.gz"

# Auto-find latest results folder
def find_latest_results(base: Path) -> Path:
    # 1) Try OUTPUT_DIR from session first
    if OUTPUT_DIR and Path(OUTPUT_DIR).is_dir():
        return Path(OUTPUT_DIR)
    # 2) Fall back to any subdirectory (sorted by name for recency)
    candidates = sorted(
        [d for d in base.iterdir()
         if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name
    )
    if not candidates:
        raise FileNotFoundError(f"No results folders found in {base}")
    return candidates[-1]

# ─────────────────────────────────────────────────────────────────────────

def banner(t):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


try:
    import numpy as np
    import nibabel as nib
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff
except ImportError as e:
    print(f"❌ Missing: {e}")
    print("Run: pip install nibabel numpy scipy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice Similarity Coefficient.
    1.0 = perfect overlap, 0.0 = no overlap.
    Clinical threshold for good segmentation: Dice > 0.80
    """
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)
    intersection = (pred_bool & gt_bool).sum()
    denom        = pred_bool.sum() + gt_bool.sum()
    if denom == 0:
        return 1.0 if pred_bool.sum() == 0 else 0.0
    return float(2 * intersection / denom)


def hausdorff_95(pred: np.ndarray, gt: np.ndarray,
                 voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    95th percentile Hausdorff Distance in mm.
    Measures worst-case boundary error (excluding top 5% outliers).
    Clinical threshold for good segmentation: HD95 < 5 mm
    """
    pred_pts = np.argwhere(pred > 0).astype(float)
    gt_pts   = np.argwhere(gt   > 0).astype(float)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    # Scale by voxel spacing to get mm distances
    pred_pts *= np.array(voxel_spacing)
    gt_pts   *= np.array(voxel_spacing)

    # Compute distances from pred→gt and gt→pred
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Sensitivity (Recall) = TP / (TP + FN)
    Fraction of true tumour that the AI found.
    1.0 = found everything, 0.5 = missed half
    """
    tp = (pred.astype(bool) & gt.astype(bool)).sum()
    fn = (~pred.astype(bool) & gt.astype(bool)).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Specificity = TN / (TN + FP)
    Fraction of non-tumour that was correctly left out.
    """
    tn = (~pred.astype(bool) & ~gt.astype(bool)).sum()
    fp = (pred.astype(bool) & ~gt.astype(bool)).sum()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def false_positive_volume(pred: np.ndarray, gt: np.ndarray,
                           vox_mm3: float = 1.0) -> float:
    """
    Volume of falsely segmented tissue in cc.
    Voxels the AI marked as tumour that are NOT in ground truth.
    """
    fp_voxels = (pred.astype(bool) & ~gt.astype(bool)).sum()
    return float(fp_voxels) * vox_mm3 / 1000.0


def false_negative_volume(pred: np.ndarray, gt: np.ndarray,
                           vox_mm3: float = 1.0) -> float:
    """
    Volume of missed tumour tissue in cc.
    Voxels in ground truth that the AI missed.
    """
    fn_voxels = (~pred.astype(bool) & gt.astype(bool)).sum()
    return float(fn_voxels) * vox_mm3 / 1000.0


def volume_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Volume Similarity = 1 - |Vpred - Vgt| / (Vpred + Vgt)
    1.0 = same volume, 0.0 = completely different volumes
    """
    vp = pred.astype(bool).sum()
    vg = gt.astype(bool).sum()
    if vp + vg == 0:
        return 1.0
    return float(1.0 - abs(vp - vg) / (vp + vg))


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray,
                         voxel_spacing: tuple,
                         name: str = "Segmentation") -> dict:
    """Compute all metrics for one pred/gt pair."""
    vox_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

    return {
        "name":                name,
        "dice":                round(float(dice_score(pred, gt)), 4),
        "hd95_mm":             round(float(hausdorff_95(pred, gt, voxel_spacing)), 2),
        "sensitivity":         round(float(sensitivity(pred, gt)), 4),
        "specificity":         round(float(specificity(pred, gt)), 4),
        "fp_volume_cc":        round(float(false_positive_volume(pred, gt, vox_mm3)), 2),
        "fn_volume_cc":        round(float(false_negative_volume(pred, gt, vox_mm3)), 2),
        "volume_similarity":   round(float(volume_similarity(pred, gt)), 4),
        "pred_volume_cc":      round(float(pred.astype(bool).sum()) * vox_mm3 / 1000, 2),
        "gt_volume_cc":        round(float(gt.astype(bool).sum())   * vox_mm3 / 1000, 2),
    }


def print_metrics(m: dict):
    """Pretty-print a metrics dict with clinical context."""

    def grade(metric, value, thresholds):
        """Return ✅ / ⚠️ / ❌ based on thresholds (good, warn)."""
        if value >= thresholds[0]: return "✅"
        if value >= thresholds[1]: return "⚠️ "
        return "❌"

    dice_g  = grade("dice",        m["dice"],        (0.80, 0.60))
    hd_g    = grade("hd95",  -m["hd95_mm"],    (-5,  -15))   # inverted: lower=better
    sens_g  = grade("sens",        m["sensitivity"], (0.80, 0.60))
    vs_g    = grade("vs",          m["volume_similarity"], (0.85, 0.70))

    print(f"\n  {'Metric':30s}  {'Value':>10}  {'Grade':>6}  Clinical threshold")
    print(f"  {'─'*30}  {'─'*10}  {'─'*6}  {'─'*30}")
    print(f"  {'Dice score':30s}  {m['dice']:>10.4f}  {dice_g}    > 0.80 = good")
    print(f"  {'HD95 (mm)':30s}  {m['hd95_mm']:>10.2f}  {hd_g}    < 5 mm = good")
    print(f"  {'Sensitivity (recall)':30s}  {m['sensitivity']:>10.4f}  {sens_g}    > 0.80 = good")
    print(f"  {'Specificity':30s}  {m['specificity']:>10.4f}  {'  '}    > 0.99 expected")
    print(f"  {'Volume similarity':30s}  {m['volume_similarity']:>10.4f}  {vs_g}    > 0.85 = good")
    print(f"  {'─'*30}  {'─'*10}")
    print(f"  {'Predicted volume (cc)':30s}  {m['pred_volume_cc']:>10.2f}")
    print(f"  {'Ground truth volume (cc)':30s}  {m['gt_volume_cc']:>10.2f}")
    print(f"  {'False positive volume (cc)':30s}  {m['fp_volume_cc']:>10.2f}  "
          f"     tissue wrongly added")
    print(f"  {'False negative volume (cc)':30s}  {m['fn_volume_cc']:>10.2f}  "
          f"     tumour missed by AI")


def interpret_results(metrics_list: list):
    """Print a plain-language summary of what the metrics mean."""
    print(f"\n  Plain language interpretation:")
    print(f"  {'─'*60}")
    for m in metrics_list:
        print(f"\n  [{m['name']}]")
        d = m['dice']
        s = m['sensitivity']
        h = m['hd95_mm']
        fp = m['fp_volume_cc']
        fn = m['fn_volume_cc']

        if d >= 0.80:
            print(f"  ✅ Dice {d:.2f} — good overlap with ground truth")
        elif d >= 0.60:
            print(f"  ⚠️  Dice {d:.2f} — moderate overlap, significant errors present")
        else:
            print(f"  ❌ Dice {d:.2f} — poor overlap, major corrections needed")

        if s >= 0.80:
            print(f"  ✅ Sensitivity {s:.2f} — AI found {s*100:.0f}% of the tumour")
        elif s >= 0.60:
            print(f"  ⚠️  Sensitivity {s:.2f} — AI missed {(1-s)*100:.0f}% of tumour "
                  f"({fn:.1f} cc)")
        else:
            print(f"  ❌ Sensitivity {s:.2f} — AI missed more than 40% of tumour "
                  f"({fn:.1f} cc) — manual correction essential")

        if h < 5:
            print(f"  ✅ HD95 {h:.1f} mm — boundary error under 5 mm")
        elif h < 15:
            print(f"  ⚠️  HD95 {h:.1f} mm — boundary off by up to {h:.0f} mm "
                  f"in worst areas")
        else:
            print(f"  ❌ HD95 {h:.1f} mm — large boundary errors, "
                  f"likely missing whole regions")

        if fp > 5:
            print(f"  ⚠️  FP volume {fp:.1f} cc — AI over-segmented, "
                  f"check for false positives")
        else:
            print(f"  ✅ FP volume {fp:.1f} cc — minimal over-segmentation")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
banner("SEGMENTATION VALIDATION METRICS")
print(f"  Ground truth: {GROUND_TRUTH}")

if not GROUND_TRUTH.exists():
    print(f"""
  ⚠️  Ground truth not found at:
      {GROUND_TRUTH}

  HOW TO CREATE IT IN 3D SLICER (30 min):
  ─────────────────────────────────────────
  1. Open 3D Slicer
  2. Run load_in_slicer.py in Python console
  3. Modules → Segment Editor
  4. Select the tumour segment
  5. Use Paint/Draw tools to correct the boundary:
       - Extend to cover the full 4.3 cm lesion
       - Include calcified dark regions (visible on CT)
       - Add the ~1 cm right frontal lesion as a new segment
  6. File → Save Data
     → Save as: {GROUND_TRUTH}

  Then re-run this script.
""")
    sys.exit(0)

# Load ground truth
try:
    latest_dir = find_latest_results(RESULTS_BASE)
    print(f"  Latest results: {latest_dir.name}\n")
except FileNotFoundError as e:
    print(f"  ❌ {e}")
    sys.exit(1)

gt_nib     = nib.load(str(GROUND_TRUTH))
gt_arr     = (gt_nib.get_fdata() > 0).astype(np.uint8)
vox_zoom   = gt_nib.header.get_zooms()[:3]
vox_mm3    = float(vox_zoom[0] * vox_zoom[1] * vox_zoom[2])

print(f"  Ground truth shape   : {gt_arr.shape}")
print(f"  Ground truth voxels  : {gt_arr.sum():,}")
print(f"  Ground truth volume  : {gt_arr.sum() * vox_mm3 / 1000:.1f} cc")
print(f"  Voxel spacing        : {tuple(round(float(v),2) for v in vox_zoom)} mm")

# Segmentations to evaluate
segs_to_evaluate = {
    "BraTS only":           latest_dir / "segmentation_full.nii.gz",
    "BraTS + CT merged":    latest_dir / "segmentation_ct_merged.nii.gz",
}

# Also check for previous runs to track improvement over time
prev_dirs = sorted([d for d in RESULTS_BASE.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                    and d != latest_dir])
if prev_dirs:
    prev_seg = prev_dirs[-1] / "segmentation_full.nii.gz"
    if prev_seg.exists():
        segs_to_evaluate["Previous run"] = prev_seg

all_metrics = []

banner("COMPUTING METRICS")

for seg_name, seg_path in segs_to_evaluate.items():
    if not seg_path.exists():
        print(f"\n  ⚠️  {seg_name}: file not found ({seg_path.name}) — skipping")
        continue

    print(f"\n  Computing metrics for: {seg_name}")
    seg_nib  = nib.load(str(seg_path))
    seg_arr  = (seg_nib.get_fdata() > 0).astype(np.uint8)

    # Match shapes if needed
    if seg_arr.shape != gt_arr.shape:
        print(f"  ⚠️  Shape mismatch: pred={seg_arr.shape} gt={gt_arr.shape}")
        min_s    = tuple(min(a,b) for a,b in zip(seg_arr.shape, gt_arr.shape))
        seg_arr  = seg_arr[:min_s[0], :min_s[1], :min_s[2]]
        gt_crop  = gt_arr[:min_s[0],  :min_s[1],  :min_s[2]]
    else:
        gt_crop = gt_arr

    print(f"  Running Dice…", end=" ", flush=True)
    m = compute_all_metrics(seg_arr, gt_crop, vox_zoom, seg_name)
    print(f"Dice={m['dice']:.4f}", end="  ", flush=True)
    print(f"HD95={m['hd95_mm']:.1f}mm", end="  ", flush=True)
    print(f"Sens={m['sensitivity']:.4f}")

    all_metrics.append(m)
    print_metrics(m)

# ── Sub-region metrics (if ground truth has labels) ───────────────────
gt_labels = np.unique(gt_nib.get_fdata().astype(np.uint8))
gt_labels = gt_labels[gt_labels > 0]

if len(gt_labels) > 1:
    banner("SUB-REGION METRICS")
    seg_full_path = latest_dir / "segmentation_full.nii.gz"
    if seg_full_path.exists():
        seg_full = nib.load(str(seg_full_path)).get_fdata().astype(np.uint8)
        gt_full  = gt_nib.get_fdata().astype(np.uint8)

        region_map = {1: "Necrotic core", 2: "Edema", 3: "Enhancing"}
        for label, name in region_map.items():
            if int(np.sum(gt_full == label)) == 0:
                continue
            pred_r = np.array(seg_full == label, dtype=np.uint8)
            gt_r   = np.array(gt_full  == label, dtype=np.uint8)
            m_r    = compute_all_metrics(pred_r, gt_r, vox_zoom, name)
            print(f"\n  {name}:")
            print(f"    Dice={m_r['dice']:.4f}  "
                  f"HD95={m_r['hd95_mm']:.1f}mm  "
                  f"Sens={m_r['sensitivity']:.4f}  "
                  f"Pred={m_r['pred_volume_cc']:.1f}cc  "
                  f"GT={m_r['gt_volume_cc']:.1f}cc")

# ── Interpretation ─────────────────────────────────────────────────────
if all_metrics:
    banner("INTERPRETATION")
    interpret_results(all_metrics)

    # Show improvement if multiple segmentations compared
    if len(all_metrics) >= 2:
        print(f"\n  Comparison table:")
        print(f"  {'Segmentation':25s}  {'Dice':>6}  {'HD95':>8}  "
              f"{'Sens':>6}  {'Vol(cc)':>8}  {'FP(cc)':>7}")
        print(f"  {'─'*25}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*7}")
        for m in all_metrics:
            print(f"  {m['name']:25s}  {m['dice']:>6.4f}  "
                  f"{m['hd95_mm']:>7.1f}mm  "
                  f"{m['sensitivity']:>6.4f}  "
                  f"{m['pred_volume_cc']:>7.1f}cc  "
                  f"{m['fp_volume_cc']:>6.1f}cc")

        # Check if CT merger actually improved things
        brats_m = next((m for m in all_metrics if "BraTS only" in m["name"]), None)
        ct_m    = next((m for m in all_metrics if "CT merged"  in m["name"]), None)
        if brats_m and ct_m:
            dice_delta = ct_m["dice"] - brats_m["dice"]
            sens_delta = ct_m["sensitivity"] - brats_m["sensitivity"]
            print(f"\n  CT integration effect:")
            print(f"    Dice change      : {dice_delta:+.4f}  "
                  f"({'improved' if dice_delta > 0 else 'worsened'})")
            print(f"    Sensitivity change: {sens_delta:+.4f}  "
                  f"({'improved' if sens_delta > 0 else 'worsened'})")
            if dice_delta < 0:
                print(f"    ⚠️  CT merger hurt Dice — CT added false positives")
                print(f"    Try increasing dilation radius in ct_integration.py")
            else:
                print(f"    ✅ CT merger improved segmentation quality")

# ── Save results JSON ──────────────────────────────────────────────────
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_json   = RESULTS_BASE / f"validation_{timestamp}.json"
with open(out_json, "w") as f:
    json.dump({
        "timestamp":     timestamp,
        "ground_truth":  str(GROUND_TRUTH),
        "results_dir":   str(latest_dir),
        "metrics":       all_metrics,
    }, f, indent=2)

banner("DONE")
print(f"""
  Validation report saved → {out_json}

  WHAT TO DO BASED ON RESULTS:

  Dice > 0.80 and HD95 < 5mm:
    → AI segmentation is clinically usable
    → Minor manual touch-up recommended

  Dice 0.60–0.80 or HD95 5–15mm:
    → Use AI as starting point in 3D Slicer
    → Manual correction of boundaries needed
    → Especially check enhancing rim

  Dice < 0.60 or HD95 > 15mm:
    → Major errors present
    → Re-run ct_integration.py if not done yet
    → Consider running Swin UNETR overnight
    → Manual correction essential before clinical use

  Sensitivity < 0.80:
    → AI missed significant tumour volume
    → Check prob_wt.nii.gz in 3D Slicer
    → Lower WT_THRESH further in brain_tumor_analysis.py

  FP volume > 5 cc:
    → AI over-segmented
    → Raise WT_THRESH or CT dilation radius
""")
