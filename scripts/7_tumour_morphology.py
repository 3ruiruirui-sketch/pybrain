#!/usr/bin/env python3
"""
Tumour Morphology Analysis
===========================
Computes detailed 3D morphological measurements:
  - Sphericity, convexity, elongation, flatness
  - Surface area and roughness
  - Necrosis fraction, enhancement ratio
  - Oedema-to-tumour ratio
  - T2-FLAIR mismatch score
  - Enhancement kinetics proxy

Requirements: pip install nibabel numpy scipy matplotlib

Run: python3 tumour_morphology.py
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# ── pybrain Imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from pybrain.io.session import get_session, get_paths, get_patient # type: ignore
    from pybrain.io.config import get_config # type: ignore
    from pybrain.io.logging_utils import setup_logging # type: ignore
    
    _sess = get_session()
    _paths = get_paths(_sess)
    _config = get_config()
    _pat_loaded = get_patient(_sess)
    PATIENT: Any = _pat_loaded
    
    OUTPUT_DIR = _paths["output_dir"]
    MONAI_DIR = _paths["monai_dir"]
    RESULTS_DIR = _paths["results_dir"]
    EXTRA_DIR = _paths["extra_dir"]
    
    logger = setup_logging(OUTPUT_DIR)
    logger.info("Stage 7 — Tumour Morphology Analysis — Initialized")

except Exception as e:
    print(f"❌ Failed to load session: {e}")
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

RESULTS_BASE = RESULTS_DIR
MRI_DIR      = Path(str(MONAI_DIR))

try:
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore
    from scipy import ndimage  # type: ignore
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.gridspec as gs  # type: ignore
except ImportError as e:
    print(f"❌ {e}\nRun: pip install nibabel numpy scipy matplotlib")
    sys.exit(1)

def banner(t):
    print("\n" + "═"*60)
    print(f"  {t}")
    print("═"*60)

def _r(v: float, n: int) -> float:
    """Round a numeric value. Typed wrapper to satisfy Pyre."""
    m: float = 10.0 ** n
    return int(float(v) * m + 0.5) / m

# ─────────────────────────────────────────────────────────────────────────

banner("TUMOUR MORPHOLOGY ANALYSIS")

# Load segmentation quality check
qual_path = OUTPUT_DIR / "segmentation_quality.json"
if qual_path.exists():
    with open(qual_path) as f:
        q_data = json.load(f)
        qual = q_data.get("quality", q_data)
        vol = qual.get("v_wt_cc", qual.get("tumour_volume_cc", 0))
        if not qual.get("tumour_inside_brain", True) or vol < 0.01:
            print("❌ Fatal: Segmentation quality insufficient for morphology analysis.")
            print(f"   Tumor volume: {vol:.3f}cc (minimum: 0.01cc)")
            if vol < 0.01:
                print("   Note: Very small tumor detected. Consider adjusting segmentation parameters.")
            sys.exit(1)

# Load segmentation from current session output directory (Priority: Ensemble > Merged > Full)
target_names = ["segmentation_ensemble.nii.gz", "segmentation_ct_merged.nii.gz", "segmentation_full.nii.gz"]
seg_path = None
for name in target_names:
    p = OUTPUT_DIR / name
    if p.exists():
        seg_path = p
        break

if seg_path is None: raise ValueError("Segmentation path is None")
_seg_p: Any = cast(Any, seg_path)
print(f"  Using segmentation: {getattr(_seg_p, 'name')}")

seg_nib   = nib.load(str(seg_path))
seg_arr   = seg_nib.get_fdata().astype(np.uint8)
vox       = seg_nib.header.get_zooms()[:3]
vox_mm3   = float(vox[0]*vox[1]*vox[2])

# ── NIfTI Orientation Diagnostic ──────────────────────────────────────
# Print the actual orientation so spatial filtering can be validated
try:
    orient_codes = nib.aff2axcodes(seg_nib.affine)
    print(f"  NIfTI orientation: {orient_codes}  (axis0={orient_codes[0]}, axis1={orient_codes[1]}, axis2={orient_codes[2]})")
except Exception:
    orient_codes = ('?', '?', '?')
    print("  NIfTI orientation: could not determine")

# ── Component Analysis (Patch v7.7 — Largest Component) ──────────────
# Two masks:
#   whole_arr         = ALL components > 0.5cc (for volume sums)
#   largest_comp_arr  = SINGLE largest component (for shape/diameter)
import skimage.measure as _m
labeled, ncomp = ndimage.label(seg_arr > 0)
min_voxels_cc = int(0.5 / (vox_mm3 / 1000.0 + 1e-8))

if ncomp > 0:
    props_list = _m.regionprops(labeled)
    
    # Print detailed component table for diagnostics
    print(f"\n  Connected components: {ncomp}")
    for p in props_list:
        vol_cc = p.area * vox_mm3 / 1000.0
        cx, cy, cz = p.centroid
        bb = p.bbox
        ext = [abs(bb[i+3]-bb[i]) * float(vox[i]) for i in range(3)]
        diam = max(ext)
        marker = " ◀ LARGEST" if p.area == max(pp.area for pp in props_list) else ""
        if vol_cc >= 0.1:
            print(f"    Comp {p.label}: {vol_cc:.2f}cc, centroid=({cx:.0f},{cy:.0f},{cz:.0f}), "
                  f"diam={diam:.0f}mm{marker}")
    
    # whole_arr: keep all components above clinical significance (0.5cc)
    sizes = np.array([p.area for p in props_list])
    valid_labels = [p.label for p in props_list if p.area >= min_voxels_cc]
    if len(valid_labels) > 0:
        whole_arr = np.isin(labeled, valid_labels)
    else:
        # Fallback: keep the single largest
        largest_label = props_list[int(np.argmax(sizes))].label
        whole_arr = (labeled == largest_label)
    
    # largest_comp_arr: the single largest solid mass (for shape metrics)
    largest_label = props_list[int(np.argmax(sizes))].label
    largest_comp_arr = (labeled == largest_label).astype(bool)
    largest_vol_cc = sizes[int(np.argmax(sizes))] * vox_mm3 / 1000.0
    print(f"\n  Largest component: label {largest_label}, {largest_vol_cc:.2f} cc")
    print(f"  Total filtered volume (>0.5cc): {whole_arr.sum() * vox_mm3 / 1000:.2f} cc")
else:
    whole_arr = (seg_arr > 0)
    largest_comp_arr = whole_arr.copy()

whole: Any = cast(Any, whole_arr)

def load_vol(path):
    if not Path(path).exists():
        return None
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    if arr.shape != seg_arr.shape:
        ms  = tuple(min(a,b) for a,b in zip(arr.shape, seg_arr.shape))
        arr = arr[:ms[0],:ms[1],:ms[2]]
    return arr

t1    = load_vol(MRI_DIR/"t1_resampled.nii.gz")
if t1 is None: t1 = load_vol(MRI_DIR/"t1.nii.gz")
t1c   = load_vol(MRI_DIR/"t1c_resampled.nii.gz")
t2    = load_vol(MRI_DIR/"t2_resampled.nii.gz")
flair = load_vol(MRI_DIR/"flair_resampled.nii.gz")

banner("MORPHOLOGY METRICS")

results: dict = {}

# ── Volumes ───────────────────────────────────────────────────────────
_whole_a: Any = cast(Any, whole)
# Apply the hemisphere filter back to the original segmentation array 
# so edema/enhancing/necrotic calculations don't include discarded contralateral noise.
seg_arr = seg_arr * whole_arr

_seg_arr_a: Any = cast(Any, seg_arr)
n_whole = int(np.sum(_whole_a))
n_ncr   = int(np.sum(_seg_arr_a == 1))
n_ed    = int(np.sum(seg_arr == 2))
n_et    = int(np.sum(seg_arr == 3))
n_tc    = n_ncr + n_et

vol_whole = n_whole * vox_mm3 / 1000
vol_ncr   = n_ncr   * vox_mm3 / 1000
vol_ed    = n_ed    * vox_mm3 / 1000
vol_et    = n_et    * vox_mm3 / 1000
vol_tc    = n_tc    * vox_mm3 / 1000

# Use the brain mask generated in Stage 3 if available
brain_mask_path = OUTPUT_DIR / "brain_mask.nii.gz"
if brain_mask_path.exists():
    brain_mask_nib = nib.load(str(brain_mask_path))
    brain_mask_arr = brain_mask_nib.get_fdata()
    vol_brain = float(np.sum(brain_mask_arr > 0)) * vox_mm3 / 1000.0
else:
    # Conservative fallback
    if t1 is not None:
        # Patch v7.5: Smooth Extraction
        p1, p99 = np.percentile(t1, 1), np.percentile(t1, 99)
        t1_norm = np.clip((t1 - p1) / (p99 - p1 + 1e-8), 0, 1)
        brain_raw = ndimage.binary_fill_holes(t1_norm > 0.08)
        brain_raw = ndimage.binary_erosion(brain_raw, iterations=1)
        vol_brain = float(brain_raw.sum()) * vox_mm3 / 1000.0
    else:
        vol_brain = 0.0

results["volumes"] = {
    "whole_cc":    _r(vol_whole, 2), 
    "necrotic_cc": _r(vol_ncr, 2),
    "edema_cc":    _r(vol_ed, 2),
    "enhancing_cc": _r(vol_et, 2),
    "core_cc":     _r(vol_tc, 2),
    "brain_cc":    _r(vol_brain, 2),
}
print(f"\n  Whole tumour   : {vol_whole:.1f} cc")
print(f"  Tumour core    : {vol_tc:.1f} cc  (NCR + ET)")
print(f"  Necrotic core  : {vol_ncr:.1f} cc")
print(f"  Edema          : {vol_ed:.1f} cc")
print(f"  Enhancing      : {vol_et:.1f} cc")
print(f"  Brain volume   : {vol_brain:.1f} cc (parenchyma only)")

# ── Clinically important ratios ───────────────────────────────────────
ncr_frac = n_ncr / (n_whole + 1)
et_frac  = n_et  / (n_whole + 1)
ed_frac  = n_ed  / (n_whole + 1)
ed_core  = n_ed  / (n_tc + 1)
ncr_core = n_ncr / (n_tc + 1)

results["clinical_ratios"] = {
    "necrosis_fraction":    _r(ncr_frac, 4),
    "enhancement_fraction": _r(et_frac, 4),
    "edema_fraction":       _r(ed_frac, 4),
    "edema_to_core_ratio":  _r(ed_core, 4),
    "necrosis_to_core":     _r(ncr_core, 4),
}
print(f"\n  Necrosis fraction    : {ncr_frac*100:.1f}%")
print(f"  Enhancement fraction : {et_frac*100:.1f}%")
print(f"  Edema fraction       : {ed_frac*100:.1f}%")
print(f"  Edema / core ratio   : {ed_core:.2f}")

# ── Shape analysis (measured on LARGEST COMPONENT only) ───────────────
# Using the single largest solid mass prevents scattered fragments
# from inflating diameter/bounding box measurements.
MIN_VOXELS = 500   # skip shape metrics on noise / tiny fragments
n_largest = int(np.sum(largest_comp_arr))

if n_largest >= MIN_VOXELS:
    coords = np.argwhere(largest_comp_arr)
    x,y,z  = coords[:,0], coords[:,1], coords[:,2]
    bb_x   = (x.max()-x.min())*float(vox[0])
    bb_y   = (y.max()-y.min())*float(vox[1])
    bb_z   = (z.max()-z.min())*float(vox[2])

    # Surface area (Marching Cubes method for high precision)
    vol_mm3  = n_largest * vox_mm3
    eroded   = ndimage.binary_erosion(largest_comp_arr)
    surface  = largest_comp_arr & ~eroded
    try:
        import skimage.measure as _m # type: ignore
        verts, faces, _, _ = _m.marching_cubes(largest_comp_arr.astype(np.uint8), level=0.5, spacing=vox)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        surf_mm2 = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
    except (ValueError, RuntimeError):
        # Fallback to erosion if meshing fails
        n_surf  = int(np.sum(surface))
        surf_mm2 = n_surf * (vox_mm3**(2/3))

    # Sphericity
    sphericity = (np.pi**(1/3) * (6*vol_mm3)**(2/3)) / (surf_mm2 + 1e-8)
    sphericity = min(float(sphericity), 1.0)
    
    # Compactness (Area^1.5 / Volume)
    compactness = (surf_mm2 ** 1.5) / (vol_mm3 + 1e-8)

    # Fill internal holes for the volume calculation
    whole_comp = ndimage.binary_fill_holes(largest_comp_arr)
    convexity = float(n_largest / (np.sum(whole_comp) + 1e-8))

    # Dimensions
    dims      = sorted([bb_x,bb_y,bb_z])
    elongation = float(dims[0]/(dims[2]+1e-8))
    flatness   = float(dims[0]/(dims[1]+1e-8))

    # Surface roughness (std of local surface curvature proxy)
    surf_coords = np.argwhere(surface).astype(float)
    roughness   = 0.0
    if len(surf_coords) > 10:
        from scipy.spatial import cKDTree  # type: ignore
        tree  = cKDTree(surf_coords * np.array(vox))
        dists,_ = tree.query(surf_coords * np.array(vox), k=min(7,len(surf_coords)))
        roughness = float(dists[:,1:].std())

    results["shape"] = {
        "sphericity":       _r(sphericity, 4),
        "convexity":        _r(convexity, 4),
        "compactness":     _r(compactness, 4),
        "elongation":       _r(elongation, 4),
        "flatness":         _r(flatness, 4),
        "surface_area_mm2": _r(surf_mm2, 1),
        "surface_roughness":_r(roughness, 3),
        "max_diameter_mm":  _r(max(bb_x, bb_y, bb_z), 1),
        "bbox_x_mm":        _r(bb_x, 1),
        "bbox_y_mm":        _r(bb_y, 1),
        "bbox_z_mm":        _r(bb_z, 1),
    }
    print(f"\n  Sphericity     : {sphericity:.4f}  (1=perfect sphere)")
    print(f"  Convexity      : {convexity:.4f}   (1=fully convex)")
    print(f"  Elongation     : {elongation:.4f}")
    print(f"  Max diameter   : {max(bb_x,bb_y,bb_z):.1f} mm")
    print(f"  Surface area   : {surf_mm2:.0f} mm2")
    print(f"  Roughness      : {roughness:.3f} mm")
else:
    print(f"\n  ⚠️  Largest component too small ({n_largest} voxels < {MIN_VOXELS}) —"
          f" shape metrics skipped (likely fragmented segmentation).")
    sphericity = convexity = elongation = flatness = roughness = 0.0
    surface = np.zeros_like(largest_comp_arr)
    results["shape"] = {"skip_reason": f"mask_too_small_{n_largest}_voxels"}

# ── Enhancement ratio (MEDIAN for GBM robustness) ────────────────────
if t1 is not None and t1c is not None:
    mask_b   = np.asarray(whole).astype(bool)
    # Use median instead of mean: more robust in heterogeneous GBM
    t1_med   = float(np.median(t1[mask_b]))
    t1c_med  = float(np.median(t1c[mask_b]))
    enh_ratio = t1c_med / (t1_med + 1e-8)

    # Also compute in tumour core vs periphery
    core_b  = np.asarray(seg_arr == 3).astype(bool)   # enhancing only
    if core_b.any():
        t1c_core = float(np.median(t1c[core_b]))
        t1_core  = float(np.median(t1[core_b]))
        core_enh = t1c_core / (t1_core + 1e-8)
    else:
        core_enh = 0.0

    results["enhancement"] = {
        "t1_median_in_tumour":  _r(t1_med, 1),
        "t1c_median_in_tumour": _r(t1c_med, 1),
        "enhancement_ratio":    _r(enh_ratio, 3),
        "core_enhancement_ratio": _r(core_enh, 3),
        "method": "median",
    }
    print(f"\n  T1 median in tumour  : {t1_med:.1f}")
    print(f"  T1c median in tumour : {t1c_med:.1f}")
    print(f"  Enhancement ratio    : {enh_ratio:.3f}  (>1.3 = significant)")
    print(f"  Core enh. ratio      : {core_enh:.3f}")

# ── T2-FLAIR mismatch (MEDIAN for IDH-status proxy) ──────────────────
if t2 is not None and flair is not None:
    mask_b    = np.asarray(whole).astype(bool)
    # Median is a more stable biomarker for IDH-status in heterogeneous tumours
    t2_med    = float(np.median(t2[mask_b]))
    fl_med    = float(np.median(flair[mask_b]))
    t2_fl_rat = t2_med / (fl_med + 1e-8)

    # T2-FLAIR mismatch: T2 hyperintense but FLAIR hypointense in core
    # Proxy: high T2/FLAIR ratio in non-enhancing region
    # Label 2 (ED) only — pure infiltrative edema, excludes necrotic core (label 1).
    # Necrotic core is T2-hypointense (blood/products) and would reduce the ratio,
    # diluting specificity for IDH-mutant lower-grade glioma detection.
    ne_mask = np.asarray(seg_arr == 2).astype(bool)  # ED only
    if ne_mask.any():
        t2_ne = float(np.median(t2[ne_mask]))
        fl_ne = float(np.median(flair[ne_mask]))
        mismatch_score = (t2_ne/(fl_ne+1e-8)) - 1.0
    else:
        mismatch_score = 0.0

    results["idh_status_prediction"] = {
        "mismatch_score":      _r(mismatch_score, 3),
        "mismatch_sign_present": mismatch_score > 0.15,
        "likely_idh_mutant":   mismatch_score > 0.15,
        "method": "t2_flair_median_mismatch",
    }
    
    results["t2_flair"] = {
        "t2_median_tumour":    _r(t2_med, 1),
        "flair_median_tumour": _r(fl_med, 1),
        "t2_flair_ratio":      _r(t2_fl_rat, 3),
        "mismatch_score":      _r(mismatch_score, 3),
        "mismatch_sign":       mismatch_score > 0.15,
        "method": "median",
    }
    print(f"\n  T2 median in tumour  : {t2_med:.1f}")
    print(f"  FLAIR median         : {fl_med:.1f}")
    print(f"  T2/FLAIR ratio       : {t2_fl_rat:.3f}")
    mismatch_sign = mismatch_score > 0.15
    if mismatch_sign:
        print(f"  T2-FLAIR mismatch  : ✅ YES (score={mismatch_score:.3f})")
        print(f"                       → IDH-mutant signature present")
    else:
        print(f"  T2-FLAIR mismatch  : No (score={mismatch_score:.3f})")
        print(f"                       → IDH-wildtype more likely")

# ── Clinical interpretation ───────────────────────────────────────────
banner("MORPHOLOGY CLINICAL INTERPRETATION")

interp = []

if sphericity < 0.55:
    interp.append(("⚠️ ", "LOW sphericity — irregular infiltrative shape typical of "
                   "primary glioma (GBM). Metastases tend to be rounder."))
elif sphericity > 0.70:
    interp.append(("ℹ️ ", "MODERATE-HIGH sphericity — relatively round lesion. "
                   "More consistent with metastasis or lower-grade glioma."))

if convexity < 0.75:
    interp.append(("⚠️ ", "LOW convexity — tumour has concave/infiltrative borders. "
                   "Typical of primary glioma infiltrating surrounding tissue."))

if ncr_frac > 0.15:
    interp.append(("⚠️ ", f"HIGH necrosis fraction ({ncr_frac*100:.1f}%) — "
                   "significant necrotic core. WHO Grade IV criterion met. "
                   "Pseudopalisading necrosis is pathognomonic of GBM."))
elif ncr_frac > 0.03:
    interp.append(("ℹ️ ", f"Moderate necrosis ({ncr_frac*100:.1f}%) — "
                   "present but limited. Can be seen in Grade III-IV."))

if et_frac > 0.15:
    interp.append(("⚠️ ", f"HIGH enhancement fraction ({et_frac*100:.1f}%) — "
                   "significant blood-brain barrier breakdown. "
                   "High-grade tumour or aggressive metastasis."))

if ed_core > 4.0:
    interp.append(("ℹ️ ", f"HIGH oedema-to-core ratio ({ed_core:.1f}) — "
                   "extensive surrounding oedema relative to core. "
                   "More typical of primary glioma infiltration pattern."))
elif ed_core < 2.0:
    interp.append(("ℹ️ ", f"LOW oedema-to-core ratio ({ed_core:.1f}) — "
                   "limited oedema. Slightly more typical of metastasis."))

if roughness > 3.0:
    interp.append(("ℹ️ ", f"HIGH surface roughness ({roughness:.1f}mm) — "
                   "complex irregular surface. Associated with infiltrative growth."))

for prefix, text in interp:
    print(f"\n  {prefix}  {text}")

# ── Morphology plot ───────────────────────────────────────────────────
banner("GENERATING MORPHOLOGY VISUALISATION")

if t1c is not None:
    # FIX G — usar centro da MAIOR componente (não dos fragmentos dispersos)
    coords = np.argwhere(largest_comp_arr)
    if len(coords) > 0:
        z_coords = coords[:, 2]
        best_z = int(0.5 * (z_coords.min() + z_coords.max()))
    else:
        best_z = seg_arr.shape[2] // 2

    fig = plt.figure(figsize=(16, 8), facecolor="#0d0d0d")
    fig.suptitle(
        "Tumour Morphology Analysis — Maria Celeste Coelho Correia Soares",
        color="white", fontsize=12, y=0.98
    )
    _gs_fig: Any = gs
    grid = _gs_fig.GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.06)

    def show_slice(ax, vol, mask, title, cmap="gray", overlay_cmap=None):
        sl  = vol[:,:,best_z].T
        msl = mask[:,:,best_z].T
        ax.imshow(sl, cmap=cmap, vmin=np.percentile(sl,2),
                  vmax=np.percentile(sl,98))
        if overlay_cmap and msl.any():
            mma = np.ma.masked_where(msl==0, msl.astype(float))
            ax.imshow(mma, cmap=overlay_cmap, alpha=0.55, vmin=0, vmax=3)
        ax.set_title(title, color="white", fontsize=8, pad=3)
        ax.axis("off")

    # Row 1: modalities
    show_slice(fig.add_subplot(grid[0,0]), t1,   whole, "T1")
    show_slice(fig.add_subplot(grid[0,1]), t1c,  whole, "T1c (enhancing)")
    if t2 is not None:
        show_slice(fig.add_subplot(grid[0,2]), t2, whole, "T2")
    if flair is not None:
        show_slice(fig.add_subplot(grid[0,3]), flair, whole, "FLAIR")

    # Row 2: segmentation maps
    show_slice(fig.add_subplot(grid[1,0]), t1c, seg_arr, "3-region seg", overlay_cmap="Set1")

    # Surface map
    ax_s = fig.add_subplot(grid[1,1])
    ax_s.imshow(t1c[:,:,best_z].T, cmap="gray",
                vmin=np.percentile(t1c[:,:,best_z],2),
                vmax=np.percentile(t1c[:,:,best_z],98))
    _surf_a: Any = cast(Any, surface)
    surf_sl = _surf_a[:,:,best_z].T.astype(float)
    sm = np.ma.masked_where(surf_sl==0, surf_sl)
    ax_s.imshow(sm, cmap="YlOrRd", alpha=0.8)
    ax_s.set_title("Tumour surface", color="white", fontsize=8, pad=3)
    ax_s.axis("off")

    # Metrics bar chart
    ax_m = fig.add_subplot(grid[1,2])
    ax_m.set_facecolor("#1a1a2e")
    metrics_names  = ["Sphericity","Convexity","Elongation",
                       "NCR frac","ET frac","ED/core"]
    metrics_vals   = [sphericity, convexity, elongation,
                      min(ncr_frac*5,1), min(et_frac*5,1),
                      min(ed_core/10,1)]
    colors_bar     = ["#4499ff","#44ee44","#ff9900",
                      "#ff4444","#ff8888","#aaaaff"]
    bars = ax_m.barh(metrics_names, metrics_vals, color=colors_bar, height=0.6)
    ax_m.set_xlim(0,1)
    ax_m.set_title("Morphology scores", color="white", fontsize=8, pad=3)
    ax_m.tick_params(colors="white", labelsize=7)
    ax_m.spines[:].set_color("#444")
    for bar, val in zip(bars, metrics_vals):
        ax_m.text(min(val+0.02,0.95), bar.get_y()+bar.get_height()/2,
                  f"{val:.2f}", va="center", ha="left",
                  color="white", fontsize=7)

    # ADC histogram inside tumour
    # Use dilated mask to capture the high-grade core periphery
    # (important for sparse segmentations of infiltrative IDH-wt tumours)
    ax_h = fig.add_subplot(grid[1,3])
    ax_h.set_facecolor("#1a1a2e")
    # Try session EXTRA_DIR first, then legacy path
    adc_path = Path(str(EXTRA_DIR)) / "adc_resampled.nii.gz"
    if adc_path.exists():
        adc_arr = nib.load(str(adc_path)).get_fdata().astype(np.float32)
        if adc_arr.shape != seg_arr.shape:
            ms = tuple(min(a,b) for a,b in zip(adc_arr.shape, seg_arr.shape))
            adc_arr = adc_arr[:ms[0],:ms[1],:ms[2]]
            _whole_box: Any = whole
            wb = _whole_box[:ms[0],:ms[1],:ms[2]]
        else:
            wb = whole
        # Dilate mask by 1 voxel to sample the tumour periphery
        # (catches infiltrative edge of high-grade core)
        wb_dilated = ndimage.binary_dilation(wb, iterations=1)
        adc_vals = adc_arr[wb_dilated.astype(bool)]
        adc_vals = adc_vals[(adc_vals > 100) & (adc_vals < 3000)]
        if len(adc_vals) > 10:
            ax_h.hist(adc_vals, bins=50, color="#44aaff", alpha=0.8, edgecolor="none")
            ax_h.axvline(800,  color="#ff4444", lw=1.5, linestyle="--",
                         label="High grade (<800)")
            ax_h.axvline(1200, color="#ffaa00", lw=1.5, linestyle="--",
                         label="Moderate (800-1200)")
            adc_median = float(np.median(adc_vals))
            ax_h.axvline(adc_median, color="white", lw=1, linestyle=":",
                         label=f"Median={adc_median:.0f}")
            ax_h.set_title("ADC histogram (tumour+periphery)",
                           color="white", fontsize=8, pad=3)
            ax_h.tick_params(colors="white", labelsize=7)
            ax_h.spines[:].set_color("#444")
            ax_h.set_xlabel("ADC value", color="#aaa", fontsize=7)
            ax_h.legend(fontsize=6, labelcolor="white",
                        facecolor="#222", edgecolor="#444")
        else:
            ax_h.text(0.5,0.5,"ADC: too few voxels", color="white",
                      ha="center", va="center", transform=ax_h.transAxes)
            ax_h.axis("off")
    else:
        ax_h.text(0.5,0.5,"ADC not available", color="white",
                  ha="center", va="center", transform=ax_h.transAxes)
        ax_h.axis("off")

    out_img = OUTPUT_DIR / "morphology_analysis.png"
    plt.savefig(str(out_img), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved → {out_img}")

# ── Save JSON ─────────────────────────────────────────────────────────
morphology = {
    "timestamp":         datetime.now().isoformat(),
    "volumes":           results.get("volumes",{}),
    "clinical_ratios":   results.get("clinical_ratios",{}),
    "shape":             results.get("shape",{}),
    "enhancement":       results.get("enhancement",{}),
    "t2_flair":          results.get("t2_flair",{}),
    "interpretation":    [t for _,t in interp],
}
out_json = OUTPUT_DIR / "morphology.json"
with open(out_json, "w") as f:
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):  # type: ignore
            import numpy as _np  # type: ignore
            if isinstance(obj, (_np.bool_,)): return bool(obj)
            if isinstance(obj, (_np.integer,)): return int(obj)
            if isinstance(obj, (_np.floating,)): return float(obj)
            if isinstance(obj, _np.ndarray): return obj.tolist()
            return super().default(obj)
    json.dump(morphology, f, indent=2, cls=_NumpyEncoder)
print(f"  Saved → {out_json}")

banner("DONE")
print(f"  Open morphology_analysis.png to see the visualisation.")
print(f"  Run generate_report.py to include in PDF report.")
