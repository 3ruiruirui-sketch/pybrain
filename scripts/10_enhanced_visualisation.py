#!/usr/bin/env python3
"""
Stage 10 — Enhanced Visualisation
===================================
Inspired by: Kaggle 3D MRI Brain Tumour Segmentation U-Net notebook
             (rastislav/3d-mri-brain-tumor-segmentation-u-net)

Produces:
  1. 4-panel MRI comparison  (T1 / T1c / T2 / FLAIR — Kaggle style)
  2. Region overlay grid     (all 3 tumour regions, 3 orientations)
  3. 3D surface mesh         (interactive HTML — plotly marching cubes)
  4. Intensity distributions (histogram + boxplot per sub-region)
  5. Slice animation         (scrollable HTML viewer)
  6. Attention heatmap       (probability map overlay)

Requirements: pip install nibabel numpy matplotlib scipy plotly scikit-image
Run:  python3 scripts/10_enhanced_visualisation.py
"""

import os
import sys
from pathlib import Path
from typing import Any, cast, Dict, List
import warnings
warnings.filterwarnings("ignore")

# ── pybrain Imports ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from pybrain.io.session import get_session, get_paths
    from pybrain.io.config import get_config
except ImportError:
    # Linter satisfaction fallback
    def get_session(): return {}
    def get_paths(s): return {}
    def get_config(): return {}

sess    = get_session()
paths   = get_paths(sess)
CONFIG  = get_config()
VIZ_CFG = CONFIG.get("visualizations", {})
MONAI   = Path(str(paths.get("monai_dir", PROJECT_ROOT)))
EXTRA   = Path(str(paths.get("extra_dir", PROJECT_ROOT)))
OUT     = Path(str(paths.get("output_dir", PROJECT_ROOT)))
_pat_dict: Any = sess.get("patient", {}) if isinstance(sess, dict) else {}
PATIENT: Dict[str, Any] = cast(Dict[str, Any], _pat_dict)

# typing already imported at top
import numpy as np
import nibabel as nib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy import ndimage
except ImportError as e:
    print(f"❌ {e}\npip install nibabel numpy matplotlib scipy")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("ℹ️  plotly not installed — 3D mesh and animation skipped")

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("ℹ️  scikit-image not installed — 3D mesh skipped")


def banner(t):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


def norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def load_vol(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return nib.load(str(p)).get_fdata().astype(np.float32)
    except (nib.filebasedimages.ImageFileError, Exception):
        return None


def best_slice(mask, axis=2):
    scores = mask.mean(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(scores))


def best_slices_n(mask, axis=2, n=6):
    scores = mask.mean(axis=tuple(i for i in range(3) if i != axis))
    return list(np.argsort(scores)[-n:])


def get_sl(arr, axis, idx):
    """Get slice from array along specified axis with bounds checking."""
    if axis == 0: 
        if idx >= arr.shape[0]:
            idx = arr.shape[0] - 1
        return arr[idx, :, :].T
    if axis == 1: 
        if idx >= arr.shape[1]:
            idx = arr.shape[1] - 1
        return arr[:, idx, :].T
    if axis == 2: 
        if idx >= arr.shape[2]:
            idx = arr.shape[2] - 1
        return arr[:, :, idx].T


def make_rgba_cmap(hex_color):
    r, g, b = (int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5))
    return LinearSegmentedColormap.from_list(
        "", [(r, g, b, 0), (r, g, b, 0.75)])

def rescale_intensity(arr, out_range=(0,1)):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    res = np.clip(arr, lo, hi)
    res = (res - lo) / (hi - lo + 1e-8)
    return res * (out_range[1] - out_range[0]) + out_range[0]


def prompt_user(question: str, default: bool = True) -> bool:
    """Non-interactive wrapper: always returns default."""
    return default


# ── Load volumes ──────────────────────────────────────────────────────────────
banner("LOADING VOLUMES")

t1    = load_vol(MONAI / "t1_resampled.nii.gz")
if t1 is None: t1 = load_vol(MONAI / "t1.nii.gz")
t1c   = load_vol(MONAI / "t1c_resampled.nii.gz")
t2    = load_vol(MONAI / "t2_resampled.nii.gz")
flair = load_vol(MONAI / "flair_resampled.nii.gz")

# Load best available segmentation
seg: np.ndarray = np.zeros((1,1,1)) # type: ignore
prob: Any = None
_found_seg = False
# Check seg_dir from paths (standardized cross-stage fallback)
seg_dir_context = Path(str(paths.get("seg_dir", OUT)))
# Ensure segmentation matches MRI shape
for fname in ["segmentation_ct_merged.nii.gz", "segmentation_ensemble.nii.gz", "segmentation_full.nii.gz"]:
    p = seg_dir_context / fname
    if p.exists():
        seg_nib = nib.load(str(p))
        _seg_vol = seg_nib.get_fdata()
        if _seg_vol.shape != t1.shape:
            print(f"  ⚠️  Found stale segmentation {fname} with mismatching shape ({_seg_vol.shape} vs MRI {t1.shape})")
            print(f"  🗑️  Removing stale file: {fname}")
            try: p.unlink()
            except: pass
            continue
        seg = _seg_vol.astype(np.uint8)
        print(f"  ✅ Using Segmentation: {fname}")
        _found_seg = True
        break

_prob_vol = load_vol(seg_dir_context / "ensemble_probability.nii.gz")
if _prob_vol is None:
    _prob_vol = load_vol(seg_dir_context / "tumor_probability.nii.gz")
prob = np.array(_prob_vol) if _prob_vol is not None else None

if t1 is None or not _found_seg:
    print("❌  T1 or segmentation not found — run Stages 1 and 3 first")
    sys.exit(1)

# Cast to Any to satisfy linter indexing
t1_a: Any = t1
t1c_a: Any = t1c
seg_a: Any = seg
prob_a: Any = prob
whole: np.ndarray = np.array(seg_a > 0).astype(np.float32)
ncr: np.ndarray = np.array(seg_a == 1).astype(np.float32)
ed: np.ndarray = np.array(seg_a == 2).astype(np.float32)
et: np.ndarray = np.array(seg_a == 3).astype(np.float32)

ref   = t1c if t1c is not None else t1

bz = best_slice(whole, 2)
by = best_slice(whole, 1)
bx = best_slice(whole, 0)

for v, n in [(t1,"T1"),(t1c,"T1c"),(t2,"T2"),(flair,"FLAIR")]:
    if v is not None:
        print(f"  {n:5s}: shape={v.shape}")
print(f"  Best axial slice: {bz}")


# ═════════════════════════════════════════════════════════════════════════
# VIZ 1 — 4-panel MRI (Kaggle notebook style)
# ═════════════════════════════════════════════════════════════════════════
banner("VIZ 1 — 4-PANEL MRI COMPARISON")

mods = [m for m in [("T1",t1),("T1c",t1c),("T2",t2),("FLAIR",flair)]
        if m[1] is not None]
n_sl = 6
slices = best_slices_n(whole, axis=2, n=n_sl)

fig, axes = plt.subplots(len(mods), n_sl,
                          figsize=(n_sl*2.8, len(mods)*2.8),
                          facecolor="#0a0a0a")
_axes: Any = axes # Cast for linter satisfaction
fig.suptitle(f"Multi-Modal MRI — {str(PATIENT.get('name','Patient'))}",
             color="white", fontsize=12, y=1.01, fontweight="bold")

for r, (name, vol) in enumerate(mods):
    for c, sl in enumerate(slices):
        if len(mods) > 1:
            _row: Any = cast(Any, _axes)[r]
        else:
            _row = cast(Any, _axes)
        ax: Any = cast(Any, _row)[c]
        ax.axis("off")
        s = get_sl(vol, 2, sl)
        ax.imshow(s, cmap="gray",
                  vmin=np.percentile(s,1), vmax=np.percentile(s,99))
        if c == 0:
            ax.set_ylabel(name, color="white", fontsize=9,
                          rotation=0, labelpad=40, va="center")
        if r == 0:
            ax.set_title(f"z={sl}", color="#888", fontsize=8)

plt.tight_layout(pad=0.3)
out1 = OUT / "viz_4panel_mri.png"
plt.savefig(str(out1), dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {out1}")


# ═════════════════════════════════════════════════════════════════════════
# VIZ 2 — Region overlay (3 orientations)
# ═════════════════════════════════════════════════════════════════════════
banner("VIZ 2 — TUMOUR REGION OVERLAY")

cmap_ed  = make_rgba_cmap("#44cc44")
cmap_ncr = make_rgba_cmap("#4488ff")
cmap_et  = make_rgba_cmap("#ff4444")

cols  = 5
fig, axes = plt.subplots(3, cols, figsize=(cols*3, 9), facecolor="#0a0a0a")
fig.suptitle(
    f"Tumour Sub-regions — {PATIENT.get('name','Patient')}\n"
    "🔵 Necrotic  🟢 Edema  🔴 Enhancing",
    color="white", fontsize=11, y=1.02)

for row, (axis, centre, label) in enumerate(
        [(2, bz, "Axial"), (1, by, "Coronal"), (0, bx, "Sagittal")]):
    offsets = [-8, -4, 0, 4, 8]
    for ci, off in enumerate(offsets):
        _shape: Any = getattr(ref, "shape", (0,0,0,0))
        sl = max(0, min(centre + off, int(_shape[axis])-1))
        _axes2: Any = axes
        _row2: Any = _axes2[row]
        ax: Any = _row2[ci]
        ax.axis("off")
        bg = get_sl(ref, axis, sl)
        ax.imshow(bg, cmap="gray",
                  vmin=np.percentile(bg,1), vmax=np.percentile(bg,99))
        for mask, cm in [(ed, cmap_ed), (ncr, cmap_ncr), (et, cmap_et)]:
            m = get_sl(mask, axis, sl)
            if m.max() > 0:
                ax.imshow(m, cmap=cm, vmin=0, vmax=1)
        if ci == 0:
            ax.set_ylabel(label, color="white", fontsize=9,
                          rotation=0, labelpad=45, va="center")

plt.tight_layout(pad=0.3)
out2 = OUT / "viz_region_overlay.png"
plt.savefig(str(out2), dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {out2}")


# ═════════════════════════════════════════════════════════════════════════
# VIZ 3 — Intensity distributions
# ═════════════════════════════════════════════════════════════════════════
banner("VIZ 3 — INTENSITY DISTRIBUTIONS")

regions = [
    ("Edema",     ed,    "#44cc44"),
    ("Necrotic",  ncr,   "#4488ff"),
    ("Enhancing", et,    "#ff4444"),
    ("Whole",     whole, "#aaaaaa"),
]

fig, axes = plt.subplots(2, len(mods), figsize=(len(mods)*4, 8),
                          facecolor="#0d0d0d")
fig.suptitle("Signal Intensity per Tumour Sub-region",
             color="white", fontsize=12, y=1.01)

for ci, (mod_name, vol) in enumerate(mods):
    _axes_flat: Any = axes
    _row_flat: Any = _axes_flat[0] if hasattr(_axes_flat[0], "__getitem__") else _axes_flat
    ax0: Any = _row_flat[ci]
    ax0.set_facecolor("#1a1a2e")
    ax0.set_title(mod_name, color="white", fontsize=10)
    ax0.tick_params(colors="white", labelsize=7)
    ax0.spines[:].set_color("#333")

    for rname, mask, color in regions:
        if mask.sum() < 10:
            continue
        _vol: Any = cast(Any, vol)
        vals = _vol[mask.astype(bool)]
        vals = vals[(vals > np.percentile(vals,1)) &
                    (vals < np.percentile(vals,99))]
        ax0.hist(vals, bins=60, alpha=0.55, color=color,
                 label=rname, density=True, histtype="stepfilled")
    ax0.legend(fontsize=7, labelcolor="white",
               facecolor="#222", edgecolor="#333")
    ax0.set_xlabel("Intensity", color="#aaa", fontsize=7)

    _axes_bot: Any = axes
    ax1_obj: Any = _axes_bot[1]
    ax1: Any = ax1_obj[ci]
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="white", labelsize=7)
    ax1.spines[:].set_color("#333")

    bdata, blabels, bcolors = [], [], []
    for rname, mask, color in regions:
        if mask.sum() < 10:
            continue
        _vol2: Any = cast(Any, vol)
        vals = _vol2[mask.astype(bool)]
        # Ensure we have enough data points for boxplot
        if len(vals) == 0:
            continue
        bdata.append(vals[::max(1, len(vals)//2000)])
        blabels.append(rname)
        bcolors.append(color)

    # Only create boxplot if we have data
    if len(bdata) > 0 and len(blabels) > 0 and len(bdata) == len(blabels):
        _medprops: Any = {"color": "white", "linewidth": 2}
        _ax1: Any = ax1
        bp = _ax1.boxplot(bdata, labels=blabels, patch_artist=True, notch=False,
                         medianprops=_medprops)
        for patch, color in zip(bp["boxes"], bcolors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_xticklabels(blabels, rotation=20, ha="right",
                             color="white", fontsize=7)
    else:
        ax1.text(0.5, 0.5, "No data available for boxplot", 
                transform=ax1.transAxes, ha="center", va="center", 
                color="white", fontsize=8)

plt.tight_layout(pad=1.0)
out3 = OUT / "viz_intensity_distributions.png"
plt.savefig(str(out3), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved → {out3}")


# VIZ 4 — Confidence heatmap
# ═════════════════════════════════════════════════════════════════════════
banner("VIZ 4 — CONFIDENCE HEATMAP")

if prob is not None:
    n_att = 8
    att_sl = best_slices_n(whole, axis=2, n=n_att)

    fig, _axes = plt.subplots(2, n_att, figsize=(n_att*2.5, 5),
                              facecolor="#0a0a0a")
    axes: Any = _axes
    fig.suptitle("Model Prediction Confidence — Probability Gradient",
                 color="white", fontsize=11, y=1.02)

    for ci, sl in enumerate(att_sl):
        bg   = get_sl(ref, 2, sl)
        pr   = get_sl(prob, 2, sl)
        sg   = get_sl(seg, 2, sl)

        _axes_heat: Any = axes
        ax0_obj: Any = _axes_heat[0]
        ax0: Any = ax0_obj[ci]
        ax0.axis("off")
        ax0.imshow(bg, cmap="gray",
                   vmin=np.percentile(bg,1), vmax=np.percentile(bg,99))
        ax0.imshow(pr, cmap="hot", alpha=0.5, vmin=pr.min(), vmax=pr.max())
        ax0.set_title(f"z={sl}", color="#888", fontsize=7)
        if ci == 0:
            ax0.set_ylabel("Confidence", color="white", fontsize=8,
                           rotation=0, labelpad=45, va="center")

        ax1_obj_heat: Any = _axes_heat[1]
        ax1: Any = ax1_obj_heat[ci]
        ax1.axis("off")
        ax1.imshow(bg, cmap="gray",
                   vmin=np.percentile(bg,1), vmax=np.percentile(bg,99))
        for lbl, color in [(2,"#44cc44"),(1,"#4488ff"),(3,"#ff4444")]:
                m = (sg == lbl)
                # Clarify types for linter
                m_f = np.array(m).astype(np.float32)
                rgba = np.zeros(tuple(list(m_f.shape) + [4]), dtype=np.float32)
                
                # Parse hex colour via bytes.fromhex — avoids str-slice TypeVar linter issue
                _c_str_obj: Any = color
                _c_hex: str = str(_c_str_obj).replace("#", "")
                _rgb_bytes: bytes = bytes.fromhex(_c_hex)
                r_v: float = _rgb_bytes[0] / 255.0
                g_v: float = _rgb_bytes[1] / 255.0
                b_v: float = _rgb_bytes[2] / 255.0
                
                rgba[..., 0] = r_v
                rgba[..., 1] = g_v
                rgba[..., 2] = b_v
                rgba[..., 3] = (m_f * 0.7)
                _ax1_a: Any = ax1
                _ax1_a.imshow(rgba)
        if ci == 0:
            ax1.set_ylabel("Segmentation", color="white", fontsize=8,
                           rotation=0, labelpad=45, va="center")

    plt.tight_layout(pad=0.2)
    out4 = OUT / "viz_attention_heatmap.png"
    plt.savefig(str(out4), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out4}")
else:
    print("  ⚠️  No probability map — run Stage 3 to generate ensemble_probability.nii.gz")
    out4 = None


# ═════════════════════════════════════════════════════════════════════════
# VIZ 5 — 3D surface mesh (marching cubes)
# ═════════════════════════════════════════════════════════════════════════
banner("VIZ 5 — 3D SURFACE MESH")

if HAS_PLOTLY and HAS_SKIMAGE:
    try:
        nib_hdr   = nib.load(str(MONAI / "t1.nii.gz")).header
        vox_sizes = tuple(float(v) for v in nib_hdr.get_zooms()[:3])

        fig3d = go.Figure()
        meshes = [
            ("Edema",     ed,    0.25, "#44cc44"),
            ("Necrotic",  ncr,   0.55, "#4488ff"),
            ("Enhancing", et,    0.80, "#ff4444"),
        ]
        for rname, mask, opacity, color in meshes:
            if mask.sum() < 50:
                continue
            smooth = ndimage.gaussian_filter(mask.astype(float), sigma=1.5)
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    smooth, level=0.5, spacing=vox_sizes)
            except Exception:
                continue
            x, y, z = verts[:,0], verts[:,1], verts[:,2]
            i, j, k = faces[:,0], faces[:,1], faces[:,2]
            fig3d.add_trace(go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                color=color, opacity=opacity,
                name=rname, showlegend=True,
                hovertemplate=f"<b>{rname}</b><extra></extra>",
            ))

        fig3d.update_layout(
            title={
                "text": (f"3D Tumour Surface — {str(PATIENT.get('name','Patient'))}<br>"
                         "<span style='font-size:10px;color:#aaa'>"
                         "🔵 Necrotic  🟢 Edema  🔴 Enhancing | "
                         "Drag to rotate · Scroll to zoom</span>"),
                "x": 0.5, "font": {"color": "white", "size": 12}},
            scene={
                "xaxis": {"title": "X mm", "color": "white", "showgrid": False},
                "yaxis": {"title": "Y mm", "color": "white", "showgrid": False},
                "zaxis": {"title": "Z mm", "color": "white", "showgrid": False},
                "bgcolor": "#111",
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.0}},
            },
            paper_bgcolor="#111", height=650,
            legend={"font": {"color": "white"},
                    "bgcolor": "#222", "bordercolor": "#444"},
            margin={"l": 0, "r": 0, "t": 80, "b": 0},
        )
        out5 = OUT / "viz_3d_surface.html"
        fig3d.write_html(str(out5))
        print(f"  Saved → {out5}")
    except Exception as e:
        print(f"  ⚠️  3D mesh failed: {e}")
        out5 = None
else:
    print("  ⚠️  Skipped — pip install plotly scikit-image")
    out5 = None


# ═════════════════════════════════════════════════════════════════════════
# VIZ 5b — Surgical Navigation 3D Reconstruction (PNG for PDF embedding)
# ═════════════════════════════════════════════════════════════════════════
banner("VIZ 5b — SURGICAL NAVIGATION 3D RECONSTRUCTION")

out5b_path = OUT / "viz_3d_surgical_navigation.png"

if HAS_SKIMAGE:
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.colors import LightSource
        import matplotlib.cm as cm

        # Load header for voxel sizes
        t1_path_candidates = [
            MONAI / "t1_resampled.nii.gz",
            MONAI / "t1.nii.gz",
        ]
        hdr_path = None
        for cp in t1_path_candidates:
            if cp.exists():
                hdr_path = cp
                break

        if hdr_path is None:
            raise RuntimeError("No T1 volume found")

        nib_hdr   = nib.load(str(hdr_path))
        vox_sizes = tuple(float(v) for v in nib_hdr.get_zooms()[:3])

        # Normalise all volumes to same space using seg shape as reference
        # (all volumes should already be same shape — use seg shape)
        vol_shape = seg_a.shape

        def _to_mm(arr):
            """Convert voxel array to mm coordinates using voxel sizes."""
            return np.array(arr) * np.array(vox_sizes)

        # ── Surface extraction ──────────────────────────────────────────
        def _get_mesh(mask_vol, level, label):
            """Extract marching-cubes mesh; return (verts, faces) or (None, None)."""
            if mask_vol.sum() < 20:
                return None, None
            smooth = ndimage.gaussian_filter(
                mask_vol.astype(np.float32), sigma=1.0)
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    smooth, level=level, spacing=(1.0, 1.0, 1.0))
                return _to_mm(verts), faces
            except Exception:
                return None, None

        # Brain surface: threshold on T1 (normalise T1 to 0-1 first)
        t1_norm = norm(t1_a.astype(np.float32))
        brain_mask = (t1_norm > 0.15).astype(np.float32)
        brain_verts, brain_faces = _get_mesh(brain_mask, 0.5, "Brain")

        ed_verts, ed_faces     = _get_mesh(ed, 0.5, "Edema")
        ncr_verts, ncr_faces   = _get_mesh(ncr, 0.5, "Necrotic")
        et_verts, et_faces     = _get_mesh(et, 0.5, "Enhancing")

        # Centroid for cross-section view (whole tumour)
        if whole.sum() > 0:
            coords = np.argwhere(whole > 0)
            centroid_vox = coords.mean(axis=0)
            centroid_mm  = _to_mm(centroid_vox)
        else:
            centroid_mm = np.array([s * v / 2 for s, v in zip(vol_shape, vox_sizes)])

        # ── Create figure ──────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 8), facecolor="#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        ls = LightSource(azdeg=315, altdeg=45)

        # ── Left panel: Full 3D surface view ───────────────────────────
        ax1 = fig.add_subplot(1, 2, 1, projection='3d', facecolor="#1a1a2e")
        ax1.set_facecolor("#1a1a2e")

        def _add_surface(verts, faces, color, alpha, label, shade=True):
            if verts is None:
                return
            mesh = Poly3DCollection(
                verts[faces], alpha=alpha,
                linewidths=0.15, edgecolors=(0.2, 0.2, 0.2, 0.3)
            )
            if shade and len(verts) > 0:
                normals = _compute_normals(verts, faces)
                col = ls.shade_samples(verts, normals, color)
                mesh.set_facecolor(col)
            else:
                mesh.set_facecolor(color)
            mesh.set_label(label)
            ax1.add_collection3d(mesh)

        def _compute_normals(verts, faces):
            """Compute face normals for shading."""
            normals = np.zeros_like(verts)
            for f in faces:
                v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
                n = np.cross(v1 - v0, v2 - v0)
                nl = np.linalg.norm(n)
                if nl > 1e-8:
                    n = n / nl
                normals[f[0]] += n
                normals[f[1]] += n
                normals[f[2]] += n
            norms_mag = np.linalg.norm(normals, axis=1, keepdims=True)
            norms_mag = np.maximum(norms_mag, 1e-8)
            return normals / norms_mag

        # Brain surface (light gray, semi-transparent)
        _add_surface(brain_verts, brain_faces,
                      (0.85, 0.82, 0.80), 0.20, "Brain")

        # Edema (green)
        _add_surface(ed_verts, ed_faces,
                      (0.27, 0.87, 0.27), 0.65, "Edema")

        # Necrotic core (blue)
        _add_surface(ncr_verts, ncr_faces,
                      (0.27, 0.53, 1.00), 0.75, "Necrotic")

        # Enhancing tumour (red)
        _add_surface(et_verts, et_faces,
                      (1.00, 0.27, 0.27), 0.85, "Enhancing")

        # Set axis limits based on brain volume extent
        if brain_verts is not None:
            ax1.set_xlim(brain_verts[:, 0].min(), brain_verts[:, 0].max())
            ax1.set_ylim(brain_verts[:, 1].min(), brain_verts[:, 1].max())
            ax1.set_zlim(brain_verts[:, 2].min(), brain_verts[:, 2].max())

        ax1.set_xlabel("R ← → L (mm)", fontsize=8, color="#aaa")
        ax1.set_ylabel("A ← → P (mm)", fontsize=8, color="#aaa")
        ax1.set_zlabel("S ← → I (mm)", fontsize=8, color="#aaa")
        ax1.tick_params(colors="#888", labelsize=7)
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

        # Title overlay
        ax1.set_title(
            f"Surgical Navigation — 3D Reconstruction\n"
            f"{str(PATIENT.get('name', 'Patient'))}  |  "
            f"Surface rendering: brain + tumour sub-regions",
            color="white", fontsize=10, pad=10
        )

        # Legend patches (manual proxy artists)
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=(1.0, 0.27, 0.27, 0.85), label='Enhancing Tumour (ET)'),
            Patch(facecolor=(0.27, 0.53, 1.00, 0.75), label='Necrotic Core (NCR)'),
            Patch(facecolor=(0.27, 0.87, 0.27, 0.65), label='Peritumoural Edema (ED)'),
            Patch(facecolor=(0.85, 0.82, 0.80, 0.20), label='Brain Surface'),
        ]
        ax1.legend(handles=legend_patches, loc='upper left',
                   fontsize=7, facecolor="#222", edgecolor="#444",
                   label_color="white", ncol=1)

        # Camera angle — slightly from above-left for depth perception
        ax1.view_init(azim=40, elev=25)

        # ── Right panel: Cross-section view with depth scale ───────────
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', facecolor="#1a1a2e")
        ax2.set_facecolor("#1a1a2e")

        # Get sagittal slice index nearest centroid
        sagittal_idx = int(np.clip(centroid_vox[0], 0, vol_shape[0] - 1))

        def _slice_panel(arr, axis, idx, cmap, vmin, vmax, label, alpha=1.0):
            """Add a 2D slice as a textured quad in 3D."""
            if axis == 0:
                sl = arr[idx, :, :]
                xx, yy = _to_mm(np.meshgrid(
                    np.arange(sl.shape[1]), np.arange(sl.shape[0])))
                zz = np.full_like(xx, arr.shape[0] * vox_sizes[0] / 2)
                normal = (1, 0, 0)
            elif axis == 1:
                sl = arr[:, idx, :]
                xx, yy = _to_mm(np.meshgrid(
                    np.arange(sl.shape[1]), np.arange(sl.shape[0])))
                zz = np.full_like(xx, arr.shape[1] * vox_sizes[1] / 2)
                normal = (0, 1, 0)
            else:
                sl = arr[:, :, idx]
                xx, yy = _to_mm(np.meshgrid(
                    np.arange(sl.shape[1]), np.arange(sl.shape[0])))
                zz = np.full_like(xx, arr.shape[2] * vox_sizes[2] / 2)
                normal = (0, 0, 1)

            sl_norm = np.clip((sl - vmin) / (vmax - vmin + 1e-8), 0, 1)
            ax2_surface = ax2.plot_surface(
                xx, yy, zz,
                rstride=4, cstride=4,
                facecolors=cmap(sl_norm),
                linewidths=0,
                shade=False,
                alpha=alpha,
            )
            return sl

        # T1 sagittal cross-section at centroid
        if t1_a is not None:
            t1_slice = t1_a[sagittal_idx, :, :] if sagittal_idx < t1_a.shape[0] \
                       else t1_a[-1, :, :]
            t1_min, t1_max = np.percentile(t1_slice[t1_slice > 0], [2, 98]) \
                              if t1_slice.max() > 0 else (0, 1)
            xx_sag, yy_sag = _to_mm(np.meshgrid(
                np.arange(t1_slice.shape[1]),
                np.arange(t1_slice.shape[0])))
            zz_sag = np.full_like(xx_sag,
                sagittal_idx * vox_sizes[0])
            t1_s = np.clip((t1_slice - t1_min) / (t1_max - t1_min + 1e-8), 0, 1)
            ax2.plot_surface(xx_sag, yy_sag, zz_sag,
                             rstride=3, cstride=3,
                             facecolors=plt.cm.gray(t1_s),
                             linewidths=0, shade=False, alpha=0.9)

        # Overlay tumour region colours on the cross-section
        for reg_mask, reg_color, reg_name in [
            (ed,  (0.27, 0.87, 0.27, 0.5), "Edema"),
            (ncr, (0.27, 0.53, 1.00, 0.6), "NCR"),
            (et,  (1.00, 0.27, 0.27, 0.7), "ET"),
        ]:
            if sagittal_idx < reg_mask.shape[0]:
                sl = reg_mask[sagittal_idx, :, :]
                if sl.sum() > 0:
                    ys, xs = np.where(sl > 0)
                    if len(xs) > 0:
                        xm = _to_mm(xs)
                        ym = _to_mm(ys)
                        zm = np.full_like(xm, sagittal_idx * vox_sizes[0])
                        ax2.scatter(xm, ym, zm,
                                   c=[reg_color[:3]], s=3,
                                   alpha=reg_color[3], marker='o')

        # Depth ruler — vertical line showing scale
        x_ruler = brain_verts[:, 0].min() - 15 if brain_verts is not None else -10
        z_ruler = brain_verts[:, 2].min() if brain_verts is not None else 0
        y_ruler = brain_verts[:, 1].mean() if brain_verts is not None else vol_shape[1] * vox_sizes[1] / 2

        for y0, y1, label in [
            (z_ruler, z_ruler + 20 / vox_sizes[2], "20 mm"),
            (z_ruler, z_ruler + 50 / vox_sizes[2], "50 mm"),
        ]:
            ax2.plot([x_ruler, x_ruler],
                     [y_ruler, y_ruler],
                     [z_ruler, y1],
                     color='white', linewidth=1.5)
            mid_z = (z_ruler + y1) / 2
            ax2.text(x_ruler - 2, y_ruler, mid_z,
                     f"{int((y1 - z_ruler) * vox_sizes[2])} mm",
                     color='white', fontsize=7, va='center')

        # Centroid marker
        ax2.scatter(*[_to_mm(centroid_vox)],
                    c='yellow', s=60, marker='*', zorder=10,
                    label=f"Tumour centroid (RAS: "
                          f"R={centroid_mm[0]:.0f} A={centroid_mm[1]:.0f} S={centroid_mm[2]:.0f})")

        if brain_verts is not None:
            ax2.set_xlim(brain_verts[:, 0].min(), brain_verts[:, 0].max())
            ax2.set_ylim(brain_verts[:, 1].min(), brain_verts[:, 1].max())
            ax2.set_zlim(brain_verts[:, 2].min(), brain_verts[:, 2].max())

        ax2.set_xlabel("R ← → L (mm)", fontsize=8, color="#aaa")
        ax2.set_ylabel("A ← → P (mm)", fontsize=8, color="#aaa")
        ax2.set_zlabel("S ← → I (mm)", fontsize=8, color="#aaa")
        ax2.tick_params(colors="#888", labelsize=7)
        ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

        ax2.set_title(
            f"Sagittal Cross-Section at Tumour Centroid\n"
            f"RAS: R={centroid_mm[0]:.0f}mm  A={centroid_mm[1]:.0f}mm  "
            f"S={centroid_mm[2]:.0f}mm  |  Slice z={sagittal_idx}",
            color="white", fontsize=10, pad=10
        )
        ax2.view_init(azim=90, elev=0)  # Pure sagittal view

        # Add a horizontal depth scale bar
        fig.text(0.5, 0.02,
                 "Surgical Navigation 3D — PY-BRAIN Research Pipeline  |  "
                 " BrainIAC Brain Tumor Analysis  |  "
                 "FOR RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS",
                 ha="center", fontsize=7, color="#666", style="italic")

        plt.tight_layout(pad=1.5)
        fig.savefig(str(out5b_path), dpi=150,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved → {out5b_path}")

    except Exception as e:
        import traceback
        print(f"  ⚠️  Surgical 3D failed: {e}")
        traceback.print_exc()
        out5b_path = None
else:
    print("  ⚠️  Skipped — scikit-image not available")
    out5b_path = None


banner("VIZ 6 — SLICE ANIMATION")

# Optional generation (can be >200MB)
gen_anim = VIZ_CFG.get("generate_animation", True)
if gen_anim:
    gen_anim = prompt_user("Generate slice animation (large file, 200MB+)?", default=True)

if gen_anim and HAS_PLOTLY and ref is not None:
    _ref: Any = ref  # narrowed non-None reference for linter
    n_z    = _ref.shape[2]
    t_sl   = [z for z in range(n_z) if whole[:,:,z].sum() > 5]
    if not t_sl:
        t_sl = list(range(n_z//4, 3*n_z//4))

    frames = []
    for sl in t_sl:
        bg   = get_sl(_ref,  2, sl)
        n_s  = get_sl(ncr,   2, sl)
        e_s  = get_sl(ed,    2, sl)
        et_s = get_sl(et,    2, sl)
        frames.append(go.Frame(data=[
            go.Heatmap(z=bg[::-1], colorscale="gray", showscale=False,
                       zmin=float(np.percentile(bg,1)),
                       zmax=float(np.percentile(bg,99))),
            go.Heatmap(z=np.where(e_s[::-1]  >0, 1.0, np.nan),
                       colorscale=[[0,"#44cc44"],[1,"#44cc44"]],
                       showscale=False, opacity=0.45),
            go.Heatmap(z=np.where(n_s[::-1]  >0, 1.0, np.nan),
                       colorscale=[[0,"#4488ff"],[1,"#4488ff"]],
                       showscale=False, opacity=0.60),
            go.Heatmap(z=np.where(et_s[::-1] >0, 1.0, np.nan),
                       colorscale=[[0,"#ff4444"],[1,"#ff4444"]],
                       showscale=False, opacity=0.70),
        ], name=str(sl)))

    mid  = t_sl[len(t_sl)//2]
    bg0  = get_sl(_ref, 2, mid)
    steps = [{"args": [[str(sl)], {"frame": {"duration": 0}, "mode": "immediate"}],
               "label": str(sl), "method": "animate"} for sl in t_sl]

    fig_anim = go.Figure(
        data=[
            go.Heatmap(z=bg0[::-1], colorscale="gray", showscale=False,
                       zmin=float(np.percentile(bg0,1)),
                       zmax=float(np.percentile(bg0,99)),
                       name=str(mid)),
            go.Heatmap(z=np.where(get_sl(ed, 2,mid)[::-1]  >0,1.,np.nan),
                       colorscale=[[0,"#44cc44"],[1,"#44cc44"]],
                       showscale=False, opacity=0.45),
            go.Heatmap(z=np.where(get_sl(ncr,2,mid)[::-1]  >0,1.,np.nan),
                       colorscale=[[0,"#4488ff"],[1,"#4488ff"]],
                       showscale=False, opacity=0.60),
            go.Heatmap(z=np.where(get_sl(et, 2,mid)[::-1]  >0,1.,np.nan),
                       colorscale=[[0,"#ff4444"],[1,"#ff4444"]],
                       showscale=False, opacity=0.70),
        ],
        frames=frames,
    )
    fig_anim.update_layout(
        title={
            "text": (f"Slice Viewer — {PATIENT.get('name','Patient')}<br>"
                     "<span style='font-size:10px;color:#aaa'>"
                     "🟢 Edema  🔵 Necrotic  🔴 Enhancing | ▶ Play or drag slider</span>"),
            "x": 0.5, "font": {"color": "white", "size": 12}},
        paper_bgcolor="#111", plot_bgcolor="#111",
        sliders=[{
            "active": len(steps)//2, "steps": steps, "pad": {"t": 50},
            "currentvalue": {"prefix": "Slice: ",
                             "font": {"color": "white", "size": 13}},
            "font": {"color": "white"}}],
        updatemenus=[{
            "type": "buttons", "showactive": False, "y": 1.18, "x": 0.5,
            "xanchor": "center",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 80}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ]}],
        height=680, width=720,
        margin={"l": 20, "r": 20, "t": 100, "b": 90},
    )
    out6 = OUT / "viz_slice_animation.html"
    fig_anim.write_html(str(out6))
    print(f"  Saved → {out6}")
else:
    if not gen_anim:
        print("  ⏭️  Skipped by user/config")
    elif not HAS_PLOTLY:
        print("  ⚠️  Skipped — pip install plotly")
    elif ref is None:
        print("  ⚠️  No reference volume — skipping slice animation")
    out6 = None


# ═════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════
banner("DONE")

for fname, desc in [
    ("viz_4panel_mri.png",              "4-panel MRI comparison"),
    ("viz_region_overlay.png",           "Region overlay — 3 orientations"),
    ("viz_intensity_distributions.png",  "Intensity histograms + box plots"),
    ("viz_attention_heatmap.png",        "AI attention heatmap"),
    ("viz_3d_surface.html",             "3D surface — open in browser"),
    ("viz_slice_animation.html",         "Slice animation — open in browser"),
    ("viz_3d_surgical_navigation.png",  "Surgical navigation 3D (PDF-ready)"),
]:
    p = OUT / fname
    if p.exists():
        size = p.stat().st_size/1024
        print(f"  ✅ {fname:<42} "
              f"{'%.1f MB' % (size/1024) if size>1024 else '%.0f KB' % size}")
    else:
        print(f"  ⚠️  {fname:<42} not created")

print(f"\n  Open 3D surface:  open \"{OUT}/viz_3d_surface.html\"")
print(f"  Open animation:   open \"{OUT}/viz_slice_animation.html\"")

# Automatically open in browser (macOS)
if sys.platform == "darwin":
    auto_open = VIZ_CFG.get("auto_open", True)
    if auto_open:
        auto_open = prompt_user("Open interactive visualizations in browser?", default=True)
    
    if auto_open:
        if out5 and out5.exists():
            os.system(f"open '{out5}'")
        if out6 and out6.exists():
            os.system(f"open '{out6}'")

print("═"*65)
print("  ✅  Stage 10 done")
print("═"*65)
