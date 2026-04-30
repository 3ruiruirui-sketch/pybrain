#!/usr/bin/env python3
"""
Stage 12 — BraTS 2021 Style Figure (Fig. 1)
=============================================
Generates the four panels exactly as presented in the BraTS 2021 paper:
A: T1c with ET (yellow) + NCR (red)
B: T2 with tumor core (magenta = ET+NCR)
C: FLAIR with whole tumor (cyan = ET+NCR+ED)
D: Combined overlay (T1c background, ET yellow, NCR red, ED green)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pybrain.io.session import get_session, get_paths

    sess = get_session()
    paths = get_paths(sess)
    MONAI_DIR = Path(paths["monai_dir"])
    OUTPUT_DIR = Path(paths["output_dir"])
    SEG_DIR = Path(paths.get("seg_dir", OUTPUT_DIR))
except ImportError:
    print("❌ Error: Could not load PY-BRAIN session. Run pipeline first.")
    sys.exit(1)


def load_volume(path):
    p = Path(path)
    if not p.exists():
        return None
    return nib.load(str(p)).get_fdata().astype(np.float32)


def load_segmentation(seg_dir):
    for name in ["segmentation_ct_merged.nii.gz", "segmentation_ensemble.nii.gz", "segmentation_full.nii.gz"]:
        p = Path(seg_dir) / name
        if p.exists():
            return nib.load(str(p)).get_fdata().astype(np.uint8)
    return None


def get_slice(vol, axis, idx):
    if axis == 0:
        return vol[idx, :, :].T
    elif axis == 1:
        return vol[:, idx, :].T
    else:
        return vol[:, :, idx].T


def main():
    print("\n" + "=" * 65)
    print("  Stage 12 — BraTS 2021 Publication Figure")
    print("=" * 65)

    t1c = load_volume(MONAI_DIR / "t1c_resampled.nii.gz")
    if t1c is None:
        t1c = load_volume(MONAI_DIR / "t1.nii.gz")
    t2 = load_volume(MONAI_DIR / "t2_resampled.nii.gz")
    flair = load_volume(MONAI_DIR / "flair_resampled.nii.gz")
    seg = load_segmentation(SEG_DIR)

    if None in [t1c, t2, flair, seg]:
        print("❌ Error: Missing required MRI volumes or segmentation.")
        sys.exit(1)

    if seg.shape != t1c.shape:
        print(f"  ⚠️  Shape mismatch: SEG {seg.shape} vs MRI {t1c.shape}. Skipping.")
        sys.exit(0)

    # FIX 3: Maximize Tumor Core (ET + NCR), not whole tumor (which is edema-dominated).
    # Edema can be 5-10x larger than the core; argmax(WT) reliably selects
    # an edema-only slice where Panels A and B would be blank.
    core_mask = ((seg == 1) | (seg == 4)).astype(np.float32)  # BraTS 2021: ET = label 4
    if core_mask.sum() > 0:
        axial_scores = core_mask.sum(axis=(0, 1))
    else:
        # Fallback only if no core exists at all
        wt = (seg > 0).astype(np.float32)
        axial_scores = wt.sum(axis=(0, 1))

    best_z = int(np.argmax(axial_scores))
    print(f"  Selected axial slice: {best_z}  (maximised for tumour core)")

    # Extract slices
    t1c_slice = get_slice(t1c, 2, best_z)
    t2_slice = get_slice(t2, 2, best_z)
    flair_slice = get_slice(flair, 2, best_z)
    seg_slice = get_slice(seg, 2, best_z)

    # Masks
    ncr = (seg_slice == 1).astype(np.float32)
    edema = (seg_slice == 2).astype(np.float32)
    et = (seg_slice == 4).astype(np.float32)  # BraTS 2021: ET = label 4
    core = ((ncr + et) > 0).astype(np.float32)
    whole = ((core + edema) > 0).astype(np.float32)

    # FIX 1: Explicit LinearSegmentedColormap with exact RGBA values.
    # Using cmap='YlOrBr'/'Reds'/'Greens' on boolean masks maps 1.0 to the
    # colormap's bright edge, producing dark/washed-out colours.
    # Solid RGBA + LinearSegmentedColormap gives precise, publication-grade colours.
    cmap_et = LinearSegmentedColormap.from_list("et", [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.7)])  # Yellow
    cmap_ncr = LinearSegmentedColormap.from_list("ncr", [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.7)])  # Red
    cmap_edema = LinearSegmentedColormap.from_list("edema", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.8, 0.0, 0.5)])  # Green
    cmap_core = LinearSegmentedColormap.from_list("core", [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.6)])  # Magenta
    cmap_wt = LinearSegmentedColormap.from_list("wt", [(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.4)])  # Cyan

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.patch.set_facecolor("black")
    axes = axes.flatten()

    def norm_img(img):
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # FIX 2: interpolation='none' on mask overlays prevents anti-aliasing blur.
    # Background uses 'bicubic' for smooth MRI appearance.
    # All overlays use vmin=0, vmax=1 with explicit RGBA colormaps (Fix 1).

    # Panel A: T1c + ET (Yellow) + NCR (Red)
    axes[0].imshow(norm_img(t1c_slice), cmap="gray", interpolation="bicubic")
    axes[0].imshow(np.ma.masked_where(et == 0, et), cmap=cmap_et, vmin=0, vmax=1, interpolation="none")
    axes[0].imshow(np.ma.masked_where(ncr == 0, ncr), cmap=cmap_ncr, vmin=0, vmax=1, interpolation="none")
    axes[0].set_title("A: Enhancing (yellow) + Necrosis (red)", color="white", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel B: T2 + Tumor Core (Magenta)
    axes[1].imshow(norm_img(t2_slice), cmap="gray", interpolation="bicubic")
    axes[1].imshow(np.ma.masked_where(core == 0, core), cmap=cmap_core, vmin=0, vmax=1, interpolation="none")
    axes[1].set_title("B: Tumour core (magenta) on T2", color="white", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel C: FLAIR + Whole Tumor (Cyan)
    axes[2].imshow(norm_img(flair_slice), cmap="gray", interpolation="bicubic")
    axes[2].imshow(np.ma.masked_where(whole == 0, whole), cmap=cmap_wt, vmin=0, vmax=1, interpolation="none")
    axes[2].set_title("C: Whole tumour (cyan) on FLAIR", color="white", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Panel D: Combined (T1c + ET + NCR + ED)
    axes[3].imshow(norm_img(t1c_slice), cmap="gray", interpolation="bicubic")
    axes[3].imshow(np.ma.masked_where(edema == 0, edema), cmap=cmap_edema, vmin=0, vmax=1, interpolation="none")
    axes[3].imshow(np.ma.masked_where(et == 0, et), cmap=cmap_et, vmin=0, vmax=1, interpolation="none")
    axes[3].imshow(np.ma.masked_where(ncr == 0, ncr), cmap=cmap_ncr, vmin=0, vmax=1, interpolation="none")
    axes[3].set_title("D: Combined (yellow=ET, red=NCR, green=ED)", color="white", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout(pad=0.5)
    out_path = OUTPUT_DIR / "brats_figure1_style.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  ✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
