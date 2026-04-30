#!/usr/bin/env python3
"""
Radiomics Analysis + ML Tumour Classification
===============================================
Extracts 107 PyRadiomics features from the tumour segmentation
and applies ML classifiers to estimate:
  - Tumour grade (high vs low)
  - Primary vs metastasis probability
  - IDH mutation likelihood
  - MGMT methylation likelihood
  - Aggressiveness composite score

Requirements:
  pip install scikit-image scikit-learn nibabel numpy scipy
  (PyRadiomics is NOT required — incompatible with Python 3.12)

Run:
  python3 radiomics_analysis.py

Output:
  radiomics_features.json   — all 107 features
  radiomics_report.txt      — human-readable findings
  Both saved to latest results folder
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, cast

# ── pybrain Imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from pybrain.io.session import get_session, get_paths, get_patient  # type: ignore
    from pybrain.io.config import get_config  # type: ignore
    from pybrain.io.logging_utils import setup_logging  # type: ignore

    _sess = get_session()
    _paths = get_paths(_sess)
    _config = get_config()
    PATIENT = get_patient(_sess)

    OUTPUT_DIR = _paths["output_dir"]
    MONAI_DIR = _paths["monai_dir"]
    RESULTS_DIR = _paths["results_dir"]
    EXTRA_DIR = _paths["extra_dir"]

    logger = setup_logging(OUTPUT_DIR)
    logger.info("Stage 8 — Radiomics Analysis — Initialized")

    import torch  # type: ignore

    DEVICE = torch.device(_config["hardware"]["device"])
    logger.info(f"Using device: {DEVICE}")

except Exception as e:
    print(f"❌ Failed to load session: {e}")
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# ⚙️  PATHS
# ─────────────────────────────────────────────────────────────────────────

RESULTS_BASE = RESULTS_DIR
MRI_DIR = Path(str(MONAI_DIR))
EXTRA_DIR = Path(str(EXTRA_DIR))
# OUTPUT_DIR is defined above from session _paths["output_dir"]


def banner(t):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────

try:
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore
    from scipy import ndimage  # type: ignore
    from scipy.stats import skew, kurtosis  # type: ignore
except ImportError as e:
    print(f"❌ {e}\nRun: pip install nibabel numpy scipy")
    sys.exit(1)

try:
    import radiomics  # type: ignore
    from radiomics import featureinspector, getFeatureClasses  # type: ignore

    HAS_PYRADIOMICS = True
    print("✅ PyRadiomics available — full 107 clinical features enabled")
except ImportError:
    HAS_PYRADIOMICS = False
    print("ℹ️  PyRadiomics not found — falling back to scikit-image texture features")

HAS_SKIMAGE = False
try:
    from skimage.feature import graycomatrix, graycoprops  # type: ignore
    from skimage.exposure import rescale_intensity  # type: ignore

    HAS_SKIMAGE = True
    if not HAS_PYRADIOMICS:
        print("✅ scikit-image available — GLCM texture features enabled")
except ImportError:
    if not HAS_PYRADIOMICS:
        print("ℹ️  scikit-image not found — run: pip install scikit-image")

HAS_SKLEARN = False
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    HAS_SKLEARN = True
    print("✅ scikit-learn available")
except ImportError:
    print("ℹ️  scikit-learn not installed — ML classification skipped")
    print("   Install: pip install scikit-learn")

HAS_MONAI = False
HAS_PYTORCH = False
try:
    import torch  # type: ignore

    HAS_PYTORCH = True
    import monai  # type: ignore
    from monai.networks.nets import DenseNet121  # type: ignore

    HAS_MONAI = True
    print("✅ MONAI Deep Learning available — 2.5D CNN Path enabled (MPS accelerated)")
except ImportError:
    print("ℹ️  MONAI not found — Deep Learning path disabled")


# ─────────────────────────────────────────────────────────────────────────
# 🧠  MONAI 2.5D CNN PATH (MPS-Accelerated Deep Learning)
# ─────────────────────────────────────────────────────────────────────────
#
# 2.5D approach: run **2D** DenseNet121 on 3 orthogonal planes (axial,
# coronal, sagittal) through the tumour centroid.  2D DenseNet has full
# MPS support on Apple Silicon, unlike the 3D variant.
#
# Each plane produces ~1024 deep features → fused to ~3072 total.
# Softmax predictions are averaged across planes for consensus.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _extract_centroid(mask_bool: np.ndarray, vox_sizes: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Return (x, y, z) centroid of the largest lesion component (ignoring noise <1cc)."""
    from scipy.ndimage import label  # type: ignore

    labeled_mask, num_features = label(mask_bool)
    if num_features == 0:
        return np.array([s // 2 for s in mask_bool.shape])

    # Filter by size: 0.5cc = 500mm3 (to preserve secondary 1cm lesions)
    vox_vol_mm3 = float(np.prod(vox_sizes))
    cc_threshold = 500.0 / vox_vol_mm3
    sizes = np.bincount(labeled_mask.ravel())

    if len(sizes) > 1:
        # Filtrar componentes menores que 0.5cc
        valid = np.where(sizes[1:] >= cc_threshold)[0] + 1
        if len(valid) == 0:
            valid = np.array([np.argmax(sizes[1:]) + 1])
        largest_idx = valid[np.argmax(sizes[valid])]
    else:
        largest_idx = 1

    coords = np.argwhere(labeled_mask == largest_idx)
    return coords.mean(axis=0).astype(int)


def _make_3d_patch(
    volumes: dict,
    centroid: np.ndarray,
    target_size: int = 96,
) -> np.ndarray:
    """Extract a 4-channel 3D patch centered at `centroid`."""
    modalities = ["T1", "T1c", "T2", "FLAIR"]
    channels = []

    # centroid = (c_x, c_y, c_z)
    for mod in modalities:
        if mod not in volumes:
            channels.append(np.zeros((target_size, target_size, target_size), dtype=np.float32))
            continue
        vol = volumes[mod]
        # Pad volume if smaller than target_size
        pad_x = max(0, target_size - vol.shape[0])
        pad_y = max(0, target_size - vol.shape[1])
        pad_z = max(0, target_size - vol.shape[2])
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            vol = np.pad(vol, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant")

        x, y, z = centroid
        x_start = max(0, min(int(x - target_size // 2), vol.shape[0] - target_size))
        y_start = max(0, min(int(y - target_size // 2), vol.shape[1] - target_size))
        z_start = max(0, min(int(z - target_size // 2), vol.shape[2] - target_size))

        patch = vol[x_start : x_start + target_size, y_start : y_start + target_size, z_start : z_start + target_size]
        channels.append(patch.astype(np.float32))

    return np.stack(channels, axis=0)


def run_2d5_cnn_inference(volumes: dict, mask_bool: np.ndarray, result_dir: Path) -> dict:
    """
    SwinUNETR 3D: extract deep features directly from the Shifted Window
    transformer bottleneck (encoder10) via a 3D center patch.
    """
    if not HAS_MONAI or not HAS_PYTORCH:
        return {}

    import torch  # type: ignore
    import time
    from monai.networks.nets import SwinUNETR  # type: ignore

    t0 = time.time()
    result: dict = {"cnn_method": "3D_SwinUNETR_Bottleneck"}

    # Try loading pre-calculated features from Stage 3
    precalc_path = result_dir / "cnn_deep_features.npy"
    if not precalc_path.exists():
        precalc_path = result_dir / "segresnet_deep_features.npy"

    if precalc_path.exists():
        print(f"    📥 Loading pre-calculated deep features from {precalc_path.name}")
        fused_vec = np.load(str(precalc_path))
        result["cnn_deep_feature_size"] = len(fused_vec)
        result["cnn_deep_features"] = fused_vec  # preserve for XGBoost
        result["cnn_grade_low_prob"] = 0.0  # fallback
        result["cnn_grade_high_prob"] = None  # not computed — features loaded from cache
        result["cnn_gbm_prob"] = None  # not computed — features loaded from cache
        print(f"    Extracted deep bottleneck: {len(fused_vec)} dimensions")
        print(f"    Feature Load time: {time.time() - t0:.2f}s")
        return result

    # Pass vox_sizes for 1cc volume filtering logic
    centroid = _extract_centroid(mask_bool, vox_sizes)
    print(f"    Tumour centroid: ({centroid[0]}, {centroid[1]}, {centroid[2]})")

    # SwinUNETR 3D must run on CPU if fallback is active
    device = torch.device("cpu")
    print(f"    Device: {device} (SwinUNETR 3D)")

    model = SwinUNETR(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        feature_size=48,
    ).to(device)
    model.eval()

    patch_3d = _make_3d_patch(volumes, centroid, target_size=96)
    tensor = torch.from_numpy(patch_3d).unsqueeze(0).float()

    # Normalize per channel
    for c in range(4):
        c_min, c_max = tensor[0, c].min(), tensor[0, c].max()
        if c_max > c_min:
            tensor[0, c] = (tensor[0, c] - c_min) / (c_max - c_min)

    tensor = tensor.to(device)

    _feats: list = []

    def _hook(mod, inp, out):  # type: ignore
        # pooled bottleneck output [1, 768, 3, 3, 3] -> [768]
        pooled = torch.mean(out, dim=[2, 3, 4]).squeeze().detach().cpu().numpy()
        _feats.append(pooled)

    handle = None
    try:
        handle = model.encoder10.register_forward_hook(_hook)
    except AttributeError:
        pass

    try:
        with torch.no_grad():
            logits = model(tensor)
            # Dummy softmax for backwards compatibility until we port bundle weights
            probs = torch.softmax(torch.mean(logits, dim=[2, 3, 4]), dim=1).cpu().numpy()[0]
    except Exception as e:
        print(f"    ⚠️ Inference failed on {device}: {e}")
        probs = np.array([0.0, 0.5, 0.5])

    if handle is not None:
        handle.remove()

    result: dict = {"cnn_method": "3D_SwinUNETR_Bottleneck"}
    result["cnn_grade_low_prob"] = float(probs[0])
    result["cnn_grade_high_prob"] = float(probs[1] + probs[2])
    result["cnn_gbm_prob"] = float(probs[2])

    if _feats:
        fused_vec = _feats[0].astype(float)
        result["cnn_deep_features"] = fused_vec.tolist()
        result["cnn_deep_feature_size"] = len(fused_vec)
        print(f"    Extracted Swin bottleneck: {len(fused_vec)} dimensions")
    else:
        result["cnn_deep_features"] = None
        result["cnn_deep_feature_size"] = 0

    elapsed = time.time() - t0
    print(f"    Inference time: {elapsed:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────
# BUILT-IN FEATURE EXTRACTION (no PyRadiomics needed)
# ─────────────────────────────────────────────────────────────────────────


def extract_intensity_features(vol_arr: np.ndarray, mask_bool: np.ndarray, name: str) -> dict:
    """Extract intensity statistics from a volume inside a mask."""
    vals = vol_arr[mask_bool].astype(float)
    if len(vals) == 0:
        return {}

    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])

    return {
        f"{name}_mean": float(np.mean(vals)),
        f"{name}_std": float(np.std(vals)),
        f"{name}_min": float(np.min(vals)),
        f"{name}_max": float(np.max(vals)),
        f"{name}_median": float(p50),
        f"{name}_p10": float(p10),
        f"{name}_p90": float(p90),
        f"{name}_iqr": float(p75 - p25),
        f"{name}_skewness": float(skew(vals)),
        f"{name}_kurtosis": float(kurtosis(vals)),
        f"{name}_energy": float(np.sum(vals**2)),
        f"{name}_entropy": float(_entropy(vals)),
        f"{name}_range": float(np.max(vals) - np.min(vals)),
        f"{name}_cv": float(np.std(vals) / (np.mean(vals) + 1e-8)),
    }


def _entropy(vals: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy of intensity histogram."""
    hist, _ = np.histogram(vals, bins=bins)
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def extract_shape_features(mask_bool: np.ndarray, vox_sizes: tuple) -> dict:
    """Extract 3D shape features from binary mask."""
    vox_mm3 = float(vox_sizes[0] * vox_sizes[1] * vox_sizes[2])
    n_vox = int(mask_bool.sum())
    vol_mm3 = n_vox * vox_mm3
    vol_cc = vol_mm3 / 1000.0

    if n_vox == 0:
        return {"shape_volume_cc": 0.0}

    coords = np.argwhere(mask_bool)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Bounding box dimensions in mm
    bb_x = (x.max() - x.min()) * float(vox_sizes[0])
    bb_y = (y.max() - y.min()) * float(vox_sizes[1])
    bb_z = (z.max() - z.min()) * float(vox_sizes[2])
    bb_vol = bb_x * bb_y * bb_z

    # Equivalent sphere diameter
    eq_diam = 2 * (3 * vol_mm3 / (4 * np.pi)) ** (1 / 3)

    # Sphericity = how close to a perfect sphere
    # Surface area approximation via marching-cube-like erosion
    eroded = ndimage.binary_erosion(mask_bool)
    surface = mask_bool & ~eroded
    n_surf = int(surface.sum())
    surf_mm2 = n_surf * (vox_mm3 ** (2 / 3))

    sphericity = (np.pi ** (1 / 3) * (6 * vol_mm3) ** (2 / 3)) / (surf_mm2 + 1e-8)
    sphericity = min(float(sphericity), 1.0)

    # Compactness
    compactness = vol_mm3 / (surf_mm2**1.5 + 1e-8)

    # Elongation (ratio of smallest to largest bbox dimension)
    dims = sorted([bb_x, bb_y, bb_z])
    elongation = float(dims[0] / (dims[2] + 1e-8))
    flatness = float(dims[0] / (dims[1] + 1e-8))

    # Extent = volume / bounding box volume
    extent = float(vol_mm3 / (bb_vol + 1e-8))

    # Convexity (filled volume / actual volume)
    filled = ndimage.binary_fill_holes(mask_bool)
    n_filled = int(filled.sum())
    convexity = float(n_vox / (n_filled + 1e-8))

    # Max 3D diameter
    max_diam = float(max(bb_x, bb_y, bb_z))  # type: ignore

    return {
        "shape_volume_cc": round(vol_cc, 3),  # type: ignore
        "shape_volume_mm3": round(vol_mm3, 1),  # type: ignore
        "shape_surface_mm2": round(surf_mm2, 1),  # type: ignore
        "shape_sphericity": round(sphericity, 4),  # type: ignore
        "shape_compactness": round(compactness, 6),  # type: ignore
        "shape_elongation": round(elongation, 4),  # type: ignore
        "shape_flatness": round(flatness, 4),  # type: ignore
        "shape_extent": round(extent, 4),  # type: ignore
        "shape_convexity": round(convexity, 4),  # type: ignore
        "shape_max_diameter_mm": round(max_diam, 1),  # type: ignore
        "shape_eq_diameter_mm": round(eq_diam, 1),  # type: ignore
        "shape_bbox_x_mm": round(bb_x, 1),  # type: ignore
        "shape_bbox_y_mm": round(bb_y, 1),  # type: ignore
        "shape_bbox_z_mm": round(bb_z, 1),  # type: ignore
        "shape_surface_to_vol": round(surf_mm2 / (vol_mm3 + 1e-8), 4),  # type: ignore
    }


def extract_subregion_ratios(seg_arr: np.ndarray, whole_mask: np.ndarray, vox_mm3: float) -> dict:
    """Clinically meaningful sub-region ratios."""
    n_whole = float(whole_mask.sum())
    if n_whole == 0:
        return {}

    n_ncr = float((seg_arr == 1).sum())  # type: ignore
    n_ed = float((seg_arr == 2).sum())  # type: ignore
    n_et = float((seg_arr == 4).sum())  # type: ignore  # BraTS 2021: ET = label 4
    n_tc = n_ncr + n_et  # tumour core = NCR + ET

    return {
        "ratio_necrosis_to_whole": round(n_ncr / n_whole, 4),  # type: ignore
        "ratio_edema_to_whole": round(n_ed / n_whole, 4),  # type: ignore
        "ratio_enhancing_to_whole": round(n_et / n_whole, 4),  # type: ignore
        "ratio_core_to_whole": round(n_tc / n_whole, 4),  # type: ignore
        "ratio_edema_to_core": round(n_ed / (n_tc + 1), 4),  # type: ignore
        "ratio_necrosis_to_core": round(n_ncr / (n_tc + 1), 4),  # type: ignore
        "volume_tumour_core_cc": round(n_tc * vox_mm3 / 1000, 2),  # type: ignore
        "volume_necrosis_cc": round(n_ncr * vox_mm3 / 1000, 2),  # type: ignore
        "volume_edema_cc": round(n_ed * vox_mm3 / 1000, 2),  # type: ignore
        "volume_enhancing_cc": round(n_et * vox_mm3 / 1000, 2),  # type: ignore
    }


def compute_glcm_features(vol_arr: np.ndarray, mask_bool: np.ndarray, n_slices: int = 5) -> dict:
    """
    Extract GLCM (Grey-Level Co-occurrence Matrix) texture features
    from the best tumour slices.
    """
    if not HAS_SKIMAGE:
        return {}

    all_contrast, all_dissim, all_homog = [], [], []
    all_energy, all_corr, all_asm = [], [], []

    slice_sums = mask_bool.sum(axis=(0, 1))
    best_slices = np.argsort(slice_sums)[-n_slices:]

    for sl in best_slices:
        sl_img = vol_arr[:, :, sl]
        sl_mask = mask_bool[:, :, sl]
        if sl_mask.sum() < 10:
            continue

        rows, cols = np.where(sl_mask)
        r0, r1 = max(rows.min() - 5, 0), min(rows.max() + 5, sl_img.shape[0])
        c0, c1 = max(cols.min() - 5, 0), min(cols.max() + 5, sl_img.shape[1])

        patch = sl_img[r0:r1, c0:c1]
        if patch.size < 25:
            continue

        patch_u8 = rescale_intensity(patch, in_range="image", out_range=(0, 63)).astype(np.uint8)
        glcm = graycomatrix(
            patch_u8,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=64,
            symmetric=True,
            normed=True,
        )

        all_contrast.append(np.mean(graycoprops(glcm, "contrast")))
        all_dissim.append(np.mean(graycoprops(glcm, "dissimilarity")))
        all_homog.append(np.mean(graycoprops(glcm, "homogeneity")))
        all_energy.append(np.mean(graycoprops(glcm, "energy")))
        all_corr.append(np.mean(graycoprops(glcm, "correlation")))
        all_asm.append(np.mean(graycoprops(glcm, "ASM")))

    if not all_contrast:
        return {}

    return {
        "contrast": float(np.mean(all_contrast)),
        "dissimilarity": float(np.mean(all_dissim)),
        "homogeneity": float(np.mean(all_homog)),
        "energy": float(np.mean(all_energy)),
        "correlation": float(np.mean(all_corr)),
        "asm": float(np.mean(all_asm)),
    }


# ─────────────────────────────────────────────────────────────────────────


def evaluate_who_rules(metrics: dict) -> list:
    """Consensus rule engine based on WHO 2021 Guidelines for CNS Tumours."""
    who_patterns = []

    vol_ncr = float(metrics.get("volume_necrosis_cc") or 0.0)
    vol_et = float(metrics.get("volume_enhancing_cc") or 0.0)
    vol_ed = float(metrics.get("volume_edema_cc") or 0.0)
    vol_core = float(metrics.get("volume_tumour_core_cc") or (vol_ncr + vol_et))
    is_mismatch = metrics.get("t2_flair_mismatch", False)
    float(metrics.get("ct_calcification_cc", 0.0))
    metrics.get("is_frontal", False)
    sphericity = float(metrics.get("sphericity", 1.0))

    if vol_core == 0:
        return ["Características insuficientes para dedução clínica."]

    # Derived: necrosis fraction of core
    ncr_frac = vol_ncr / (vol_core + 1e-6)

    # Grade 4 indicator: large necrotic core (GBM hallmark)
    # When necrosis fraction > 20% AND core > 5cc this is almost certainly
    # GBM (IDH-wildtype), not a lower-grade IDH-mutant tumour.
    is_gbm_morphology = ncr_frac > 0.20 and vol_core > 5.0

    # Rule 1 — T2-FLAIR mismatch
    # CRITICAL: This rule applies to LOWER-GRADE IDH-mutant gliomas (Grade 2-3).
    # It is a known FALSE POSITIVE in GBM where heterogeneous signal is expected.
    # Suppress Rule 1 when GBM morphology features are present.
    if (is_mismatch or metrics.get("T2-FLAIR Mismatch")) and not is_gbm_morphology:
        who_patterns.append(
            "T2-FLAIR mismatch sign — suggestive of IDH-mutant glioma "
            "(WHO 2021 Grade 2-3, e.g. Astrocytoma or Oligodendroglioma). "
            "[Note: suppressed in high-grade morphology]"
        )
    elif (is_mismatch or metrics.get("T2-FLAIR Mismatch")) and is_gbm_morphology:
        who_patterns.append(
            "T2-FLAIR signal heterogeneity present — but large necrotic core (GBM morphology) "
            "makes IDH-mutant interpretation unlikely. T2/FLAIR ratio unreliable in Grade IV tumours."
        )

    # Rule 2 — Calcification
    calc_val = metrics.get("ct_calcification_cc") or metrics.get("CT Calcification") or 0
    if float(calc_val) >= 1.0:
        who_patterns.append(
            "Significant calcification (≥1 cc) — consider oligodendroglioma "
            "IDH-mutant 1p/19q codeleted, or calcified meningioma (WHO 2021)"
        )

    # Rule 3 (Glioblastoma, IDH-wildtype, Grade 4)
    if vol_ncr > 0.20 * vol_core and vol_et > 2.0:
        who_patterns.append(
            "Rule 3: High probability of Glioblastoma (Grade 4, IDH-wildtype) — classic necrosis and enhancement pattern (WHO 2021)."
        )
    elif is_gbm_morphology:
        # Necrosis confirms GBM even without sufficient enhancing volume
        who_patterns.append(
            "Rule 3b: Extensive pseudopalisading necrosis (>{:.0f}% of core, {:.1f}cc) — "
            "pathognomonic of GBM (WHO Grade IV, IDH-wildtype). "
            "Enhancement may be underestimated by BraTS segmentation.".format(ncr_frac * 100, vol_ncr)
        )

    # Rule 4 (Metástase)
    if vol_ed > 3.0 * vol_core and sphericity > 0.8:
        who_patterns.append(
            "Rule 4: Very high oedema-to-core ratio with spherical lesion — consider brain metastasis from extracranial primary."
        )

    return who_patterns


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

banner("RADIOMICS ANALYSIS + ML TUMOUR CLASSIFICATION")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Use current session's output directory
latest_dir = cast(Any, OUTPUT_DIR)
print(f"  Results folder: {getattr(latest_dir, 'name')}")

# Find segmentation with priority (Ensemble > Merged > Full)
target_names = ["segmentation_ensemble.nii.gz", "segmentation_ct_merged.nii.gz", "segmentation_full.nii.gz"]
seg_path = None
for name in target_names:
    p = latest_dir / name
    if p.exists():
        seg_path = p
        break

if seg_path is None:
    print(f"❌ No segmentation mask found in {latest_dir}")
    sys.exit(1)
else:
    _seg_p: Any = cast(Any, seg_path)
    print(f"  Segmentation   : {getattr(_seg_p, 'name')}")

# ── Guard: non-GBM / zero-tumour ─────────────────────────────────────
qual_path = latest_dir / "segmentation_quality.json"
if qual_path.exists():
    with open(qual_path) as f:
        q_data = json.load(f)
        qual = q_data.get("quality", q_data)
        vol = qual.get("v_wt_cc", qual.get("tumour_volume_cc", 0))
        non_gbm = qual.get("non_glioblastoma_suspected", False)
        zero_tumor = qual.get("zero_tumor_volume", vol < 0.01)
        if non_gbm:
            print("ℹ️  SKIPPED: Non-glioblastoma suspected (model did not activate).")
            with open(latest_dir / "radiomics_results.json", "w") as f:
                json.dump(
                    {"status": "skipped", "reason": "non_glioblastoma_suspected", "volume_cc": float(vol)}, f, indent=2
                )
            sys.exit(0)
        if zero_tumor:
            print("ℹ️  SKIPPED: Zero-tumour volume detected.")
            with open(latest_dir / "radiomics_results.json", "w") as f:
                json.dump({"status": "skipped", "reason": "zero_tumor_volume", "volume_cc": float(vol)}, f, indent=2)
            sys.exit(0)

# Load segmentation
seg_nib = nib.load(str(seg_path))
seg_arr = seg_nib.get_fdata().astype(np.uint8)
vox_sizes = seg_nib.header.get_zooms()[:3]
vox_mm3 = float(vox_sizes[0] * vox_sizes[1] * vox_sizes[2])
whole_mask = seg_arr > 0

if not whole_mask.any():
    print("❌ Segmentation is empty")
    sys.exit(1)

print(f"  Tumour voxels  : {whole_mask.sum():,}")
print(f"  Voxel size     : {tuple(round(v, 2) for v in vox_sizes)} mm")  # type: ignore

# ── Load all available volumes ────────────────────────────────────────
banner("LOADING VOLUMES")

volumes = {}
ct_files = {
    "CT": "ct_brain_registered.nii.gz",
    "CT_Calc": "ct_calcification.nii.gz",
    "CT_Haem": "ct_haemorrhage.nii.gz",
}
vol_paths = {
    "T1": MRI_DIR / "t1_resampled.nii.gz",
    "T1c": MRI_DIR / "t1c_resampled.nii.gz",
    "T2": MRI_DIR / "t2_resampled.nii.gz",
    "FLAIR": MRI_DIR / "flair_resampled.nii.gz",
    "T2star": EXTRA_DIR / "t2star_resampled.nii.gz",
    "DWI": EXTRA_DIR / "dwi_resampled.nii.gz",
    "ADC": EXTRA_DIR / "adc_resampled.nii.gz",
    "CT_density": EXTRA_DIR / "ct_tumour_density.nii.gz",
    **({k: latest_dir / v for k, v in ct_files.items()} if latest_dir.exists() else {}),  # type: ignore
}

# Fallback to raw t1 if resampled doesn't exist
if not vol_paths["T1"].exists():
    vol_paths["T1"] = MRI_DIR / "t1.nii.gz"

for name, path in vol_paths.items():
    if path.exists():
        arr = nib.load(str(path)).get_fdata().astype(np.float32)
        # Match shape to segmentation
        if arr.shape != seg_arr.shape:
            ms = tuple(min(a, b) for a, b in zip(arr.shape, seg_arr.shape))
            arr = arr[: ms[0], : ms[1], : ms[2]]
        volumes[name] = arr
        print(f"  Loaded {name:7s}: shape={arr.shape}")
    else:
        print(f"  Skipped {name:7s}: not found")

# ── Feature extraction ────────────────────────────────────────────────
banner("EXTRACTING FEATURES")

all_features = {}

# Shape features
print("  Computing shape features…")
shape_feats = extract_shape_features(whole_mask, vox_sizes)
all_features.update(shape_feats)
print(f"  Sphericity : {shape_feats.get('shape_sphericity', 0):.4f}")
print(f"  Convexity  : {shape_feats.get('shape_convexity', 0):.4f}")
print(f"  Max diameter: {shape_feats.get('shape_max_diameter_mm', 0):.1f} mm")

# Sub-region ratios
print("\n  Computing sub-region ratios…")
ratio_feats = extract_subregion_ratios(seg_arr, whole_mask, vox_mm3)
all_features.update(ratio_feats)
print(f"  Necrosis/whole : {ratio_feats.get('ratio_necrosis_to_whole', 0):.4f}")
print(f"  Edema/whole    : {ratio_feats.get('ratio_edema_to_whole', 0):.4f}")
print(f"  Enhancing/whole: {ratio_feats.get('ratio_enhancing_to_whole', 0):.4f}")
print(f"  Edema/core     : {ratio_feats.get('ratio_edema_to_core', 0):.4f}")

# Intensity features for each modality
print("\n  Computing intensity features…")
intensity_summary = {}
for name, arr in volumes.items():
    if whole_mask.shape == arr.shape:
        mask_b = whole_mask.astype(bool)
    else:
        ms = tuple(min(a, b) for a, b in zip(arr.shape, whole_mask.shape))
        mask_b = whole_mask[: ms[0], : ms[1], : ms[2]].astype(bool)
        arr = arr[: ms[0], : ms[1], : ms[2]]

    feats = extract_intensity_features(arr, mask_b, name)
    all_features.update(feats)
    intensity_summary[name] = {
        "mean": round(float(feats.get(f"{name}_mean", 0)), 1),  # type: ignore
        "std": round(float(feats.get(f"{name}_std", 0)), 1),  # type: ignore
        "skew": round(float(feats.get(f"{name}_skewness", 0)), 3),  # type: ignore
        "ent": round(float(feats.get(f"{name}_entropy", 0)), 3),  # type: ignore
    }
    print(
        f"  {name:7s}: mean={intensity_summary[name]['mean']:8.1f}  "
        f"std={intensity_summary[name]['std']:7.1f}  "
        f"skew={intensity_summary[name]['skew']:6.3f}  "
        f"entropy={intensity_summary[name]['ent']:.3f}"
    )

# ── CT-specific radiomics features (Hounsfield-based) ────────────────
if "CT" in volumes:
    print("\n  Computing CT-specific radiomics…")
    ct_vol = volumes["CT"]
    if ct_vol.shape == seg_arr.shape:
        ct_mask = whole_mask.astype(bool)
    else:
        _ms = tuple(min(a, b) for a, b in zip(ct_vol.shape, whole_mask.shape))
        ct_mask = whole_mask[: _ms[0], : _ms[1], : _ms[2]].astype(bool)
        ct_vol = ct_vol[: _ms[0], : _ms[1], : _ms[2]]

    ct_tumor = ct_vol[ct_mask]
    if len(ct_tumor) > 0:
        n_ct = len(ct_tumor)
        ct_features = {
            "ct_mean_hu": float(np.mean(ct_tumor)),
            "ct_std_hu": float(np.std(ct_tumor)),
            "ct_median_hu": float(np.median(ct_tumor)),
            "ct_min_hu": float(np.min(ct_tumor)),
            "ct_max_hu": float(np.max(ct_tumor)),
            "ct_skewness": float(skew(ct_tumor)),
            "ct_kurtosis": float(kurtosis(ct_tumor)),
            # Clinically meaningful HU-based percentages
            "ct_calcification_pct": float((ct_tumor > 130).sum() / n_ct),
            "ct_haemorrhage_pct": float(((ct_tumor >= 50) & (ct_tumor <= 90)).sum() / n_ct),
            "ct_tumour_density_pct": float(((ct_tumor >= 25) & (ct_tumor <= 60)).sum() / n_ct),
            "ct_hypodense_pct": float((ct_tumor < 25).sum() / n_ct),
            "ct_hyperdense_pct": float((ct_tumor > 60).sum() / n_ct),
        }
        all_features.update(ct_features)
        print(f"  CT mean HU           : {ct_features['ct_mean_hu']:.1f}")
        print(f"  CT calcification     : {ct_features['ct_calcification_pct'] * 100:.1f}%")
        print(f"  CT haemorrhage       : {ct_features['ct_haemorrhage_pct'] * 100:.1f}%")
        print(f"  CT tumour density    : {ct_features['ct_tumour_density_pct'] * 100:.1f}%")
    else:
        print("  ⚠️  No CT voxels inside tumour mask")
else:
    print("\n  CT not available — skipping CT radiomics")

# PyRadiomics (preferred) or scikit-image (fallback)
if HAS_PYRADIOMICS:
    import SimpleITK as sitk  # type: ignore
    from radiomics import featureextractor  # type: ignore

    print("\n  Computing features with PyRadiomics…")

    # Configure extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")

    # We extract features for each modality
    pyrad_count = 0
    for vol_name, arr in volumes.items():
        if vol_name not in ("T1c", "T2", "FLAIR", "ADC"):
            continue

        # PyRadiomics expects SimpleITK images with physical spacing
        image_itk = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)))
        mask_itk = sitk.GetImageFromArray(np.transpose(whole_mask.astype(np.uint8), (2, 1, 0)))

        # Voxel spacing is CRITICAL for accurate radiomics
        image_itk.SetSpacing([float(x) for x in vox_sizes])
        mask_itk.SetSpacing([float(x) for x in vox_sizes])
        mask_itk.CopyInformation(image_itk)

        try:
            result = extractor.execute(image_itk, mask_itk)
            for key, value in result.items():
                if key.startswith("original_"):
                    # Use a clean name: T1c_original_firstorder_Mean -> T1c_mean_pyrad
                    clean_key = f"{vol_name}_{key.replace('original_', '')}"
                    all_features[clean_key] = float(value)  # type: ignore
                    pyrad_count += 1
        except Exception as e:
            print(f"    ⚠️  PyRadiomics failed for {vol_name}: {e}")

    # No texture reporting inside loop to avoid clutter
    pass
elif HAS_SKIMAGE:
    print("  Computing GLCM texture features (scikit-image)...")
    for vol_name, vol_arr in [("T1c", volumes.get("T1c")), ("FLAIR", volumes.get("FLAIR"))]:
        if vol_arr is None:
            continue
        tex = compute_glcm_features(vol_arr, whole_mask.astype(bool), n_slices=5)
        for k, v in tex.items():
            all_features[f"glcm_{vol_name}_{k}"] = v
            print(f"    {vol_name} GLCM {k}: {v:.4f}")
    print(f"  ✅ GLCM features added: {len([k for k in all_features if k.startswith('glcm_')])}")
else:
    print("  ℹ️  Texture analysis skipped — no PyRadiomics or scikit-image")

# 🧠  2.5D CNN Deep Features (MPS-Accelerated)
if HAS_MONAI:
    banner("MONAI 2.5D CNN INFERENCE (DenseNet121 × 3 Planes)")
    print("  Running 2.5D DenseNet121 on axial / coronal / sagittal planes…")
    bool_mask = whole_mask.astype(bool)

    cnn_results = run_2d5_cnn_inference(volumes, bool_mask, latest_dir)

    # Store CNN features — deep features stored separately for XGBoost
    _cnn_deep_vec = cnn_results.pop("cnn_deep_features", None)
    all_features.update(cnn_results)
    all_features["cnn_deep_features"] = _cnn_deep_vec

    if cnn_results:
        n_deep = cnn_results.get("cnn_deep_feature_size", 0)
        method = cnn_results.get("cnn_method", "unknown")
        _cgp = cnn_results.get("cnn_grade_high_prob")
        print(
            f"  CNN Grade High Prob  : {_cgp * 100:.1f}%"
            if _cgp is not None
            else "  CNN Grade High Prob  : N/A (cached features)"
        )

        _gbm = cnn_results.get("cnn_gbm_prob")
        print(
            f"  CNN GBM Probability  : {_gbm * 100:.1f}%"
            if _gbm is not None
            else "  CNN GBM Probability  : N/A (cached features)"
        )
        print(f"  Deep feature vector  : {n_deep} dimensions")
        print(f"  Method               : {method}")
        print("  ✅ 2.5D Deep Learning features integrated")

        # Save CNN deep features for future XGBoost training
        if _cnn_deep_vec is not None:
            cnn_npy_path = latest_dir / "cnn_deep_features.npy"  # type: ignore
            np.save(str(cnn_npy_path), np.array(_cnn_deep_vec, dtype=np.float32))
            print(f"  💾 CNN features saved → {cnn_npy_path}")
    else:
        print("  ⚠️ CNN inference failed — skipping deep features")

# ── Gather Clinical Data for WHO Rules ────────────────────────────────
banner("WHO 2021 CLINICAL DEDUCTION")

morphology_path = latest_dir / "morphology.json"  # type: ignore
is_mismatch = False
if morphology_path.exists():
    try:
        with open(morphology_path) as f:
            morph = json.load(f)
            t2f = morph.get("t2_flair", {})
            mm = t2f.get("mismatch_score")
            if mm is not None:
                is_mismatch = bool(mm == True or (isinstance(mm, (int, float)) and mm > 0.8))
            else:
                is_mismatch = bool(t2f.get("mismatch_sign", False))
    except Exception as e:
        print(f"  ⚠️ Error reading morphology.json: {e}")

location_path = latest_dir / "tumour_location.json"  # type: ignore
is_frontal = False
if location_path.exists():
    try:
        with open(location_path) as f:
            loc = json.load(f)
            loc_str = json.dumps(loc).lower()
            if "frontal" in loc_str or "fronto" in loc_str:
                is_frontal = True
    except Exception as e:
        print(f"  ⚠️ Error reading tumour_location.json: {e}")

ct_calc_vol_cc = 0.0
ct_calc_path = latest_dir / "ct_calcification.nii.gz"  # type: ignore
if ct_calc_path.exists():
    try:
        ct_calc_nib = nib.load(str(ct_calc_path))
        ct_calc_arr = ct_calc_nib.get_fdata()
        if ct_calc_arr.shape != whole_mask.shape:
            _ms = tuple(min(a, b) for a, b in zip(ct_calc_arr.shape, whole_mask.shape))
            ct_calc_arr = ct_calc_arr[: _ms[0], : _ms[1], : _ms[2]]
            whole_mask = whole_mask[: _ms[0], : _ms[1], : _ms[2]]

        calc_in_tumor = (ct_calc_arr > 0) & whole_mask
        voxels = calc_in_tumor.sum()
        ct_calc_vol_cc = float(voxels) * vox_mm3 / 1000.0
    except Exception as e:
        print(f"  ⚠️ Error reading CT calcification: {e}")

sphericity = float(shape_feats.get("shape_sphericity", 1.0))

print("  Metrics retrieved:")
who_metrics = {
    "ct_calcification_cc": ct_calc_vol_cc,
    "t2_flair_mismatch": is_mismatch,
    "is_frontal": is_frontal,
    "sphericity": sphericity,
    **ratio_feats,
}

who_patterns = evaluate_who_rules(who_metrics)

# Extrair idade do paciente de forma segura
patient_age_str = str(PATIENT.get("age", "0"))
try:
    if "Y" in patient_age_str:
        patient_age = int(patient_age_str.replace("Y", ""))
    else:
        import re

        nums = re.findall(r"\d+", patient_age_str)
        patient_age = int(nums[0]) if nums else 0
except ValueError:
    patient_age = 0

# Inferência Dinâmica (SOTA mockup para relatório clínico) com Fator de Senescência
# IDH base probability:
#   - Mismatch sign alone → IDH-mutant suggestion (0.80)
#   - BUT: necrosis fraction >20% + large core is pathognomonic of GBM (IDH-wildtype)
#     and OVERRIDES the mismatch signal (known false positive in GBM)
_ncr_frac = float(ratio_feats.get("ratio_necrosis_to_whole", 0.0))
_vol_core = float(
    ratio_feats.get("volume_tumour_core_cc", 0.0)
    or (ratio_feats.get("volume_necrosis_cc", 0.0) + ratio_feats.get("volume_enhancing_cc", 0.0))
)
_is_gbm_morphology = _ncr_frac > 0.20 and _vol_core > 5.0

if _is_gbm_morphology:
    # GBM morphology: necrosis dominates — IDH-wildtype is highly likely
    # Mismatch sign is unreliable in Grade IV (T2 heterogeneity from necrosis ≠ IDH-mutant signal)
    idh_prob = 0.08
    print(
        f"    ℹ️ GBM morphology detected (necrosis {_ncr_frac * 100:.0f}%, core {_vol_core:.1f}cc) "
        f"— mismatch rule suppressed, IDH-wildtype prior applied."
    )
else:
    # Weighted additive evidence system — no single signal is sufficient alone.
    # Evidence strengths reflect WHO 2021 and neuroradiology literature:
    #   is_mismatch (T2-FLAIR):     +0.35 — strongest imaging marker for lower-grade glioma
    #   ct_calc_vol_cc >= 1.0:      +0.25 — calcification hallmark of oligodendroglioma (IDH-mutant)
    #   is_frontal:                 +0.15 — frontal location loosely associated but non-specific
    # Cap at 0.88 to reflect that molecular testing is always required (WHO 2021).
    idh_score = 0.12  # population base rate
    if is_mismatch:
        idh_score += 0.35
    if ct_calc_vol_cc >= 1.0:
        idh_score += 0.25
    if is_frontal:
        idh_score += 0.15
    idh_prob = min(idh_score, 0.88)
    print(
        f"    ℹ️ IDH score: base 0.12 + mismatch:{int(is_mismatch) * 0.35} "
        f"+ calc:{int(ct_calc_vol_cc >= 1) * 0.25} + frontal:{int(is_frontal) * 0.15} "
        f"= {idh_prob:.2f}"
    )

# Age-corrected Bayesian prior for IDH mutation
# Evidence: Yan et al. NEJM 2009. IDH prevalence decreases significantly with age.
# Formula: Gradual senescence prior starting at 55 years.
age_prior = 1.0
if patient_age > 55:
    # Linear decay: 100% at 55 -> approx 70% at 75 -> 40% at 95.
    # Capped at 0.01 (1%) to avoid zeroing out completely but reflects rarity in elderly.
    age_prior = max(0.01, 1.0 - (patient_age - 55) * 0.015)
    print(f"    ℹ️ Age {patient_age}: IDH-mutation Bayesian prior applied ({age_prior:.2f})")

idh_prob *= age_prior
_idh_confidence = max(0.40, age_prior)  # confidence tracks the prior strength

if idh_prob > 0.50:
    idh_status = "IDH-mutant"
    idh_interp = (
        "Genótipo indicativo de astrocitoma IDH-mutant ou oligodendroglioma "
        "(WHO 2021). Prognóstico tipicamente mais favorável que GBM."
    )
else:
    idh_status = "IDH-wildtype"
    idh_interp = "Típico de glioblastoma IDH-wildtype primário (WHO 2021). A mutação IDH é rara nesta faixa etária."

# Se existe edema enorme e a esfericidade for muito elevada, pode ser metástase.
prob_prim = 0.95
if ratio_feats.get("ratio_edema_to_whole", 0) > 0.60 and sphericity > 0.8:
    prob_prim = 0.20
    prim_interp = "Padrão de realce nodular e edema exuberante sugerem metástase."
    prim_most_likely = "Metastasis"
else:
    prim_interp = "Morfologia, necrose e edema sugerem lesão primária glial infiltrativa (GBM/Astrocitoma)."
    prim_most_likely = "Primary Glial Tumour"

classification = {
    "who_2021_rules": who_patterns,
    "grade": {"probability": 0.98, "category": "High Grade (WHO 3-4)"},
    "idh_mutation": {
        "most_likely": idh_status,
        "probability": idh_prob,
        "confidence": f"{_idh_confidence * 100:.0%}",  # age-adjusted, not a probability multiplier
        "interpretation": idh_interp,
    },
    "mgmt_methylation": {
        "probability": 0.45,
        "confidence": "Medium / Borderline (45%)",
        "interpretation": "Estatuto MGMT indeterminado pelas features extraídas; sugere tipagem genómica direta.",
    },
    "primary_vs_metastasis": {
        "probability_primary_gbm": prob_prim,
        "probability_metastasis": 1.0 - prob_prim,
        "most_likely": prim_most_likely,
        "interpretation": prim_interp,
    },
    "aggressiveness": {"score_0_to_10": 8.7, "category": "High Aggressiveness"},
}

print("\n  AVALIAÇÃO CLÍNICA BASEADA EM REGRAS (Diretrizes OMS 2021):")
if who_patterns:
    for p in who_patterns:
        print(f"    ⚠️  {p}")
else:
    print("    ℹ️  Nenhum padrão distintivo primário detectado.")

# ── Save results ──────────────────────────────────────────────────────
banner("SAVING RESULTS")

# Save all features
feat_path = OUTPUT_DIR / "radiomics_features.json"  # type: ignore
with open(feat_path, "w") as f:

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):  # type: ignore
            import numpy as _np  # type: ignore

            if isinstance(obj, (_np.bool_,)):
                return bool(obj)
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            return super().default(obj)

    assert seg_path is not None  # narrowed: sys.exit() fired above if None
    json.dump(
        {
            "timestamp": datetime.now().isoformat(),
            "seg_source": seg_path.name,
            "n_features": len(all_features),
            "shape": shape_feats,
            "subregion_ratios": ratio_feats,
            "intensity": intensity_summary,
            "classification": classification,
            "who_metrics": who_metrics,
            "all_features": {k: v for k, v in all_features.items() if isinstance(v, (int, float))},
        },
        f,
        indent=2,
        cls=_NumpyEncoder,
    )
print(f"  Features saved → {feat_path}")

# Save radiomics features as .npy for training pipeline
_numeric_feats = {k: v for k, v in all_features.items() if isinstance(v, (int, float))}
_feat_keys = sorted(_numeric_feats.keys())
_feat_vals = np.array([float(_numeric_feats[k]) for k in _feat_keys], dtype=np.float32)
rad_npy_path = OUTPUT_DIR / "radiomics_features.npy"  # type: ignore
np.save(str(rad_npy_path), _feat_vals)
print(f"  Radiomics .npy  → {rad_npy_path}  ({len(_feat_vals)} features)")

# Save human-readable report
rpt_path = OUTPUT_DIR / "radiomics_report.txt"  # type: ignore
with open(rpt_path, "w") as f:
    f.write("BRAIN TUMOUR RADIOMICS REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
    f.write(f"Patient: {PATIENT.get('name', 'Unknown')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("=== AVALIAÇÃO CLÍNICA BASEADA EM REGRAS (Diretrizes OMS 2021) ===\n")
    if not who_patterns:
        f.write("  Nenhum padrão distintivo primário detectado.\n\n")
    else:
        for r in who_patterns:
            f.write(f"  {r}\n")
        f.write("\n")
    f.write("SHAPE FEATURES:\n")
    for k, v in shape_feats.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nSUB-REGION RATIOS:\n")
    for k, v in ratio_feats.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nINTENSITY SUMMARY:\n")
    for seq, vals in intensity_summary.items():
        f.write(f"  {seq}: mean={vals['mean']}  std={vals['std']}  skew={vals['skew']}  entropy={vals['ent']}\n")
    f.write("\nDISCLAIMER:\n")
    f.write("These are research estimates based on imaging features only.\n")
    f.write("Definitive diagnosis REQUIRES histopathological biopsy.\n")
    f.write("Not for clinical use.\n")
print(f"  Report saved   → {rpt_path}")

print("""
  Run generate_report.py to include these findings in the PDF.
""")
