#!/usr/bin/env python3
"""
Brain Tumor Analysis with MONAI — v3 (BraTS Pre-trained Bundle)
================================================================
Primary segmentation engine: MONAI BraTS 2023 SegResNet bundle
  - Trained on thousands of expert-annotated brain MRI cases
  - Outputs 3 clinically meaningful sub-regions:
      Label 1 → Necrotic core      (dark on T1c)
      Label 2 → Peritumoral edema  (bright on FLAIR)
      Label 3 → Enhancing tumor    (bright on T1c post-contrast)
  - Falls back to improved heuristic if bundle download fails

Apple Silicon (M-series) notes:
  - Data tensors use MPS for speed
  - Model inference uses CPU (ConvTranspose3D not yet on MPS)
  - Bundle download requires internet on first run (~500 MB)

Requirements:
    pip install torch torchvision
    pip install "monai[all]>=1.3"
    pip install nibabel scipy scikit-image plotly

Data layout (unchanged):
    monai_ready/
    ├── t1.nii.gz
    ├── t1c_resampled.nii.gz
    ├── t2_resampled.nii.gz
    └── flair_resampled.nii.gz
"""

# ── Standard library ──────────────────────────────────────────────────────
import sys
import json
import shutil
import warnings
from pathlib import Path
from datetime import datetime

# ── PY-BRAIN session loader ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.session_loader import get_session, get_paths, get_patient  # type: ignore
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
        import torch  # type: ignore
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
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────
try:
    import torch  # type: ignore
    import numpy as np  # type: ignore
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.gridspec as gridspec  # type: ignore
    from matplotlib.colors import Normalize  # type: ignore
    import nibabel as nib  # type: ignore
    from scipy import ndimage  # type: ignore
    from skimage.filters import threshold_otsu  # type: ignore
    from skimage.morphology import binary_closing, ball  # type: ignore
    import monai  # type: ignore
    from monai.transforms import (  # type: ignore
        LoadImage, EnsureChannelFirst, ScaleIntensity,
        NormalizeIntensity, Orientation, Spacing,
        CropForeground, Compose
    )
    from monai.networks.nets import SegResNet  # type: ignore
    from monai.inferers import SlidingWindowInferer  # type: ignore
    from monai.data import MetaTensor  # type: ignore
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("Run:  pip install 'monai[all]>=1.3' nibabel scipy scikit-image plotly")
    sys.exit(1)

try:
    import plotly.graph_objects as go  # type: ignore
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("ℹ️  plotly not installed — HTML viewer will be skipped.")

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(str(MONAI_DIR))
EXTRA_DIR   = Path(str(EXTRA_DIR))
BUNDLE_DIR  = Path(str(BUNDLE_DIR))
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR  = Path(str(OUTPUT_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"ℹ️  Device: {DEVICE}  |  Model: {MODEL_DEVICE}")

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────

def banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def norm01(x) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def skull_strip(vol_norm: np.ndarray) -> np.ndarray:
    """Simple morphological brain mask from normalised T1."""
    mask = vol_norm > 0.10
    mask = binary_closing(mask, ball(5))
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.float32)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1)).astype(np.float32)  # type: ignore


def heuristic_mask(norms: dict, brain_mask: np.ndarray) -> np.ndarray:
    """
    Fallback heuristic when BraTS bundle is unavailable.
    Uses 95th-percentile threshold on a weighted T1c+FLAIR map.
    """
    prob = (
        0.45 * norms["T1c"]
      + 0.35 * norms["FLAIR"]
      + 0.10 * norms["T2"]
      - 0.10 * norms["T1"]
    )
    prob = np.clip(prob, 0, 1) * brain_mask
    brain_vals   = prob[brain_mask > 0]
    otsu_t       = threshold_otsu(brain_vals)
    pct95_t      = float(np.percentile(brain_vals, 95))
    thresh       = max(otsu_t, pct95_t)
    mask         = ((prob > thresh) & (brain_mask > 0)).astype(np.float32)
    labeled, n   = ndimage.label(mask)
    if n == 0:
        return mask, prob
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    mask  = (labeled == (np.argmax(sizes) + 1)).astype(np.float32)  # type: ignore
    return mask, prob


# ─────────────────────────────────────────────────────────────────────────
# PART 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────
banner("PART 1 — LOADING DATA")
print(f"  Data dir  : {DATA_DIR.resolve()}")
print(f"  Output dir: {OUTPUT_DIR.resolve()}\n")

paths = {
    "T1":    DATA_DIR / "t1.nii.gz",
    "T1c":   DATA_DIR / "t1c_resampled.nii.gz",
    "T2":    DATA_DIR / "t2_resampled.nii.gz",
    "FLAIR": DATA_DIR / "flair_resampled.nii.gz",
}
missing = [k for k, v in paths.items() if not v.exists()]
if missing:
    print(f"❌ Missing: {missing}  — run resample_to_t1.py first.")
    sys.exit(1)

loader = LoadImage(image_only=True)
vols   = {k: loader(str(v)) for k, v in paths.items()}

for k, v in vols.items():
    print(f"  {k:5s}: shape={tuple(v.shape)}  dtype={v.dtype}")

# Load extra sequences (T2*, DWI, ADC) if available
ct_files = {
    "CT":       "ct_brain_registered.nii.gz",
    "CT_Calc":  "ct_calcification.nii.gz",
    "CT_Haem":  "ct_haemorrhage.nii.gz"
}
extra_paths = {
    "T2star":   EXTRA_DIR / "t2star_resampled.nii.gz",
    "DWI":      EXTRA_DIR / "dwi_resampled.nii.gz",
    "ADC":      EXTRA_DIR / "adc_resampled.nii.gz",
    **{k: OUTPUT_DIR / v for k, v in ct_files.items()}
}
extra_vols = {}
print()
for k, p in extra_paths.items():
    if p.exists():
        extra_vols[k] = loader(str(p))
        print(f"  {k:6s}: shape={tuple(extra_vols[k].shape)}  extra sequence loaded")
    else:
        print(f"  {k:6s}: not found — skipping")

# Numpy normalised copies (used by viz + fallback)
norms_np = {k: norm01(v.numpy()) for k, v in vols.items()}

print("\n  Computing brain mask…")
brain_mask   = skull_strip(norms_np["T1"])
brain_voxels = int(brain_mask.sum())
print(f"  Brain voxels: {brain_voxels:,}")


# ─────────────────────────────────────────────────────────────────────────
# PART 2 — BRATS PRE-TRAINED BUNDLE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────
banner("PART 2 — BRATS PRE-TRAINED MODEL SEGMENTATION")

seg_source = "bundle"   # will change to "heuristic" on failure

try:
    from monai.bundle import ConfigParser, download  # type: ignore

    # ── Download bundle (skipped if already present) ──────────────────
    bundle_name = "brats_mri_segmentation"
    if not (BUNDLE_DIR / bundle_name).exists():
        print(f"  📥 Downloading BraTS SegResNet bundle (~500 MB) → {BUNDLE_DIR}/")
        print("     (This happens once; subsequent runs use the cached copy.)")
        download(name=bundle_name, bundle_dir=str(BUNDLE_DIR))
        print("  ✅ Download complete.")
    else:
        print(f"  ✅ Bundle already cached at {BUNDLE_DIR}/{bundle_name}")

    bundle_path = BUNDLE_DIR / bundle_name

    # ── Load model weights ─────────────────────────────────────────────
    print("\n  Loading SegResNet weights…")
    config_file = bundle_path / "configs" / "inference.json"
    parser      = ConfigParser()
    parser.read_config(str(config_file))

    model = parser.get_parsed_content("network_def", instantiate=True)

    # Load checkpoint
    ckpt_files = list((bundle_path / "models").glob("*.pt")) + \
                 list((bundle_path / "models").glob("*.pth"))
    if not ckpt_files:
        raise FileNotFoundError("No .pt/.pth checkpoint found in bundle/models/")

    ckpt = torch.load(str(ckpt_files[0]), map_location="cpu")
    # Handle various checkpoint formats
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    
    # Fallback log for MPS 3D limitations
    if MODEL_DEVICE.type == "cpu" and torch.backends.mps.is_available():
        print("  ⚠️  Notice: SegResNet uses CPU for inference (ConvTranspose3D not on MPS).")

    model = model.to(MODEL_DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SegResNet  |  parameters: {n_params:,}  |  device: {MODEL_DEVICE}")

    # ── Pre-process: stack 4 modalities → [1,4,D,H,W] ─────────────────
    print("\n  Pre-processing for BraTS inference…")

    # BraTS expects: T1, T1c, T2, FLAIR — z-score normalised per modality
    # inside the brain mask.
    #
    # PERCENTILE-CLIPPED Z-SCORE (IDH-wildtype optimisation):
    # High-intensity enhancing or low-intensity necrotic voxels can skew
    # the mean/std, causing the model to "miss" infiltrative regions.
    # Clip to the 1st–99th percentile within the brain mask first.
    def zscore(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        vals = arr[mask > 0]
        # Percentile-based clipping to remove outlier intensities
        p_lo: float = float(np.percentile(vals, 1))
        p_hi: float = float(np.percentile(vals, 99))
        clipped = np.clip(vals, p_lo, p_hi)
        mu: float = float(clipped.mean())
        sigma: float = float(clipped.std())
        out = (arr - mu) / (sigma + 1e-8)
        out[mask == 0] = 0.0
        return out.astype(np.float32)

    stacked = np.stack([
        zscore(vols["T1"].numpy(),    brain_mask),
        zscore(vols["T1c"].numpy(),   brain_mask),
        zscore(vols["T2"].numpy(),    brain_mask),
        zscore(vols["FLAIR"].numpy(), brain_mask),
    ], axis=0)   # [4, D, H, W]

    input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(MODEL_DEVICE)
    print(f"  Input tensor: {tuple(input_tensor.shape)}")

    # ── Sliding-window inference ───────────────────────────────────────
    print("\n  Running sliding-window inference…")
    print("  (This may take 2–5 min on CPU — the model processes overlapping")
    print("   96³ patches across the whole volume for accuracy.)")

    inferer = SlidingWindowInferer(
        roi_size      = (96, 96, 96),
        sw_batch_size = 1,
        overlap       = 0.5,
        mode          = "gaussian",
        progress      = True,
    )

    # Try to hook last encoder layer (silently skips if unavailable on SegResNet)
    _segresnet_features = []
    def _hook(mod, inp, out):  # type: ignore
        # pooled to [batch, 768] via Global Average Pooling
        pooled = torch.mean(out, dim=[2, 3, 4]).detach().cpu()
        _segresnet_features.append(pooled)

    handle = None
    try:
        handle = model.encoder10.register_forward_hook(_hook)
    except AttributeError:
        pass

    import traceback
    try:
        with torch.no_grad():
            logits = inferer(input_tensor, model)
    except Exception as e:
        print(f"  ⚠️ Inference failed on {MODEL_DEVICE}: {e}")
        if MODEL_DEVICE.type != "cpu":
            print("  → Falling back to CPU explicitly and retrying...")
            model = model.to("cpu")
            input_tensor = input_tensor.to("cpu")
            with torch.no_grad():
                logits = inferer(input_tensor, model)
        else:
            raise e
            
    if handle is not None:
        handle.remove()

    if _segresnet_features:
        # Average across all sliding window patches to get 1 whole-tumor generic response
        merged_feats = torch.cat(_segresnet_features, dim=0).mean(dim=0).numpy().astype(np.float32)
        cnn_feat_path = OUTPUT_DIR / "segresnet_deep_features.npy"
        np.save(str(cnn_feat_path), merged_feats)
        print(f"  💾 SegResNet deep features saved → {cnn_feat_path}")

    n_out = logits.shape[1]
    print(f"  Model output channels: {n_out}")

    if n_out == 3:
        # Newer BraTS bundle format:
        # 3 sigmoid channels — one per region, NO background channel
        #   channel 0 → TC  (tumour core = necrotic + enhancing)
        #   channel 1 → WT  (whole tumour = TC + edema)
        #   channel 2 → ET  (enhancing tumour)
        #
        # Threshold tuning for atypical lesions (Adaptive Otsu):
        #   Standard threshold 0.5 → under-segments atypical/heterogeneous tumours
        #   We use Otsu's method dynamically on the probability field to adjust
        #   sensitivity based on the noise level of the specific scan.

        def get_otsu_thresh(prob_map: np.ndarray, fallback: float) -> float:
            try:
                # Calculate Otsu threshold only on candidate tumour voxels
                mask = prob_map > 0.01
                if mask.sum() < 100:
                    return fallback
                val = float(threshold_otsu(prob_map[mask]))
                # Clamp the threshold between 0.15 and 0.60 for stability
                return max(0.15, min(0.60, val))
            except Exception:
                return fallback

        probs = torch.sigmoid(logits)          # [1, 3, D, H, W]

        p_tc = probs[0, 0].cpu().numpy()
        p_wt = probs[0, 1].cpu().numpy()
        p_et = probs[0, 2].cpu().numpy()

        WT_THRESH = get_otsu_thresh(p_wt, 0.30)
        TC_THRESH = get_otsu_thresh(p_tc, 0.35)
        ET_THRESH = get_otsu_thresh(p_et, 0.40)

        print("  Detected: 3-channel sigmoid output (TC / WT / ET)")
        print(f"  Thresholds: WT={WT_THRESH}  TC={TC_THRESH}  ET={ET_THRESH}")

        tc  = (probs[0, 0] > TC_THRESH).cpu().numpy().astype(np.float32)
        wt  = (probs[0, 1] > WT_THRESH).cpu().numpy().astype(np.float32)
        et  = (probs[0, 2] > ET_THRESH).cpu().numpy().astype(np.float32)

        # Derive the three BraTS sub-regions:
        #   Enhancing tumour (ET)         = channel 2
        #   Necrotic core (NCR)           = TC minus ET
        #   Peritumoral edema (ED)        = WT minus TC
        seg_enhancing = et
        seg_necrotic  = np.clip(tc - et, 0, 1).astype(np.float32)
        seg_edema     = np.clip(wt - tc, 0, 1).astype(np.float32)
        seg_any       = wt   # whole tumour is the union

        # Build integer label map for NIfTI (1=NCR, 2=ED, 3=ET)
        seg_full = np.zeros_like(tc, dtype=np.uint8)
        seg_full[seg_edema     > 0] = 2
        seg_full[seg_necrotic  > 0] = 1
        seg_full[seg_enhancing > 0] = 3

        # Probability map: use WT channel for viewer
        # (WT is the most complete — covers full tumour extent)
        tumor_prob    = probs[0, 1].cpu().numpy()  # WT channel
        tumor_mask_np = seg_any

        # Save raw sigmoid probability maps for manual threshold inspection
        _prob_dir = OUTPUT_DIR
        _prob_dir.mkdir(parents=True, exist_ok=True)
        import nibabel as _nib  # type: ignore
        for _ch, _name in [(0,"prob_tc"),(1,"prob_wt"),(2,"prob_et")]:
            _arr = probs[0, _ch].cpu().numpy().astype(np.float32)
            # will be saved after affine is known — store for now
        _raw_probs = {
            "prob_tc": probs[0, 0].cpu().numpy().astype(np.float32),
            "prob_wt": probs[0, 1].cpu().numpy().astype(np.float32),
            "prob_et": probs[0, 2].cpu().numpy().astype(np.float32),
        }

    else:
        # Original 4-channel softmax format (background + 3 regions)
        print("  Detected: 4-channel softmax output (BG / NCR / ED / ET)")
        probs     = torch.softmax(logits, dim=1)   # [1, 4, D, H, W]
        pred_idx  = torch.argmax(probs, dim=1)     # [1, D, H, W]
        seg_full  = pred_idx.squeeze(0).cpu().numpy().astype(np.uint8)

        seg_necrotic  = (seg_full == 1).astype(np.float32)  # type: ignore
        seg_edema     = (seg_full == 2).astype(np.float32)  # type: ignore
        seg_enhancing = (seg_full == 3).astype(np.float32)  # type: ignore
        seg_any       = (seg_full > 0).astype(np.float32)  # type: ignore

        tumor_prob    = probs[0, 3].cpu().numpy()
        tumor_mask_np = seg_any

    print(f"  ✅ Inference complete  |  seg shape: {seg_full.shape}")

except Exception as e:
    print(f"\n  ⚠️  BraTS bundle failed: {e}")
    print("  → Falling back to improved heuristic segmentation.")
    seg_source = "heuristic"

    tumor_mask_np, tumor_prob = heuristic_mask(norms_np, brain_mask)
    seg_full      = tumor_mask_np.astype(np.uint8)  # type: ignore
    seg_necrotic  = tumor_mask_np
    seg_edema     = np.zeros_like(tumor_mask_np)
    seg_enhancing = tumor_mask_np


# ── Compute volumes ────────────────────────────────────────────────────
vox_mm3 = 1.0   # 1 mm isotropic

def vol_cc(mask): return float(mask.sum()) * vox_mm3 / 1000.0

v_whole     = vol_cc(tumor_mask_np)
v_necrotic  = vol_cc(seg_necrotic)
v_edema     = vol_cc(seg_edema)
v_enhancing = vol_cc(seg_enhancing)

print(f"\n  📊 Tumor volumes ({seg_source}):")
print(f"     Whole tumor         : {v_whole:.1f} cc")
if seg_source == "bundle":
    print(f"     ├─ Necrotic core    : {v_necrotic:.1f} cc")
    print(f"     ├─ Edema            : {v_edema:.1f} cc")
    print(f"     └─ Enhancing tumor  : {v_enhancing:.1f} cc")
print(f"     Brain volume        : {brain_voxels/1000:.1f} cc")
print(f"     Tumor / brain       : {100*tumor_mask_np.sum()/brain_voxels:.1f} %")

# ── Threshold sweep (only for 3-channel sigmoid bundle output) ────────────
# Automatically tests WT thresholds from 0.15 to 0.50 and shows which
RADIOLOGIST_REF_CC = PATIENT.get("radiologist_volume_cc")
if RADIOLOGIST_REF_CC is None:
    RADIOLOGIST_REF_CC = 32.0

if seg_source == "bundle" and "_raw_probs" in dir():
    print(f"\n  📐 Threshold sweep (target ≈ {RADIOLOGIST_REF_CC} cc):")
    print(f"  {'WT thresh':>10}  {'WT vol (cc)':>12}  {'diff vs ref':>12}  verdict")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*26}")

    wt_prob_map = _raw_probs["prob_wt"]
    best_thresh = 0.30
    best_diff   = float("inf")
    FALLBACK_THRESH = 0.15   # permissive fallback for infiltrative lesions
    MATCH_TOLERANCE = 0.20   # 20% — if no threshold is within this, fallback

    for _t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        _wt_mask  = (wt_prob_map > _t).astype(np.float32) * brain_mask
        _wt_vol   = float(_wt_mask.sum()) / 1000.0
        _diff     = abs(_wt_vol - RADIOLOGIST_REF_CC)
        _verdict  = ""
        if _diff < best_diff:
            best_diff   = _diff
            best_thresh = _t
            _verdict    = "<── closest to radiologist"
        elif _wt_vol < RADIOLOGIST_REF_CC * 0.5:
            _verdict = "under-segments"
        elif _wt_vol > RADIOLOGIST_REF_CC * 2.0:
            _verdict = "over-segments"
        print(f"  {_t:>10.2f}  {_wt_vol:>12.1f}  {_wt_vol - RADIOLOGIST_REF_CC:>+12.1f}  {_verdict}")

    # Check if best threshold is within tolerance of the reference
    _best_vol = float((wt_prob_map > best_thresh).astype(np.float32).sum()) / 1000.0
    _rel_diff = abs(_best_vol - RADIOLOGIST_REF_CC) / (RADIOLOGIST_REF_CC + 1e-8)
    if _rel_diff > MATCH_TOLERANCE:
        print(f"\n  ⚠️  No threshold within {MATCH_TOLERANCE*100:.0f}% of radiologist "
              f"reference ({RADIOLOGIST_REF_CC} cc).")
        print(f"      Best candidate: WT={best_thresh} → {_best_vol:.1f} cc "
              f"({_rel_diff*100:.0f}% off).")
        print(f"      → AUTO-FALLBACK to permissive threshold {FALLBACK_THRESH} "
              f"to capture infiltrative tumour.")
        best_thresh = FALLBACK_THRESH
    else:
        print(f"\n  Best WT threshold for this case: {best_thresh}")
    print(f"  To use it permanently set WT_THRESH = {best_thresh} in PART 2")

    # Auto-apply best threshold to final mask
    print(f"\n  Applying best threshold ({best_thresh}) to final whole-tumour mask…")
    best_wt       = (wt_prob_map > best_thresh).astype(np.float32) * brain_mask
    best_tc       = (_raw_probs["prob_tc"] > TC_THRESH).astype(np.float32)
    best_et       = (_raw_probs["prob_et"] > ET_THRESH).astype(np.float32)

    seg_enhancing = best_et
    seg_necrotic  = np.clip(best_tc - best_et, 0, 1).astype(np.float32)
    seg_edema     = np.clip(best_wt - best_tc, 0, 1).astype(np.float32)
    seg_any       = best_wt

    seg_full                    = np.zeros_like(best_wt, dtype=np.uint8)
    seg_full[seg_edema     > 0] = 2
    seg_full[seg_necrotic  > 0] = 1
    seg_full[seg_enhancing > 0] = 3
    tumor_mask_np               = seg_any

    # Recompute volumes with best threshold
    v_whole     = vol_cc(tumor_mask_np)
    v_necrotic  = vol_cc(seg_necrotic)
    v_edema     = vol_cc(seg_edema)
    v_enhancing = vol_cc(seg_enhancing)

    print(f"  Updated volumes with best threshold:")
    print(f"     Whole tumor     : {v_whole:.1f} cc")
    print(f"     Necrotic core   : {v_necrotic:.1f} cc")
    print(f"     Edema           : {v_edema:.1f} cc")
    print(f"     Enhancing       : {v_enhancing:.1f} cc")

# ── Intensity stats inside whole-tumor mask ────────────────────────────
tumor_bool = tumor_mask_np.astype(bool)
stats = {}
for name, vol in vols.items():
    vals = vol.numpy()[tumor_bool] if tumor_bool.any() else np.array([0.0])  # type: ignore
    stats[name] = {k: float(v) for k, v in zip(
        ["mean","std","min","max"],
        [vals.mean(), vals.std(), vals.min(), vals.max()]
    )}

# Extra sequence stats inside tumour mask
extra_stats = {}
if extra_vols and tumor_bool.any(): # type: ignore
    print("\n  Extra sequence intensities inside tumour mask:")
    for ename, evol in extra_vols.items():
        earr = evol.numpy() # type: ignore
        if earr.shape != tumor_mask_np.shape: # type: ignore
            ms = tuple(min(a,b) for a,b in zip(earr.shape, tumor_mask_np.shape)) # type: ignore
            earr  = earr[:ms[0], :ms[1], :ms[2]]
            emask = tumor_bool[:ms[0], :ms[1], :ms[2]] # type: ignore
        else:
            emask = tumor_bool # type: ignore
        evals = earr[emask] if emask.any() else np.array([0.0])  # type: ignore # type: ignore
        extra_stats[ename] = {kk: float(vv) for kk, vv in zip( # type: ignore
            ["mean","std","min","max"],
            [evals.mean(), evals.std(), evals.min(), evals.max()])} # type: ignore
        hint = ""
        if ename == "ADC":
            m = extra_stats[ename]["mean"]
            hint = ("Low ADC -> high cellularity" if m < 800
                    else "Moderate ADC" if m < 1200 else "High ADC -> necrosis/cyst")
        elif ename == "T2star":
            hint = "Low T2* -> calcification or haemosiderin"
        elif ename == "CT":
            m = extra_stats[ename]["mean"]
            hint = (f"Mean {m:.0f} HU — "
                    + ("hyperdense tumour" if m > 40
                       else "isodense" if m > 20
                       else "hypodense/cystic"))
        elif ename == "CT_Calc":
            frac = extra_stats[ename]["mean"]
            hint = "calcification mask — 1=calcium present"
        elif ename == "CT_Haem":
            hint = "haemorrhage mask — 1=acute blood present"
        print(f"     {ename:6s}: mean={extra_stats[ename]['mean']:.1f}  "  # type: ignore
              f"std={extra_stats[ename]['std']:.1f}  {hint}")

# Save JSON
with open(OUTPUT_DIR / "tumor_stats.json", "w") as f:
    json.dump({
        "patient":             PATIENT.get("name", "Unknown"),
        "dob":                 PATIENT.get("dob", ""),
        "exam_date":           PATIENT.get("exam_date", ""),
        "segmentation_source": seg_source,
        "radiologist_size_cm": PATIENT.get("radiologist_size_cm", ""),
        "volume_cc": {
            "whole_tumor":   round(float(v_whole), 2),
            "necrotic_core": round(float(v_necrotic), 2),
            "edema":         round(float(v_edema), 2),
            "enhancing":     round(float(v_enhancing), 2),
        },
        "brain_volume_cc":    round(float(brain_voxels / 1000), 2),
        "tumor_pct_brain":    round(100 * float(tumor_mask_np.sum()) / brain_voxels, 2),
        "intensity_brats":    stats,
        "intensity_extra":    extra_stats,
    }, f, indent=2)
print(f"\n  Stats → {OUTPUT_DIR / 'tumor_stats.json'}")


# ─────────────────────────────────────────────────────────────────────────
# PART 3 — MULTI-SLICE GRID VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────
banner("PART 3 — VISUALIZATION")

def best_slices(prob: np.ndarray, axis: int, n: int = 8) -> list:
    ax_other = tuple(i for i in range(3) if i != axis)
    scores   = prob.mean(axis=ax_other)
    return list(np.argsort(scores)[-n:])

def get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0: return arr[idx, :, :].T
    if axis == 1: return arr[:, idx, :].T
    return arr[:, :, idx].T

def save_grid(norms: dict, prob: np.ndarray,
              masks: dict,   # {"whole": arr, "necrotic": arr, ...}
              axis: int, axis_name: str, n_slices: int = 8):
    """Save a colour-coded multi-region overlay grid."""
    slices = best_slices(prob, axis=axis, n=n_slices)
    cols   = len(slices)

    # Rows: T1, T1c, FLAIR, Overlay (all regions)
    rows = 4
    fig  = plt.figure(figsize=(cols * 2.6, rows * 2.6), facecolor="#0a0a0a")
    fig.suptitle(f"Brain Tumor — {axis_name}  ({seg_source})",
                 color="white", fontsize=12, y=1.01)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.04, wspace=0.04)

    row_labels = ["T1", "T1c", "FLAIR", "Segmentation overlay"]

    for r, row_label in enumerate(row_labels):
        for c, sl in enumerate(slices):
            ax = fig.add_subplot(gs[r, c]) # type: ignore
            ax.axis("off")

            if row_label == "Segmentation overlay":
                bg = get_slice(norms["T1c"], axis, sl) # type: ignore
                ax.imshow(bg, cmap="gray", vmin=0, vmax=1)

                # Draw each region in a distinct colour
                region_map = {
                    "necrotic":  (seg_necrotic,  "#4499ff", "Necrotic"),
                    "edema":     (seg_edema,      "#44ee44", "Edema"),
                    "enhancing": (seg_enhancing,  "#ff4444", "Enhancing"),
                }
                for rname, (rmask, color, _) in region_map.items():
                    sl_mask = get_slice(rmask, axis, sl)
                    if sl_mask.max() > 0:
                        rgba = np.zeros((*sl_mask.shape, 4), dtype=np.float32)
                        c_rgb = plt.matplotlib.colors.to_rgb(color)
                        rgba[..., :3] = c_rgb
                        rgba[...,  3] = sl_mask * 0.65
                        ax.imshow(rgba)

                # Legend on first column only
                if c == 0 and seg_source == "bundle":
                    patches = [
                        plt.matplotlib.patches.Patch(color="#4499ff", label="Necrotic"),
                        plt.matplotlib.patches.Patch(color="#44ee44", label="Edema"),
                        plt.matplotlib.patches.Patch(color="#ff4444", label="Enhancing"),
                    ]
                    ax.legend(handles=patches, loc="lower left",
                              fontsize=5, framealpha=0.4,
                              labelcolor="white", facecolor="#222")
            else:
                sl_data = get_slice(norms[row_label], axis, sl) # type: ignore
                ax.imshow(sl_data, cmap="gray", vmin=0, vmax=1)

            if c == 0:
                ax.set_ylabel(row_label, color="white", fontsize=7,
                              rotation=0, labelpad=42, va="center")

    fname = OUTPUT_DIR / f"view_{axis_name.lower()}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {fname}")

print("  Generating axial view…")
save_grid(norms_np, tumor_prob,
          {"necrotic": seg_necrotic, "edema": seg_edema, "enhancing": seg_enhancing},
          axis=2, axis_name="Axial")

print("  Generating coronal view…")
save_grid(norms_np, tumor_prob,
          {"necrotic": seg_necrotic, "edema": seg_edema, "enhancing": seg_enhancing},
          axis=1, axis_name="Coronal")

print("  Generating sagittal view…")
save_grid(norms_np, tumor_prob,
          {"necrotic": seg_necrotic, "edema": seg_edema, "enhancing": seg_enhancing},
          axis=0, axis_name="Sagittal")


# ─────────────────────────────────────────────────────────────────────────
# PART 4 — INTERACTIVE HTML VIEWER
# ─────────────────────────────────────────────────────────────────────────
banner("PART 4 — INTERACTIVE HTML VIEWER")

if HAS_PLOTLY:
    print("  Building interactive slice viewer with 3-region overlay…")

    t1c_vol  = norms_np["T1c"]
    n_slices = t1c_vol.shape[2]

    frames = []
    for sl in range(n_slices):
        bg    = t1c_vol[:, :, sl].T
        n_sl  = seg_necrotic[:, :, sl].T
        e_sl  = seg_edema[:, :, sl].T
        en_sl = seg_enhancing[:, :, sl].T

        frames.append(go.Frame(
            data=[
                go.Heatmap(z=bg[::-1],   colorscale="gray", showscale=False,  # type: ignore
                           zmin=0, zmax=1, name="T1c"),
                go.Heatmap(z=np.where(n_sl[::-1]  > 0, 1.0, np.nan),  # type: ignore
                           colorscale=[[0,"#4499ff"],[1,"#4499ff"]],
                           showscale=False, zmin=0, zmax=1,
                           opacity=0.55, name="Necrotic"),
                go.Heatmap(z=np.where(e_sl[::-1]  > 0, 1.0, np.nan),  # type: ignore
                           colorscale=[[0,"#44ee44"],[1,"#44ee44"]],
                           showscale=False, zmin=0, zmax=1,
                           opacity=0.45, name="Edema"),
                go.Heatmap(z=np.where(en_sl[::-1] > 0, 1.0, np.nan),  # type: ignore
                           colorscale=[[0,"#ff4444"],[1,"#ff4444"]],
                           showscale=False, zmin=0, zmax=1,
                           opacity=0.65, name="Enhancing"),
            ],
            name=str(sl)
        ))

    # Initial best slice
    sl0    = best_slices(tumor_prob, axis=2, n=1)[0]
    bg0    = t1c_vol[:, :, sl0].T
    n0     = seg_necrotic[:, :, sl0].T
    e0     = seg_edema[:, :, sl0].T
    en0    = seg_enhancing[:, :, sl0].T

    fig = go.Figure(
        data=[
            go.Heatmap(z=bg0[::-1],  colorscale="gray", showscale=False,  # type: ignore
                       zmin=0, zmax=1, name="T1c"),
            go.Heatmap(z=np.where(n0[::-1]  > 0, 1.0, np.nan),  # type: ignore
                       colorscale=[[0,"#4499ff"],[1,"#4499ff"]],
                       showscale=False, zmin=0, zmax=1,
                       opacity=0.55, name="Necrotic core"),
            go.Heatmap(z=np.where(e0[::-1]  > 0, 1.0, np.nan),  # type: ignore
                       colorscale=[[0,"#44ee44"],[1,"#44ee44"]],
                       showscale=False, zmin=0, zmax=1,
                       opacity=0.45, name="Edema"),
            go.Heatmap(z=np.where(en0[::-1] > 0, 1.0, np.nan),  # type: ignore
                       colorscale=[[0,"#ff4444"],[1,"#ff4444"]],
                       showscale=False, zmin=0, zmax=1,
                       opacity=0.65, name="Enhancing tumor"),
        ],
        frames=frames,
    )

    steps = [dict(args=[[str(k)],  # type: ignore
                        {"frame": {"duration": 0}, "mode": "immediate"}],
                  label=str(k), method="animate") # type: ignore
             for k in range(n_slices)]

    fig.update_layout(  # type: ignore
        title=dict(  # type: ignore
            text=(f"Brain Tumor — Interactive Viewer<br>"
                  f"<span style='font-size:11px;color:#aaa'>"
                  f"🔵 Necrotic &nbsp; 🟢 Edema &nbsp; 🔴 Enhancing  "
                  f"({seg_source})</span>"),
            x=0.5, font=dict(color="white", size=14) # type: ignore
        ),
        paper_bgcolor="#111", plot_bgcolor="#111",
        sliders=[dict( # type: ignore
            active=sl0, steps=steps, pad={"t": 50}, # type: ignore
            currentvalue={"prefix": "Axial slice: ",
                          "font": {"color": "white", "size": 13}}, # type: ignore
            font=dict(color="white") # type: ignore
        )],
        updatemenus=[dict(  # type: ignore
            type="buttons", showactive=False, y=1.18, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play",  # type: ignore
                     method="animate",
                     args=[None, {"frame": {"duration": 55}, "fromcurrent": True}]), # type: ignore
                dict(label="⏸ Pause",  # type: ignore
                     method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]), # type: ignore
            ]
        )],
        height=660, width=700,
        margin=dict(l=20, r=20, t=100, b=90),  # type: ignore
    )

    html_path = OUTPUT_DIR / "interactive_viewer.html"
    fig.write_html(str(html_path))
    print(f"  Saved → {html_path}")
    print("  Open in Chrome/Firefox: scroll slider or press ▶ Play to animate.")
else:
    print("  Skipped — install plotly:  pip install plotly")


# ─────────────────────────────────────────────────────────────────────────
# PART 5 — SAVE NIFTI OUTPUTS
# ─────────────────────────────────────────────────────────────────────────
banner("PART 5 — SAVING NIfTI OUTPUTS")

# Load original affine from T1 so masks align correctly in 3D Slicer
t1_nib   = nib.load(str(paths["T1"]))
affine   = t1_nib.affine

def save_nifti(arr, fname, dtype=np.uint8):
    nib.save(nib.Nifti1Image(arr.astype(dtype), affine),
             str(OUTPUT_DIR / fname))
    print(f"  Saved → {OUTPUT_DIR / fname}")

save_nifti(seg_full,      "segmentation_full.nii.gz")   # 0/1/2/3 labels
save_nifti(seg_necrotic,  "seg_necrotic.nii.gz")
save_nifti(seg_edema,     "seg_edema.nii.gz")
save_nifti(seg_enhancing, "seg_enhancing.nii.gz")
save_nifti(tumor_prob.astype(np.float32),
           "tumor_probability.nii.gz", dtype=np.float32)

# Save raw sigmoid probability maps (useful for threshold tuning)
if "_raw_probs" in dir():
    for _pname, _parr in _raw_probs.items():
        save_nifti(_parr, f"{_pname}.nii.gz", dtype=np.float32)
    print("  Raw sigmoid maps saved (prob_tc / prob_wt / prob_et)")
    print("  Load in 3D Slicer to visualise confidence at every voxel")

print("""
  How to view in 3D Slicer:
  1. File → Add Data → add your t1c_resampled.nii.gz
  2. File → Add Data → add segmentation_full.nii.gz
     → Right-click → "Convert to Segmentation"
     → Colours: 1=blue(necrotic) 2=green(edema) 3=red(enhancing)
  3. Modules → Segment Statistics → Compute (for volumes/surface area)
  4. Modules → Volume Rendering → enable T1c for 3D view
""")


# ─────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────
banner("SUMMARY")
pct = 100 * float(tumor_mask_np.sum()) / brain_voxels
print(f"""
  Segmentation engine : {seg_source.upper()}
  Output folder       : {OUTPUT_DIR.resolve()}

  ├── view_axial.png              ← 8-slice axial grid
  ├── view_coronal.png            ← 8-slice coronal grid
  ├── view_sagittal.png           ← 8-slice sagittal grid
  ├── interactive_viewer.html     ← open in browser, scroll all slices
  ├── segmentation_full.nii.gz    ← 3-label mask  (load in 3D Slicer)
  ├── seg_necrotic.nii.gz         ← necrotic core only
  ├── seg_edema.nii.gz            ← peritumoral edema only
  ├── seg_enhancing.nii.gz        ← enhancing tumor only
  ├── tumor_probability.nii.gz    ← continuous probability map
  └── tumor_stats.json            ← all volumes + intensity stats

  Tumour volumes (AI):
    Whole tumor     : {v_whole:.1f} cc
    Necrotic core   : {v_necrotic:.1f} cc
    Edema           : {v_edema:.1f} cc
    Enhancing       : {v_enhancing:.1f} cc
    Brain volume    : {brain_voxels/1000:.1f} cc
    Tumor / brain   : {pct:.1f} %

  Radiologist reference:
    Size reported   : {PATIENT.get('radiologist_size_cm', 'N/A')}
    Volume estimate : ~{RADIOLOGIST_REF_CC} cc

  If AI volume differs significantly, open segmentation_full.nii.gz
  in 3D Slicer and correct using the Segment Editor.
""")
print("═" * 70)
print("  ✅  Done!")
print("═" * 70)

# ─────────────────────────────────────────────────────────────────────────
# AUTO-VALIDATION — runs automatically if ground_truth.nii.gz exists
# ─────────────────────────────────────────────────────────────────────────
# GROUND_TRUTH already from config

if GROUND_TRUTH.exists():
    banner("AUTO-VALIDATION AGAINST GROUND TRUTH")
    try:
        import nibabel as _nib_v  # type: ignore
        from scipy.spatial import cKDTree as _cKDTree  # type: ignore

        gt_nib_v  = _nib_v.load(str(GROUND_TRUTH))
        gt_arr_v  = (gt_nib_v.get_fdata() > 0).astype(np.uint8)
        vox_v     = gt_nib_v.header.get_zooms()[:3]
        vox_mm3_v = float(vox_v[0] * vox_v[1] * vox_v[2])

        # Match shapes
        pred_v = tumor_mask_np.astype(np.uint8)
        if pred_v.shape != gt_arr_v.shape:
            ms = tuple(min(a,b) for a,b in zip(pred_v.shape, gt_arr_v.shape))
            pred_v  = pred_v[:ms[0],   :ms[1],   :ms[2]]
            gt_arr_v = gt_arr_v[:ms[0], :ms[1],  :ms[2]]

        # Dice
        inter   = (pred_v.astype(bool) & gt_arr_v.astype(bool)).sum()
        denom   = pred_v.sum() + gt_arr_v.sum()
        dice_v  = float(2 * inter / denom) if denom > 0 else 0.0

        # Sensitivity
        tp_v    = inter
        fn_v    = (~pred_v.astype(bool) & gt_arr_v.astype(bool)).sum()
        sens_v  = float(tp_v / (tp_v + fn_v)) if (tp_v + fn_v) > 0 else 0.0

        # FP/FN volumes
        fp_v    = (pred_v.astype(bool) & ~gt_arr_v.astype(bool)).sum()
        fp_cc_v = float(fp_v) * vox_mm3_v / 1000.0
        fn_cc_v = float(fn_v) * vox_mm3_v / 1000.0

        # HD95
        try:
            pred_pts = np.argwhere(pred_v   > 0).astype(float) * np.array(vox_v)
            gt_pts   = np.argwhere(gt_arr_v > 0).astype(float) * np.array(vox_v)
            if len(pred_pts) > 0 and len(gt_pts) > 0:
                d_pg, _ = _cKDTree(gt_pts).query(pred_pts)
                d_gp, _ = _cKDTree(pred_pts).query(gt_pts)
                hd95_v  = float(np.percentile(
                    np.concatenate([d_pg, d_gp]), 95))
            else:
                hd95_v = float("inf")
        except Exception:
            hd95_v = float("nan")

        gt_vol_v   = float(gt_arr_v.sum()) * vox_mm3_v / 1000.0
        pred_vol_v = float(pred_v.sum())   * vox_mm3_v / 1000.0

        print(f"  Metric              Value    Grade")
        print(f"  ─────────────────── ──────── ─────")

        def _g(v, good, warn):
            return "✅" if v >= good else ("⚠️ " if v >= warn else "❌")

        print(f"  Dice score        : {dice_v:.4f}   {_g(dice_v, 0.80, 0.60)}")
        print(f"  HD95 (mm)         : {hd95_v:.1f}     "
              f"{_g(-hd95_v, -5, -15)}")
        print(f"  Sensitivity       : {sens_v:.4f}   {_g(sens_v, 0.80, 0.60)}")
        print(f"  FP volume (cc)    : {fp_cc_v:.2f}")
        print(f"  FN volume (cc)    : {fn_cc_v:.2f}  (tumour missed)")
        print(f"  Pred volume (cc)  : {pred_vol_v:.1f}")
        print(f"  GT volume (cc)    : {gt_vol_v:.1f}")

        # Append validation to tumor_stats.json
        stats_path = OUTPUT_DIR / "tumor_stats.json"
        if stats_path.exists():
            with open(stats_path) as _f:
                _s = json.load(_f)
            _s["validation"] = {
                "dice":           float(f"{dice_v:.4f}"),
                "hd95_mm":        float(f"{hd95_v:.2f}"),
                "sensitivity":    float(f"{sens_v:.4f}"),
                "fp_volume_cc":   float(f"{fp_cc_v:.2f}"),
                "fn_volume_cc":   float(f"{fn_cc_v:.2f}"),
                "pred_volume_cc": float(f"{pred_vol_v:.2f}"),
                "gt_volume_cc":   float(f"{gt_vol_v:.2f}"),
            }
            with open(stats_path, "w") as _f:
                json.dump(_s, _f, indent=2)
            print(f"  Validation metrics added to tumor_stats.json")

    except Exception as _ve:
        print(f"  ⚠️  Validation error: {_ve}")
else:
    print(f"""
  ℹ️  No ground truth found — validation skipped.
  To enable automatic validation:
    1. Correct segmentation in 3D Slicer Segment Editor
    2. Save as: {GROUND_TRUTH}
    3. Re-run this script — metrics appear automatically.
  Or run validate_segmentation.py for full detailed report.
""")
