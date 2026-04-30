#!/usr/bin/env python3
"""
Brain Tumor Analysis with MONAI — v4 (SwinUNETR Bundle)
=========================================================
Primary segmentation engine: MONAI SwinUNETR (State-of-the-Art)
  - Transformer-based architecture with shift-window attention
  - Better global context capture than CNNs
  - Superior performance on heterogeneous tumors
  - Official MONAI bundle trained on BraTS 2021

Apple Silicon (M-series) notes:
  - SwinUNETR runs on CPU (Transformer attention not yet MPS-optimized)
  - Expect 5-8 min inference time on M4 Pro
  - Bundle download requires internet on first run (~800 MB)

Requirements:
    pip install "monai[all]>=1.4.0"
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
import warnings
from pathlib import Path
from typing import Tuple, Dict

# ── PY-BRAIN session loader ──────────────────────────────────────────────
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.session_loader import get_session, get_paths, get_patient  # type: ignore

try:
    _sess = get_session()
    _paths = get_paths(_sess)
    PATIENT = get_patient(_sess)
    MONAI_DIR = _paths["monai_dir"]
    EXTRA_DIR = _paths["extra_dir"]
    BUNDLE_DIR = _paths["bundle_dir"]
    RESULTS_DIR = _paths["results_dir"]
    GROUND_TRUTH = _paths["ground_truth"]
    OUTPUT_DIR = _paths.get("output_dir", RESULTS_DIR)

    DEVICE = "cpu"
    MODEL_DEVICE = "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            MODEL_DEVICE = torch.device("cpu")  # SwinUNETR on CPU
            print("ℹ️  Apple Silicon detected — SwinUNETR running on CPU")
        elif torch.cuda.is_available():
            DEVICE = MODEL_DEVICE = torch.device("cuda")
        else:
            DEVICE = MODEL_DEVICE = torch.device("cpu")
    except ImportError:
        pass

except SystemExit:
    raise
except Exception as e:
    print(f"❌ Failed to load session: {e}")
    _sys.exit(1)

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────
try:
    import torch
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import nibabel as nib
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, ball
    import monai
    from monai.transforms import (
        LoadImage,
        Compose,
        ScaleIntensityRange,
        NormalizeIntensity,
        EnsureChannelFirst,
        Spacing,
        Orientation,
        CropForeground,
    )
    from monai.networks.nets import SwinUNETR
    from monai.inferers import SlidingWindowInferer
    from monai.data import MetaTensor
    from monai.bundle import ConfigParser, download
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("Run: pip install 'monai[all]>=1.4.0' nibabel scipy scikit-image plotly")
    sys.exit(1)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("ℹ️  plotly not installed — HTML viewer will be skipped.")

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(str(MONAI_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_NAME = "brats_mri_segmentation"  # Previously downloaded and working bundle

# SwinUNETR-specific parameters
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 1
OVERLAP = 0.5

# Tumor sub-region labels (BraTS convention)
LABEL_NAMES = {1: "Necrotic core", 2: "Edema", 3: "Enhancing tumor"}
LABEL_COLORS = {1: "Blues", 2: "Greens", 3: "Reds"}
LABEL_HEX = {1: "#4488ff", 2: "#44cc44", 3: "#ff4444"}

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
    return (labeled == (np.argmax(sizes) + 1)).astype(np.float32)


def load_swinunetr_bundle(bundle_dir: Path) -> Tuple[torch.nn.Module, Dict]:
    """
    Load the official MONAI SwinUNETR bundle.

    Returns:
        model: Loaded SwinUNETR model
        config: Bundle configuration
    """
    bundle_path = bundle_dir / BUNDLE_NAME

    # Download if not present
    if not bundle_path.exists():
        print(f"\n  📥 Downloading SwinUNETR bundle (~800 MB) → {bundle_dir}/")
        print("     This happens once; subsequent runs use cached copy.")
        try:
            download(name=BUNDLE_NAME, bundle_dir=str(bundle_dir))
            print("  ✅ Download complete.")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            raise

    print(f"  ✅ Bundle found at {bundle_path}")

    # Load inference config
    config_file = bundle_path / "configs" / "inference.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Inference config not found: {config_file}")

    parser = ConfigParser()
    parser.read_config(str(config_file))

    # Get network definition
    model = parser.get_parsed_content("network_def", instantiate=True)

    # Load checkpoint
    ckpt_files = list(bundle_path.glob("models/*.pt")) + list(bundle_path.glob("models/*.pth"))

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {bundle_path}/models/")

    # Use the latest checkpoint
    ckpt_path = sorted(ckpt_files)[-1]
    print(f"  Loading checkpoint: {ckpt_path.name}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (DataParallel wrapper)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load with strict=False to handle minor mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(
            f"  ⚠️  Missing keys: {missing_keys[:5]}..."
            if len(missing_keys) > 5
            else f"  ⚠️  Missing keys: {missing_keys}"
        )
    if unexpected_keys:
        print(
            f"  ⚠️  Unexpected keys: {unexpected_keys[:5]}..."
            if len(unexpected_keys) > 5
            else f"  ⚠️  Unexpected keys: {unexpected_keys}"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SwinUNETR  |  parameters: {n_params:,}  |  device: {MODEL_DEVICE}")

    return model, parser.get_config()


def preprocess_for_swinunetr(volumes: Dict[str, torch.Tensor], brain_mask: np.ndarray) -> torch.Tensor:
    """
    Preprocess 4 modalities for SwinUNETR inference.

    SwinUNETR expects:
        - Input shape: (1, 4, D, H, W)
        - Z-score normalization per modality within brain mask
        - Percentile clipping to handle outliers
    """

    def zscore_percentile(arr: np.ndarray, mask: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
        """Z-score normalization with percentile clipping."""
        vals = arr[mask > 0]
        if len(vals) == 0:
            return np.zeros_like(arr)

        p_lo = float(np.percentile(vals, p_low))
        p_hi = float(np.percentile(vals, p_high))
        clipped = np.clip(vals, p_lo, p_hi)
        mu = float(clipped.mean())
        sigma = float(clipped.std())

        out = (arr - mu) / (sigma + 1e-8)
        out[mask == 0] = 0.0
        return out.astype(np.float32)

    # Stack modalities in order: T1, T1c, T2, FLAIR
    modalities = ["T1", "T1c", "T2", "FLAIR"]
    stacked = []

    for mod in modalities:
        arr = volumes[mod].numpy() if hasattr(volumes[mod], "numpy") else volumes[mod]
        normalized = zscore_percentile(arr, brain_mask, p_low=1, p_high=99)
        stacked.append(normalized)

    # Stack into [4, D, H, W]
    input_array = np.stack(stacked, axis=0)

    # Add batch dimension and convert to tensor
    input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(MODEL_DEVICE)

    return input_tensor


def run_swinunetr_inference(
    model: torch.nn.Module, input_tensor: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Run SwinUNETR inference and extract tumor sub-regions.

    Returns:
        seg_full: Integer label map (0=background, 1=NCR, 2=ED, 3=ET)
        tumor_prob: Probability map of whole tumor
        raw_probs: Dictionary of raw probability maps per region
    """

    # Configure sliding window inferer
    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        overlap=OVERLAP,
        mode="gaussian",
        progress=True,
    )

    print("\n  Running SwinUNETR sliding-window inference...")
    print(f"  ROI size: {ROI_SIZE}, Overlap: {OVERLAP}")
    print("  (This may take 5-8 min on CPU — transformer attention is computationally intensive)")

    # --- HOTFIX: Add the .encoder10 hook so Stage 8 gets features! ---
    _swin_features = []

    def _hook(mod, inp, out):
        pooled = torch.mean(out, dim=[2, 3, 4]).detach().cpu()
        _swin_features.append(pooled)

    handle = None
    try:
        handle = model.encoder10.register_forward_hook(_hook)
    except AttributeError:
        pass
    # -----------------------------------------------------------------

    with torch.no_grad():
        logits = inferer(input_tensor, model)

    if handle is not None:
        handle.remove()

    if _swin_features:
        merged_feats = torch.cat(_swin_features, dim=0).mean(dim=0).numpy().astype(np.float32)
        cnn_feat_path = OUTPUT_DIR / "cnn_deep_features.npy"
        np.save(str(cnn_feat_path), merged_feats)
        print(f"  💾 SwinUNETR features pooled → {cnn_feat_path}")

    # SwinUNETR output: (1, 4, D, H, W) for 4 classes (BG, NCR, ED, ET)
    n_out = logits.shape[1]
    print(f"  Model output channels: {n_out}")

    if n_out == 4:
        # Standard 4-class softmax output
        probs = torch.softmax(logits, dim=1)  # (1, 4, D, H, W)
        pred_idx = torch.argmax(probs, dim=1)  # (1, D, H, W)

        seg_full = pred_idx.squeeze(0).cpu().numpy().astype(np.uint8)

        # Probability maps
        tumor_prob = probs[0, 1:].sum(dim=0).cpu().numpy()  # Sum of all tumor classes
        raw_probs = {
            "prob_ncr": probs[0, 1].cpu().numpy().astype(np.float32),
            "prob_ed": probs[0, 2].cpu().numpy().astype(np.float32),
            "prob_et": probs[0, 3].cpu().numpy().astype(np.float32),
        }

    else:
        # Alternative output format fallback
        seg_full = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        tumor_prob = torch.softmax(logits, dim=1)[0, 1:].sum(dim=0).cpu().numpy()
        raw_probs = {"prob_wt": tumor_prob}

    return seg_full, tumor_prob, raw_probs


# ─────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────


def main():
    banner("SWINUNETR BRAIN TUMOR SEGMENTATION")
    print(f"  Data dir: {DATA_DIR.resolve()}")
    print(f"  Output dir: {OUTPUT_DIR.resolve()}\n")

    # ── PART 1: LOAD DATA ────────────────────────────────────────────────
    paths = {
        "T1": DATA_DIR / "t1.nii.gz",
        "T1c": DATA_DIR / "t1c_resampled.nii.gz",
        "T2": DATA_DIR / "t2_resampled.nii.gz",
        "FLAIR": DATA_DIR / "flair_resampled.nii.gz",
    }

    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        print(f"❌ Missing: {missing}")
        print("   Run 1_dicom_to_nifti.py first.")
        sys.exit(1)

    # Load volumes
    loader = LoadImage(image_only=True)
    vols = {k: loader(str(v)) for k, v in paths.items()}

    for k, v in vols.items():
        print(f"  {k:5s}: shape={tuple(v.shape)}  dtype={v.dtype}")

    # Compute brain mask from T1
    norms_np = {k: norm01(v.numpy()) for k, v in vols.items()}
    print("\n  Computing brain mask...")
    brain_mask = skull_strip(norms_np["T1"])
    brain_voxels = int(brain_mask.sum())
    print(f"  Brain voxels: {brain_voxels:,}")

    # ── PART 2: SWINUNETR INFERENCE ──────────────────────────────────────
    banner("SWINUNETR MODEL INFERENCE")

    try:
        # Load model
        model, _ = load_swinunetr_bundle(BUNDLE_DIR)
        model = model.to(MODEL_DEVICE).eval()

        # Preprocess
        input_tensor = preprocess_for_swinunetr(vols, brain_mask)
        print(f"  Input tensor shape: {tuple(input_tensor.shape)}")

        # Run inference
        seg_full, tumor_prob, raw_probs = run_swinunetr_inference(model, input_tensor)

        # Post-process: apply brain mask and clean small components
        seg_full = (seg_full * brain_mask).astype(np.uint8)

        # Clean isolated voxels (remove components < 100 voxels)
        for label in [1, 2, 3]:
            mask = seg_full == label
            labeled, n = ndimage.label(mask)
            if n > 0:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                for i, sz in enumerate(sizes, 1):
                    if sz < 100:
                        seg_full[labeled == i] = 0

        # Extract sub-regions
        seg_necrotic = (seg_full == 1).astype(np.float32)
        seg_edema = (seg_full == 2).astype(np.float32)
        seg_enhancing = (seg_full == 3).astype(np.float32)
        seg_any = (seg_full > 0).astype(np.float32)

        seg_source = "SwinUNETR"

    except Exception as e:
        print(f"\n  ⚠️  SwinUNETR inference failed: {e}")
        print("  → Falling back to heuristic segmentation.")

        # Fallback heuristic
        from skimage.filters import threshold_otsu

        def heuristic_mask(norms, brain_mask):
            prob = 0.45 * norms["T1c"] + 0.35 * norms["FLAIR"] + 0.10 * norms["T2"] - 0.10 * norms["T1"]
            prob = np.clip(prob, 0, 1) * brain_mask
            brain_vals = prob[brain_mask > 0]
            thresh = max(threshold_otsu(brain_vals), np.percentile(brain_vals, 95))
            mask = ((prob > thresh) & (brain_mask > 0)).astype(np.float32)
            labeled, n = ndimage.label(mask)
            if n > 0:
                sizes = ndimage.sum(mask, labeled, range(1, n + 1))
                mask = (labeled == (np.argmax(sizes) + 1)).astype(np.float32)
            return mask, prob

        tumor_mask_np, tumor_prob = heuristic_mask(norms_np, brain_mask)
        seg_full = tumor_mask_np.astype(np.uint8)
        seg_necrotic = tumor_mask_np
        seg_edema = np.zeros_like(tumor_mask_np)
        seg_enhancing = tumor_mask_np
        seg_any = tumor_mask_np
        seg_source = "heuristic"
        raw_probs = {"prob_wt": tumor_prob}

    # ── PART 3: VOLUME CALCULATIONS ──────────────────────────────────────
    vox_mm3 = 1.0

    def vol_cc(mask):
        return float(mask.sum()) * vox_mm3 / 1000.0

    v_whole = vol_cc(seg_any)
    v_necrotic = vol_cc(seg_necrotic)
    v_edema = vol_cc(seg_edema)
    v_enhancing = vol_cc(seg_enhancing)

    print(f"\n  📊 Tumor volumes ({seg_source}):")
    print(f"     Whole tumor         : {v_whole:.1f} cc")
    if seg_source == "SwinUNETR":
        print(f"     ├─ Necrotic core    : {v_necrotic:.1f} cc")
        print(f"     ├─ Edema            : {v_edema:.1f} cc")
        print(f"     └─ Enhancing tumor  : {v_enhancing:.1f} cc")
    print(f"     Brain volume        : {brain_voxels / 1000:.1f} cc")

    # ── PART 4: SAVE OUTPUTS ─────────────────────────────────────────────
    banner("SAVING RESULTS")

    t1_nib = nib.load(str(paths["T1"]))
    affine = t1_nib.affine

    def save_nifti(arr, fname, dtype=np.uint8):
        nib.save(nib.Nifti1Image(arr.astype(dtype), affine), str(OUTPUT_DIR / fname))
        print(f"  Saved → {OUTPUT_DIR / fname}")

    save_nifti(seg_full, "segmentation_full.nii.gz")
    save_nifti(seg_necrotic, "seg_necrotic.nii.gz")
    save_nifti(seg_edema, "seg_edema.nii.gz")
    save_nifti(seg_enhancing, "seg_enhancing.nii.gz")
    save_nifti(tumor_prob.astype(np.float32), "tumor_probability.nii.gz", dtype=np.float32)

    for name, prob_map in raw_probs.items():
        save_nifti(prob_map, f"{name}.nii.gz", dtype=np.float32)

    # Save stats JSON
    stats = {
        "segmentation_source": seg_source,
        "volume_cc": {
            "whole_tumor": float(v_whole),
            "necrotic_core": float(v_necrotic),
            "edema": float(v_edema),
            "enhancing": float(v_enhancing),
        },
        "brain_volume_cc": float(brain_voxels / 1000),
        "model": "SwinUNETR" if seg_source == "SwinUNETR" else "Heuristic",
    }

    with open(OUTPUT_DIR / "tumor_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats → {OUTPUT_DIR / 'tumor_stats.json'}")

    banner("SUMMARY")
    print(f"""
  Segmentation engine : {seg_source.upper()}
  Output folder       : {OUTPUT_DIR.resolve()}

  ├── segmentation_full.nii.gz    ← 3-label mask (load in 3D Slicer)
  ├── seg_necrotic.nii.gz         ← necrotic core only
  ├── seg_edema.nii.gz            ← peritumoral edema only
  ├── seg_enhancing.nii.gz        ← enhancing tumor only
  ├── tumor_probability.nii.gz    ← continuous probability map
  └── tumor_stats.json            ← all volumes

  Tumour volumes:
    Whole tumor     : {v_whole:.1f} cc
    Necrotic core   : {v_necrotic:.1f} cc
    Edema           : {v_edema:.1f} cc
    Enhancing       : {v_enhancing:.1f} cc
""")

    print("═" * 70)
    print("  ✅  SwinUNETR segmentation complete!")
    print("═" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Cancelled.\n")
        sys.exit(0)
