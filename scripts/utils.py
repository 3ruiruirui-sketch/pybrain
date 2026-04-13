"""
PY-BRAIN — Shared Utilities
============================
Helper functions used across all pipeline scripts.
"""

import logging
import sys
import warnings
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
    
    def get_stable_device():
        """Detect MPS/CUDA and perform a proactive test for 3D operations."""
        import torch
        if not torch.backends.mps.is_available():
            if torch.cuda.is_available(): return torch.device("cuda")
            return torch.device("cpu")
        
        # Proactive MPS Check for 3D Layers (Hardware compatibility for M4 Pro etc.)
        dev = torch.device("mps")
        try:
            _t = torch.zeros(1, 1, 4, 4, 4, device=dev)
            torch.nn.functional.conv3d(_t, torch.zeros(1, 1, 3, 3, 3, device=dev), padding=1)
            # Test trilinear interpolation (used in 8b and resampling)
            torch.nn.functional.interpolate(_t, size=(2,2,2), mode="trilinear", align_corners=False)
            return dev
        except (RuntimeError, NotImplementedError):
            return torch.device("cpu")

    DEVICE = get_stable_device()
    MODEL_DEVICE = DEVICE # Default to detected stable device
    if str(DEVICE) != "cpu":
        print(f"  ⚡ Hardware acceleration enabled: {DEVICE}")
    else:
        print("  ℹ️  Using CPU (either no GPU or MPS 3D layers not supported on this Mac)")
except ImportError:
    DEVICE = MODEL_DEVICE = "cpu"

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

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import ball, closing
from skimage.filters import threshold_otsu

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "py-brain") -> logging.Logger:
    """Return a configured logger with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def banner(title: str):
    """Print a section banner to stdout."""
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────
# ARRAY UTILITIES
# ─────────────────────────────────────────────────────────────────────────

def norm01(x) -> np.ndarray:
    """Min-max normalise array to [0, 1]."""
    x = np.array(x, dtype=np.float32)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def match_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop or pad array to match target shape."""
    out = np.zeros(target_shape, dtype=arr.dtype)
    s   = tuple(min(a, b) for a, b in zip(arr.shape, target_shape))
    out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]
    return out


def vol_cc(mask: np.ndarray, vox_mm3: float = 1.0) -> float:
    """Convert voxel count to cubic centimetres."""
    return float(mask.astype(bool).sum()) * vox_mm3 / 1000.0


# ─────────────────────────────────────────────────────────────────────────
# NIFTI I/O
# ─────────────────────────────────────────────────────────────────────────

def load_nifti(path: Path) -> tuple:
    """Load NIfTI file. Returns (array, affine, header)."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine, img.header


def save_nifti(arr: np.ndarray, affine: np.ndarray,
               path: Path, dtype=np.uint8):
    """Save array as NIfTI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(dtype), affine), str(path))


# ─────────────────────────────────────────────────────────────────────────
# BRAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────

def skull_strip(vol_norm: np.ndarray) -> np.ndarray:
    """
    Simple morphological brain mask from normalised T1.
    Thresholds at 10% of max, closes holes, keeps largest component.
    """
    mask = vol_norm > 0.10
    mask = closing(mask, footprint=ball(5))
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(np.float32)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1)).astype(np.float32)


def heuristic_mask(norms: dict,
                   brain_mask: np.ndarray) -> tuple:
    """
    Fallback tumour mask when BraTS bundle unavailable.
    Uses 95th-percentile threshold on weighted T1c+FLAIR map.
    Returns (mask, probability_map).
    """
    prob = (
        0.45 * norms["T1c"]
      + 0.35 * norms["FLAIR"]
      + 0.10 * norms["T2"]
      - 0.10 * norms["T1"]
    )
    prob = np.clip(prob, 0, 1) * brain_mask
    brain_vals = prob[brain_mask > 0]
    otsu_t     = threshold_otsu(brain_vals)
    pct95_t    = float(np.percentile(brain_vals, 95))
    thresh     = max(otsu_t, pct95_t)
    mask       = ((prob > thresh) & (brain_mask > 0)).astype(np.float32)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask, prob
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    mask  = (labeled == (np.argmax(sizes) + 1)).astype(np.float32)
    return mask, prob


# ─────────────────────────────────────────────────────────────────────────
# RESULTS MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────

def make_output_dir(base: Path) -> Path:
    """Create a timestamped results folder."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"results_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_latest_results(base: Path) -> Path:
    """Return the most recent output folder.
    First checks OUTPUT_DIR from session, then searches all subdirectories.
    """
    # 1) Try OUTPUT_DIR from session first
    if OUTPUT_DIR and Path(OUTPUT_DIR).is_dir():
        return Path(OUTPUT_DIR)
    # 2) Fall back to any subdirectory (sorted by name for recency)
    candidates = sorted([
        d for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if not candidates:
        raise FileNotFoundError(f"No results folders in {base}")
    return candidates[-1]
