import sys
import json
from typing import Optional, Iterable
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# ── PY-BRAIN session loader ──────────────────────────────────────────
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts.session_loader import get_session, get_paths, get_patient
except ImportError:
    from session_loader import get_session, get_paths

_sess = get_session()
_paths = get_paths(_sess)
PROJECT_ROOT = Path(_sess["project_root"])
OUTPUT_DIR = Path(_paths["output_dir"])
MONAI_DIR = Path(_paths["monai_dir"])
REPO_DIR = PROJECT_ROOT / "models" / "brainIAC"
SRC_DIR = REPO_DIR / "src"
REPORT_PATH = OUTPUT_DIR / "radiomics_features.json"
# ─────────────────────────────────────────────────────────────────────

BASE_CKPT = REPO_DIR / "BrainIAC.ckpt"
IDH_CKPT = REPO_DIR / "idh.ckpt"
PLACEHOLDER_CKPT = REPO_DIR / "weights" / "idh_weights.pth"

TARGET_SIZE = (96, 96, 96)


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def get_stable_device():
    # Hardware-Specific Optimization (M4 Pro) - Standardized Check
    import torch

    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        try:
            _t = torch.zeros(1, 1, 4, 4, 4, device=dev)
            # BrainIAC uses 3D trilinear interpolation for resampling inputs to 96x96x96
            torch.nn.functional.interpolate(_t, size=(2, 2, 2), mode="trilinear", align_corners=False)
            return dev
        except (RuntimeError, NotImplementedError):
            return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = get_stable_device()
print(f"🚀 Initializing on device: {DEVICE}")


def resolve_input_path(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def load_nifti(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {arr.shape}")

    std = float(arr.std())
    if std > 0:
        arr = (arr - arr.mean()) / (std + 1e-8)

    return arr


def compute_tumour_crop_bbox(seg_path: Path, margin_mm: float = 15.0) -> Optional[tuple]:
    """
    Compute a tight bounding box around the largest tumour component.
    Returns (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates,
    or None if no tumour is found.
    Margin is added in mm and converted to voxels using the image zooms.
    """
    seg_nib = nib.load(str(seg_path))
    seg_arr = seg_nib.get_fdata()
    vox_sizes = seg_nib.header.get_zooms()[:3]

    tumour_mask = seg_arr > 0
    if not tumour_mask.any():
        return None

    # Keep only the largest connected component to avoid scattered fragments
    labeled, n_comp = ndimage.label(tumour_mask)
    if n_comp > 1:
        sizes = ndimage.sum(tumour_mask, labeled, range(1, n_comp + 1))
        largest_label = int(np.argmax(sizes)) + 1
        tumour_mask = labeled == largest_label

    coords = np.argwhere(tumour_mask)
    if coords.size == 0:
        return None

    # Convert mm margin to voxels per axis
    margin_vox = tuple(int(np.ceil(margin_mm / vs)) for vs in vox_sizes)

    x_min = max(0, int(coords[:, 0].min()) - margin_vox[0])
    x_max = min(seg_arr.shape[0] - 1, int(coords[:, 0].max()) + margin_vox[0])
    y_min = max(0, int(coords[:, 1].min()) - margin_vox[1])
    y_max = min(seg_arr.shape[1] - 1, int(coords[:, 1].max()) + margin_vox[1])
    z_min = max(0, int(coords[:, 2].min()) - margin_vox[2])
    z_max = min(seg_arr.shape[2] - 1, int(coords[:, 2].max()) + margin_vox[2])

    return (x_min, x_max, y_min, y_max, z_min, z_max)


def crop_to_bbox(arr: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crop a 3D volume to the given bounding box.
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return arr[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]


def resize_to_96(arr: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]
    x = x.permute(0, 1, 4, 2, 3)  # [1,1,D,H,W]
    x = F.interpolate(x, size=TARGET_SIZE, mode="trilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)  # [D,H,W]
    return x


def _find_segmentation() -> Optional[Path]:
    """Locate the best available segmentation in OUTPUT_DIR."""
    candidates = [
        OUTPUT_DIR / "segmentation_ct_merged.nii.gz",
        OUTPUT_DIR / "segmentation_ensemble.nii.gz",
        OUTPUT_DIR / "segmentation_full.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def prepare_two_sequence_input(t1c_path: Path, flair_path: Optional[Path]) -> torch.Tensor:
    """
    Load T1ce (+ optional FLAIR), apply ROI crop if segmentation is available,
    then resize to 96×96×96.
    """
    # ── ROI crop: tighten resolution on tumour region before resize ──
    seg_path = _find_segmentation()
    crop_bbox = None
    if seg_path is not None:
        crop_bbox = compute_tumour_crop_bbox(seg_path, margin_mm=15.0)
        if crop_bbox is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = crop_bbox
            crop_vol = (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
            print(f"  ROI crop from segmentation: bbox={crop_vol} voxels (+15mm margin)")
        else:
            print("  Segmentation found but no tumour voxels — using full volume")
    else:
        print("  No segmentation found — using full volume (no ROI crop)")

    def _load_and_maybe_crop(path: Path) -> np.ndarray:
        arr = load_nifti(path)
        if crop_bbox is not None:
            arr = crop_to_bbox(arr, crop_bbox)
        return arr

    t1c_arr = _load_and_maybe_crop(t1c_path)
    t1c = resize_to_96(t1c_arr)

    if flair_path is not None and flair_path.exists():
        flair_arr = _load_and_maybe_crop(flair_path)
        flair = resize_to_96(flair_arr)
        using_flair = True
    else:
        flair = t1c.clone()
        using_flair = False

    # [batch, sequences, channel, D, H, W]
    x = torch.stack([t1c, flair], dim=0).unsqueeze(0).unsqueeze(2)

    print(f"T1ce path:  {t1c_path}")
    print(f"FLAIR path: {flair_path if using_flair else 'MISSING -> duplicated T1ce as compatibility fallback'}")
    print(f"Input tensor shape: {tuple(x.shape)}")

    return x


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")


def normalize_state_dict_keys(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def load_model():
    require_file(SRC_DIR, "BrainIAC src directory")
    require_file(BASE_CKPT, "BrainIAC backbone checkpoint")
    require_file(IDH_CKPT, "BrainIAC IDH checkpoint")

    if PLACEHOLDER_CKPT.exists() and PLACEHOLDER_CKPT.stat().st_size < 1024 * 1024:
        print(f"Skipping placeholder checkpoint: {PLACEHOLDER_CKPT} ({PLACEHOLDER_CKPT.stat().st_size} bytes)")

    sys.path.insert(0, str(SRC_DIR.resolve()))
    from model import ViTBackboneNet, Classifier, SingleScanModelBP

    print("Instantiating BrainIAC backbone...")
    backbone = ViTBackboneNet(str(BASE_CKPT))
    classifier = Classifier(d_model=768, num_classes=1)
    model = SingleScanModelBP(backbone, classifier)

    print(f"Loading downstream IDH checkpoint: {IDH_CKPT}")
    ckpt = torch.load(IDH_CKPT, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict_keys(extract_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    print("Missing sample:", missing[:10])
    print("Unexpected sample:", unexpected[:10])

    return model


def run_inference(model, x, device):
    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        out = model(x)

    if isinstance(out, (list, tuple)):
        out = out[0]

    if isinstance(out, dict):
        for key in ["logits", "pred", "output"]:
            if key in out:
                out = out[key]
                break

    out = torch.as_tensor(out).float().detach().cpu()

    if out.numel() == 0:
        raise RuntimeError("Model returned an empty output.")

    logit = out.flatten()[0].item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())

    return logit, prob


def write_report(idh_prob: float, logit: float, t1c_path: Path, flair_path: Optional[Path]):
    report_path = REPORT_PATH

    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("classification", {})
    data["classification"].setdefault("idh", {})
    data["classification"].setdefault("grade", {})

    # ── Bayesian Conditional Fusion ─────────────────────────────────────
    # Stage 8 (radiomics) produces a clinical prior: WHO 2021 rules,
    # T2-FLAIR mismatch, calcification, age-corrected Bayesian prior.
    # Stage 8b (BrainIAC) produces a raw ViT sigmoid probability.
    # We fuse them conditionally — clinical prior dominates at extremes,
    # ViT informs the ambiguous middle.
    clinical_prob = data.get("classification", {}).get("idh_mutation", {}).get("probability")

    if clinical_prob is not None:
        if clinical_prob < 0.15:
            # Strong GBM morphology evidence → IDH-wildtype is essentially certain.
            # ViT cannot override this; keep clinical prior.
            ensemble_prob = clinical_prob
            fusion_method = "clinical_prior_locked"
        elif clinical_prob > 0.70:
            # Strong lower-grade evidence → ViT should confirm, not override.
            # Take the higher confidence signal.
            ensemble_prob = max(clinical_prob, idh_prob)
            fusion_method = "clinical_prior_confirm"
        else:
            # Ambiguous zone → equal-weight Bayesian fusion.
            ensemble_prob = (idh_prob * 0.5) + (clinical_prob * 0.5)
            fusion_method = "weighted_fusion"
    else:
        # No prior from Stage 8 → use ViT alone.
        ensemble_prob = idh_prob
        clinical_prob = idh_prob
        fusion_method = "vit_only"

    vit_raw_prob = idh_prob
    final_prediction = "Mutant" if ensemble_prob > 0.5 else "Wildtype"
    final_confidence = f"{max(ensemble_prob, 1 - ensemble_prob) * 100:.1f}%"

    if ensemble_prob > 0.5:
        if fusion_method == "clinical_prior_locked":
            interpretation = (
                "IDH-wildtype (strongly indicated by GBM morphology on MRI). "
                f"ViT raw: {vit_raw_prob:.2f}, clinical prior: {clinical_prob:.2f}."
            )
        else:
            interpretation = (
                "IDH-mutant suggested by imaging features and ViT analysis. "
                f"ViT raw: {vit_raw_prob:.2f}, clinical prior: {clinical_prob:.2f}."
            )
    else:
        if fusion_method == "clinical_prior_locked":
            interpretation = (
                "IDH-wildtype (strongly indicated by GBM morphology on MRI). "
                f"ViT raw: {vit_raw_prob:.2f}, clinical prior: {clinical_prob:.2f}."
            )
        else:
            interpretation = (
                "IDH-wildtype suggested by imaging features and ViT analysis. "
                f"ViT raw: {vit_raw_prob:.2f}, clinical prior: {clinical_prob:.2f}."
            )

    data["classification"]["idh"]["probability"] = ensemble_prob
    data["classification"]["idh"]["logit"] = logit
    data["classification"]["idh"]["prediction"] = final_prediction
    data["classification"]["idh"]["input_t1ce"] = str(t1c_path)
    data["classification"]["idh"]["input_flair"] = str(flair_path) if flair_path else None

    # Fix 1: Do not overwrite grade_probability if it already exists
    if "grade_probability" not in data or data["grade_probability"] is None:
        data["grade_probability"] = None  # only set if not already computed

    data["idh_probability"] = ensemble_prob
    data["idh_logit"] = logit

    data["classification"]["idh_mutation"] = {
        "most_likely": final_prediction,
        "probability": round(ensemble_prob, 4),
        "confidence": final_confidence,
        "fusion_method": fusion_method,
        "vit_raw_prob": round(vit_raw_prob, 4),
        "clinical_prior_prob": round(clinical_prob, 4) if clinical_prob is not None else None,
        "interpretation": interpretation,
    }

    # Also initialize high_grade if missing
    if "classification" in data and "high_grade" not in data["classification"]:
        data["classification"]["high_grade"] = {"probability": None, "prediction": "Not available from IDH-only model"}

    with open(report_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved report to: {report_path.resolve()}")
    print(f"  Fusion method: {fusion_method} | ensemble_prob={ensemble_prob:.4f} | clinical_prior={clinical_prob}")


def main():
    print(f"Repo dir: {REPO_DIR}")
    print(f"Base ckpt: {BASE_CKPT}")
    print(f"IDH ckpt:  {IDH_CKPT}")

    t1c_candidates = [
        MONAI_DIR / "t1c_resampled.nii.gz",
        MONAI_DIR / "t1ce_resampled.nii.gz",
        PROJECT_ROOT / "data/UPENN-GBM-0001_t1ce.nii.gz",
    ]

    flair_candidates = [
        MONAI_DIR / "flair_resampled.nii.gz",
        PROJECT_ROOT / "data/UPENN-GBM-0001_flair.nii.gz",
    ]

    t1c_path = resolve_input_path(t1c_candidates)
    flair_path = resolve_input_path(flair_candidates)

    if t1c_path is None:
        raise FileNotFoundError("No T1ce input found. Expected one of:\n" + "\n".join(str(p) for p in t1c_candidates))

    # Fix 2: MPS pre-check before run_inference
    device = DEVICE
    if device.type == "mps":
        try:
            # Test Conv3D support to avoid slow fallback
            _t = torch.zeros(1, 1, 4, 4, 4, device=device)
            torch.nn.functional.conv3d(_t, torch.zeros(1, 1, 3, 3, 3, device=device), padding=1)
        except (RuntimeError, NotImplementedError):
            print("  ℹ️  MPS Conv3D not supported — using CPU directly (stable)")
            device = torch.device("cpu")

    print(f"Using device: {device}")

    model = load_model()
    x = prepare_two_sequence_input(t1c_path, flair_path)

    try:
        logit, idh_prob = run_inference(model, x, device)
    except RuntimeError as e:
        error_str = str(e).lower()
        if ("out of memory" in error_str or "not supported on mps" in error_str) and device.type == "mps":
            print(f"MPS error encountered: {e}. Falling back to CPU...")
            device = torch.device("cpu")
            logit, idh_prob = run_inference(model, x, device)
        else:
            raise

    write_report(idh_prob, logit, t1c_path, flair_path)

    print(f"Final logit: {logit:.6f}")
    print(f"IDH probability: {idh_prob:.6f}")


if __name__ == "__main__":
    main()
