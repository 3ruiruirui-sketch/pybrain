import sys
import json
from typing import Optional, Iterable
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

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


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def resize_to_96(arr: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]
    x = x.permute(0, 1, 4, 2, 3)  # [1,1,D,H,W]
    x = F.interpolate(x, size=TARGET_SIZE, mode="trilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)  # [D,H,W]
    return x


def prepare_two_sequence_input(t1c_path: Path, flair_path: Optional[Path]) -> torch.Tensor:
    t1c = resize_to_96(load_nifti(t1c_path))

    if flair_path is not None and flair_path.exists():
        flair = resize_to_96(load_nifti(flair_path))
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

    data["classification"]["idh"]["probability"] = idh_prob
    data["classification"]["idh"]["logit"] = logit
    data["classification"]["idh"]["prediction"] = "Mutant" if idh_prob > 0.5 else "Wildtype"
    data["classification"]["idh"]["input_t1ce"] = str(t1c_path)
    data["classification"]["idh"]["input_flair"] = str(flair_path) if flair_path else None

    data["classification"]["grade"]["probability"] = None
    data["classification"]["grade"]["category"] = "Not computed by BrainIAC IDH model"

    data["idh_probability"] = idh_prob
    data["grade_probability"] = None
    data["idh_logit"] = logit

    data["classification"]["idh_mutation"] = {
        "most_likely": "Mutant" if idh_prob > 0.5 else "Wildtype",
        "confidence": f"{max(idh_prob, 1 - idh_prob) * 100:.1f}%",
        "interpretation": (
            "IDH-mutant suggests better prognosis."
            if idh_prob > 0.5
            else "IDH-wildtype suggests more aggressive behavior."
        ),
    }
    data["classification"]["high_grade"] = {"probability": None, "prediction": "Not available from IDH-only model"}

    with open(report_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved report to: {report_path.resolve()}")


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

    device = choose_device()
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
