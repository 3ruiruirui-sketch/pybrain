#!/usr/bin/env bash
set -euo pipefail

STAGE3_TARGET="${1:-3_brain_tumor_analysis.py}"
if [[ ! -f "$STAGE3_TARGET" ]]; then
  echo "Stage 3 file not found: $STAGE3_TARGET" >&2
  exit 1
fi

python3 - "$STAGE3_TARGET" <<'PY'
from pathlib import Path
import sys
import re
from datetime import datetime

stage3 = Path(sys.argv[1])
stage3_text = stage3.read_text(encoding="utf-8")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

def backup(path: Path):
    b = path.with_name(path.name + f".bak.{ts}")
    b.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[backup] {b}")

def patch_stage3(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text

    if "def _compute_nnunet_target_shape(" not in text:
        anchor = '''def _gpu_cache_clear(device: torch.device) -> None:
    """Flush GPU memory allocator cache for MPS or CUDA."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
'''
        insert = '''def _gpu_cache_clear(device: torch.device) -> None:
    """Flush GPU memory allocator cache for MPS or CUDA."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _compute_nnunet_target_shape(roi_shape: Tuple[int, int, int], nn_cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """Compute a padded ROI shape compatible with DynUNet/nnU-Net strides and patch size."""
    patch_size = tuple(nn_cfg.get("patch_size", roi_shape))
    divisibility = tuple(nn_cfg.get("shape_multiple", (16, 16, 16)))
    target = []
    for dim, patch_dim, mult in zip(roi_shape, patch_size, divisibility):
        base = max(int(dim), int(patch_dim))
        m = max(1, int(mult))
        padded = ((base + m - 1) // m) * m
        target.append(padded)
    return tuple(target)


def _pad_tensor_to_shape(x: Tensor, target_shape: Tuple[int, int, int]) -> Tuple[Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Symmetrically pad (B,C,D,H,W) tensor to target spatial shape."""
    import torch.nn.functional as F

    _, _, d, h, w = x.shape
    td, th, tw = target_shape
    pd = max(0, td - d)
    ph = max(0, th - h)
    pw = max(0, tw - w)
    pad_d = (pd // 2, pd - pd // 2)
    pad_h = (ph // 2, ph - ph // 2)
    pad_w = (pw // 2, pw - pw // 2)
    x_pad = F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]))
    return x_pad, (pad_d, pad_h, pad_w)


def _crop_prob_to_roi(prob: np.ndarray, pads: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], roi_shape: Tuple[int, int, int]) -> np.ndarray:
    """Remove symmetric padding and restore exact ROI geometry."""
    pd, ph, pw = pads
    d0, d1 = pd[0], pd[0] + roi_shape[0]
    h0, h1 = ph[0], ph[0] + roi_shape[1]
    w0, w1 = pw[0], pw[0] + roi_shape[2]
    return np.asarray(prob[:, d0:d1, h0:h1, w0:w1], dtype=np.float32)


def run_nnunet_roi_inference(input_tensor: Tensor, model_device: torch.device, nn_cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Run nnU-Net on ROI while preserving exact ROI geometry for reassembly."""
    logger = get_logger("pybrain")
    roi_shape = tuple(int(x) for x in input_tensor.shape[2:])
    target_shape = _compute_nnunet_target_shape(roi_shape, nn_cfg)
    x_pad, pads = _pad_tensor_to_shape(input_tensor, target_shape)

    local_cfg = dict(nn_cfg)
    local_cfg["roi_mode"] = True
    local_cfg["input_roi_shape"] = roi_shape
    local_cfg["padded_roi_shape"] = target_shape

    logger.info(f"nnU-Net ROI mode: roi_shape={roi_shape}, padded_shape={target_shape}, pads={pads}")
    prob = run_nnunet_inference(x_pad, model_device, local_cfg)
    if prob is None:
        return None

    prob = np.asarray(prob)
    if prob.ndim == 5:
        prob = prob.squeeze(0)
    if prob.ndim != 4:
        raise ValueError(f"nnU-Net returned invalid ndim={prob.ndim}, shape={prob.shape}")
    if prob.shape[0] != 3:
        raise ValueError(f"nnU-Net expected 3 channels [TC,WT,ET], got shape={prob.shape}")
    if tuple(prob.shape[1:]) != target_shape:
        raise ValueError(f"nnU-Net padded output shape mismatch: expected {target_shape}, got {prob.shape[1:]}")

    prob = _crop_prob_to_roi(prob, pads, roi_shape)
    if tuple(prob.shape[1:]) != roi_shape:
        raise ValueError(f"nnU-Net cropped ROI shape mismatch: expected {roi_shape}, got {prob.shape[1:]}")
    return np.ascontiguousarray(prob, dtype=np.float32)
'''
        if anchor not in text:
            raise SystemExit("Could not find _gpu_cache_clear anchor in Stage 3")
        text = text.replace(anchor, insert, 1)

    old_block = '''    # Optional nnU-Net (DynUNet — adds pure U-Net architecture diversity)
    nn_cfg = config.models.get("nnunet", {})
    if nn_cfg.get("enabled", False):
        try:
            logger.info("Running nnU-Net (DynUNet) inference...")
            nn_cfg["bundle_dir"] = config.bundle_dir
            prob = run_nnunet_inference(input_tensor, config.model_device, nn_cfg)
            if prob is not None:
                results["nnunet"] = prob # already squeezed inside run_nnunet_inference
                logger.info(f"nnU-Net output shape: {prob.shape}")
            gc.collect()
            if config.model_device.type == "mps":
                _gpu_cache_clear(config.model_device)
        except Exception as e:
            logger.warning(f"nnU-Net failed: {e}")
'''
    new_block = '''    # Optional nnU-Net (DynUNet — ROI-safe wrapper preserving exact ROI geometry)
    nn_cfg = dict(config.models.get("nnunet", {}))
    if nn_cfg.get("enabled", False):
        try:
            logger.info("Running nnU-Net (DynUNet) inference in ROI-safe mode...")
            nn_cfg["bundle_dir"] = config.bundle_dir
            prob = run_nnunet_roi_inference(input_tensor, config.model_device, nn_cfg)
            if prob is not None:
                results["nnunet"] = prob
                logger.info(f"nnU-Net ROI output shape: {prob.shape}")
            gc.collect()
            if config.model_device.type == "mps":
                _gpu_cache_clear(config.model_device)
        except Exception as e:
            logger.warning(f"nnU-Net failed in ROI-safe mode: {e}")
'''
    if old_block in text:
        text = text.replace(old_block, new_block, 1)
    elif 'run_nnunet_roi_inference(input_tensor, config.model_device, nn_cfg)' not in text:
        raise SystemExit("Could not find nnU-Net block in Stage 3")

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(f"[patched] {path}")
    else:
        print(f"[skip] Stage 3 already patched: {path}")

def find_ensemble_file(stage3_path: Path):
    candidates = []
    for p in Path(".").rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "def run_nnunet_inference(" in txt:
            candidates.append(p)
    if not candidates:
        return None
    preferred = [p for p in candidates if "ensemble" in str(p).lower()]
    if preferred:
        return preferred[0]
    return candidates[0]

def patch_ensemble(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text

    if "def _normalize_nnunet_output(" not in text:
        func_anchor = re.search(r'def run_nnunet_inference\s*\(', text)
        if not func_anchor:
            raise SystemExit("run_nnunet_inference not found in ensemble file")

        helper = '''

def _normalize_nnunet_output(prob: Any) -> np.ndarray:
    """Normalize nnU-Net output to (C,D,H,W) = (3,D,H,W) without collapsing spatial singleton dims."""
    arr = np.asarray(prob)
    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"nnU-Net batch dimension must be 1, got shape={arr.shape}")
        arr = arr.squeeze(0)
    if arr.ndim != 4:
        raise ValueError(f"nnU-Net output must be 4D after batch removal, got shape={arr.shape}")
    if arr.shape[0] == 3:
        return np.ascontiguousarray(arr.astype(np.float32))
    if arr.shape[-1] == 3:
        arr = np.moveaxis(arr, -1, 0)
        return np.ascontiguousarray(arr.astype(np.float32))
    raise ValueError(f"nnU-Net cannot identify class channel in shape={arr.shape}")
'''
        text = text[:func_anchor.start()] + helper + "\n" + text[func_anchor.start():]

    pattern_return = re.compile(
        r'(?P<indent>[ \t]*)return\s+(?P<expr>[A-Za-z_][A-Za-z0-9_\.\[\]]*)\.squeeze\(\)',
        re.MULTILINE
    )
    text = pattern_return.sub(r'\g<indent>return _normalize_nnunet_output(\g<expr>)', text)

    pattern_prob_return = re.compile(
        r'(?P<indent>[ \t]*)return\s+(?P<expr>[A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*$',
        re.MULTILINE
    )

    lines = text.splitlines()
    in_func = False
    func_indent = None
    for i, line in enumerate(lines):
        if re.match(r'^def run_nnunet_inference\s*\(', line):
            in_func = True
            func_indent = len(line) - len(line.lstrip())
            continue
        if in_func:
            curr_indent = len(line) - len(line.lstrip())
            if line.strip().startswith("def ") and curr_indent <= func_indent:
                in_func = False
                func_indent = None
                continue
            if line.strip().startswith("return ") and "_normalize_nnunet_output(" not in line:
                m = re.match(r'^(\s*)return\s+(.+?)\s*$', line)
                if m:
                    expr = m.group(2)
                    if ".squeeze(" not in expr and expr not in ("None",):
                        lines[i] = f'{m.group(1)}return _normalize_nnunet_output({expr})'
                        break

    text = "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(f"[patched] {path}")
    else:
        print(f"[skip] Ensemble already patched: {path}")

patch_stage3(stage3)
ensemble = find_ensemble_file(stage3)
if ensemble is None:
    print("[warn] Could not find ensemble file containing run_nnunet_inference()")
else:
    patch_ensemble(ensemble)
PY
SH
