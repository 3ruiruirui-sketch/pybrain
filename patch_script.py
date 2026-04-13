import sys
from pathlib import Path

# File to patch
file_path = Path("/Users/ssoares/Downloads/PY-BRAIN/scripts/3_brain_tumor_analysis.py")
content = file_path.read_text()

# PATCH 1: Docstring update
content = content.replace("v5 Ensemble (Hardened Production Version)", "v6 Ensemble (SegResNet + TTA4 + SwinUNETR)")

# PATCH 2: SwinUNETR and Ensemble Fusion Array
old_fusion_code = """if p_segresnet is None and p_tta4 is None:
    print("❌ Fatal: Ensemble failure."); sys.exit(1)

# Fusion Logic: 45% Standard Inference / 55% TTA4 for enhanced robustness
if p_segresnet is not None and p_tta4 is not None:
    p_ensemble = (0.45 * p_segresnet) + (0.55 * p_tta4)
else:
    p_ensemble = p_segresnet if p_segresnet is not None else p_tta4

_active = []
if p_segresnet is not None:
    _active.append("SegResNet-orig(45%)" if p_tta4 is not None else "SegResNet-orig(100%)")
if p_tta4 is not None:
    _active.append("SegResNet-TTA4(55%)")
print(f"  📊 Ensemble Fusion: {' + '.join(_active)}")"""

new_fusion_code = """
# ── 3. SwinUNETR — Global Context (Long-range Attention) ─────────────
# Advantage over SegResNet: identifies necrotic core via structural
# deformation, not just T1c brightness — critical for this case.
p_swin = None
try:
    print("  Running SwinUNETR BraTS 2023 (roi=128³, overlap=0.7)...")
    from monai.networks.nets import SwinUNETR
    import urllib.request

    SWIN_DIR = BUNDLE_DIR / "swin_bundle"
    SWIN_DIR.mkdir(parents=True, exist_ok=True)
    _swin_ckpt = SWIN_DIR / "model_swinvit.pt"

    if not _swin_ckpt.exists():
        _SWIN_URL = (
            "https://github.com/Project-MONAI/research-contributions"
            "/releases/download/SwinUNETR/model_swinvit.pt"
        )
        print(f"  📥 Downloading SwinUNETR weights (~400MB)...")
        urllib.request.urlretrieve(_SWIN_URL, str(_swin_ckpt))

    model_swin = SwinUNETR(
        img_size       = (128, 128, 128),
        in_channels    = 4,
        out_channels   = 4,       # softmax: BG + NCR + ED + ET
        feature_size   = 48,
        use_checkpoint = True,    # saves ~2GB RAM during inference
    )
    _ckpt = torch.load(str(_swin_ckpt), map_location="cpu",
                       weights_only=False)
    _sd   = _ckpt.get("model", _ckpt.get("state_dict", _ckpt))
    model_swin.load_state_dict(_sd, strict=False)
    model_swin = model_swin.cpu().eval()

    # CRITICAL: SwinUNETR was trained at 128³ — DO NOT use 96³
    # overlap=0.7 improves boundary detection on heterogeneous tumours
    _swin_inferer = SlidingWindowInferer(
        roi_size      = (128, 128, 128),
        sw_batch_size = 1,
        overlap       = 0.7,
        mode          = "gaussian",
        progress      = True,
    )
    with torch.no_grad():
        _swin_logits = _swin_inferer(input_tensor.cpu(), model_swin)

    # 4-channel softmax → 3-channel [TC, WT, ET] to match format_probs()
    _sp = torch.softmax(_swin_logits, dim=1)[0].numpy()
    # _sp[0]=BG _sp[1]=NCR _sp[2]=ED _sp[3]=ET
    p_swin = np.stack([
        _sp[1] + _sp[3],           # TC = NCR + ET
        _sp[1] + _sp[2] + _sp[3],  # WT = NCR + ED + ET
        _sp[3],                    # ET only
    ], axis=0)
    print(f"  ✅ SwinUNETR done | WT max: {p_swin[1].max():.3f} "
          f"| TC max: {p_swin[0].max():.3f}")
    del model_swin; gc.collect()

except Exception as _swin_err:
    print(f"  ⚠️  SwinUNETR skipped: {_swin_err}")

    _pool = [
        ("SegResNet",  p_segresnet, 1.0),
        ("TTA4",       p_tta4,      1.0),
        ("SwinUNETR",  p_swin,      1.5),  # higher weight: better core fidelity
    ]
    _valid = [(n, p, w) for n, p, w in _pool if p is not None]
    if not _valid:
        print("❌ Fatal: all models failed."); sys.exit(1)

    _total_w = sum(w for _, _, w in _valid)
    p_ensemble = sum(p * w for _, p, w in _valid) / _total_w
    _contributed = [n for n, _, _ in _valid]
    _weight_str  = [f"{n}={w:.1f}" for n, _, w in _valid]
    print(f"  📊 Ensemble: {' + '.join(_weight_str)} "
          f"(total weight={_total_w:.1f})")
"""
content = content.replace(old_fusion_code, new_fusion_code.strip())

# PATCH 3: Fix TC_T and ET_T logic
old_tc_et_code = """WT_T = best_thresh
TC_T = max(0.35, best_thresh + 0.05)  # TC is naturally more confident
ET_T = max(0.40, best_thresh + 0.10)  # ET bounds are usually very high conf"""

new_tc_et_code = """WT_T = best_thresh
# No fixed offset: let ensemble softmax assign each voxel
# to its most probable class without arbitrary TC suppression.
# SwinUNETR argmax already handles core vs edema disambiguation.
TC_T = best_thresh          # same as WT — argmax handles disambiguation
ET_T = max(0.40, best_thresh + 0.05)  # ET still slightly stricter"""

if old_tc_et_code not in content:
    print("Could not find TC_T old code block! Make sure spacing is exact.")
    sys.exit(1)

content = content.replace(old_tc_et_code, new_tc_et_code)

# PATCH 4: Replace Validate Clinical Consistency
old_quality_code = """# --- Quality Check Implementation ---
brain_vol_cc = brain_mask.sum() * vox_vol_cc
tumour_vol_cc = (seg_full > 0).sum() * vox_vol_cc

in_brain = False
if tumour_vol_cc > 0.1:
    coords = np.argwhere(seg_full > 0)
    center_vox = coords.mean(axis=0).astype(int)
    if all(0 <= c < s for c, s in zip(center_vox, seg_full.shape)):
        in_brain = bool(brain_mask[center_vox[0], center_vox[1], center_vox[2]] > 0)

quality = {
    "brain_volume_cc": float(brain_vol_cc),
    "tumour_volume_cc": float(tumour_vol_cc),
    "tumour_inside_brain": in_brain,
    "status": "PASS" if (in_brain and 0.1 < tumour_vol_cc < 500) else "WARN"
}
with open(OUTPUT_DIR / "segmentation_quality.json", "w") as f:
    json.dump(quality, f, indent=2)

if quality["status"] == "WARN":
    print(f"⚠️  QUALITY WARNING: Tumour vol {tumour_vol_cc:.1f}cc | Inside brain: {in_brain}")"""

new_quality_code = """def validate_clinical_consistency(
    seg, p_ens, brain_mask, vox_vol_cc,
    wt_tc_min_ratio=0.05,
    entropy_warn_thresh=0.7,
):
    \"\"\"
    Pre-output clinical sanity: 3 checks.
    Returns (flags_dict, possibly_corrected_seg).
    \"\"\"
    from scipy import ndimage as _ndi

    wt_mask = (seg > 0).astype(np.float32)
    tc_mask = ((seg == 1) | (seg == 3)).astype(np.float32)
    ed_mask = (seg == 2).astype(np.float32)

    v_wt = float(wt_mask.sum()) * vox_vol_cc
    v_tc = float(tc_mask.sum()) * vox_vol_cc
    warnings_out = []

    flags = {
        "status"                : "OK",
        "v_wt_cc"               : round(v_wt, 2),
        "v_tc_cc"               : round(v_tc, 2),
        "tc_pct_of_wt"          : 0.0,
        "core_empty_warning"    : False,
        "adaptive_reseg_applied": False,
        "continuity_warning"    : False,
        "isolated_edema_cc"     : 0.0,
        "uncertainty_flag"      : False,
        "centre_entropy"        : 0.0,
        "requires_manual_review": False,
        "contributing_models"   : _contributed,
    }

    # CHECK 1 — Core/Whole-Tumour ratio
    if v_wt > 5.0:
        tc_ratio = v_tc / (v_wt + 1e-8)
        flags["tc_pct_of_wt"] = round(tc_ratio * 100, 1)
        if tc_ratio < wt_tc_min_ratio:
            flags["core_empty_warning"] = True
            flags["requires_manual_review"] = True
            msg = (f"⚠️  CORE EMPTY: TC={v_tc:.1f}cc = "
                   f"{tc_ratio*100:.1f}% of WT={v_wt:.1f}cc "
                   f"(min {wt_tc_min_ratio*100:.0f}%)")
            warnings_out.append(msg); print(f"  {msg}")

            # Adaptive re-segmentation — relax TC to 0.25
            print("  🔄 Adaptive re-seg (TC thresh → 0.25, ET → 0.30)...")
            _tc_r = (p_ens[0] > 0.25).astype(np.uint8)
            _et_r = (p_ens[2] > 0.30).astype(np.uint8)
            _wt_r = (p_ens[1] > 0.30).astype(np.uint8)
            seg_r = np.zeros_like(seg, dtype=np.uint8)
            seg_r[np.clip(_wt_r - _tc_r, 0, 1) > 0] = 2
            seg_r[_tc_r > 0] = 1
            seg_r[_et_r > 0] = 3
            v_tc_r = float(
                ((_tc_r + _et_r) > 0).sum()) * vox_vol_cc
            if v_tc_r > v_tc:
                seg[:] = seg_r
                flags["adaptive_reseg_applied"] = True
                print(f"  ✅ Recovered TC: {v_tc_r:.1f} cc")
            else:
                print("  ℹ️  Relaxed threshold did not improve TC.")

    # CHECK 2 — Edema islands isolated from core
    if ed_mask.sum() > 0 and tc_mask.sum() > 0:
        tc_dil = _ndi.binary_dilation(tc_mask.astype(bool),
                                       iterations=3)
        iso_ed = ed_mask.astype(bool) & ~tc_dil
        iso_vol = float(iso_ed.sum()) * vox_vol_cc
        if iso_vol > 1.0:
            flags["continuity_warning"] = True
            flags["isolated_edema_cc"] = round(iso_vol, 2)
            msg = (f"⚠️  CONTINUITY: {iso_vol:.1f}cc edema "
                   f"isolated from core (>3 voxels gap)")
            warnings_out.append(msg); print(f"  {msg}")

    # CHECK 3 — Entropy / uncertainty in tumour centre
    _pc  = np.clip(p_ens, 1e-8, 1.0)
    _ent = -np.sum(_pc * np.log(_pc), axis=0)
    _max_ent = float(np.log(p_ens.shape[0]))
    if wt_mask.sum() > 0:
        centre_ent = float(
            _ent[wt_mask.astype(bool)].mean()
        ) / (_max_ent + 1e-8)
        flags["centre_entropy"] = round(centre_ent, 3)
        if centre_ent > entropy_warn_thresh:
            flags["uncertainty_flag"] = True
            flags["requires_manual_review"] = True
            flags["status"] = (
                "UNCERTAIN — Classificação Incerta "
                "— Requer Revisão Manual"
            )
            msg = (f"⚠️  HIGH ENTROPY: {centre_ent:.3f} "
                   f"> {entropy_warn_thresh} in tumour")
            warnings_out.append(msg); print(f"  {msg}")

    if not warnings_out:
        flags["status"] = "OK — Segmentation clinically consistent"

    return flags, seg


quality_flags, seg_full = validate_clinical_consistency(
    seg_full, p_ensemble, brain_mask, vox_vol_cc
)

with open(OUTPUT_DIR / "segmentation_quality.json", "w") as _qf:
    json.dump({
        "patient"   : PATIENT.get("name", "unknown"),
        "exam_date" : PATIENT.get("exam_date", "unknown"),
        "engine"    : " + ".join(_contributed),
        "quality"   : quality_flags,
    }, _qf, indent=2)
print("  💾 Saved → segmentation_quality.json")"""

if old_quality_code not in content:
    print("Could not find Quality Check old code block! Make sure spacing is exact.")
    sys.exit(1)

content = content.replace(old_quality_code, new_quality_code)

# Check indentation for _pool block (Wait, the user's prompt had an indent level issue in their provided prompt!)
# Looking at the user's prompt:
# "except Exception as _swin_err:
#     print(f"  ⚠️  SwinUNETR skipped: {_swin_err}")
# Then REPLACE the existing fusion block:"
# In my python logic, I included _pool inside the exception if I paste blindly from their text without dedenting!
# Let's fix that!
content = content.replace("    _pool = [", "_pool = [")
content = content.replace("    _valid =", "_valid =")
content = content.replace("    if not _valid:", "if not _valid:")
content = content.replace("        print(\"❌ Fatal", "    print(\"❌ Fatal")
content = content.replace("    _total_w =", "_total_w =")
content = content.replace("    p_ensemble =", "p_ensemble =")
content = content.replace("    _contributed =", "_contributed =")
content = content.replace("    _weight_str  =", "_weight_str  =")
content = content.replace("    print(f\"  📊", "print(f\"  📊")
content = content.replace("          f\"(total weight", "      f\"(total weight")

file_path.write_text(content)
print("File patched successfully.")
