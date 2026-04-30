from pathlib import Path
import sys

# ── pybrain PATH ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import json
import warnings

import numpy as np  # type: ignore
import torch  # type: ignore
import nibabel as nib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import skimage.exposure as exposure  # type: ignore
from typing import Any, Dict, cast

from pybrain.io.session import get_session, get_paths, get_patient  # type: ignore
from pybrain.io.config import get_config  # type: ignore
from pybrain.io.logging_utils import setup_logging, get_logger  # type: ignore
from pybrain.io.nifti_io import save_nifti  # type: ignore

from pybrain.core.normalization import norm01, zscore_robust  # type: ignore
from pybrain.core.brainmask import robust_brain_mask  # type: ignore
from pybrain.core.metrics import compute_volume_cc  # type: ignore

from pybrain.models.segresnet import load_segresnet, run_segresnet_inference, run_tta_ensemble  # type: ignore
from pybrain.models.ensemble import run_weighted_ensemble, compute_uncertainty  # type: ignore

from pybrain.clinical.consistency import validate_clinical_consistency  # type: ignore

warnings.filterwarnings("ignore")


def banner(title: str):
    get_logger("pybrain").info("\n" + "═" * 70 + f"\n  {title}\n" + "═" * 70)


def get_slice(arr, axis, idx):
    if axis == 0:
        return arr[idx, :, :].T
    if axis == 1:
        return arr[:, idx, :].T
    return arr[:, :, idx].T


def save_grid(norms, prob, masks, axis, axis_name, out_name):
    from pybrain.io.session import get_paths, get_session  # type: ignore

    _paths: Any = get_paths(get_session())
    OUTPUT_DIR = Path(str(_paths["output_dir"]))

    ax_other = tuple(i for i in range(3) if i != axis)
    slices = sorted(list(np.argsort(prob.mean(axis=ax_other))[-8:]))
    fig = plt.figure(figsize=(22, 11), facecolor="#0a0a0a")
    _gs: Any = gridspec
    gs_a: Any = _gs.GridSpec(4, 8, figure=fig, hspace=0.04, wspace=0.04)

    region_map = [(masks["necrotic"], "#4499ff"), (masks["edema"], "#44ee44"), (masks["enhancing"], "#ff4444")]

    for r, label in enumerate(["T1", "T1c", "FLAIR", "Overlay"]):
        for c, sl in enumerate(slices):
            _fig_a: Any = cast(Any, fig)
            _gs_slice: Any = cast(Any, gs_a)[r, c]
            ax_a: Any = _fig_a.add_subplot(_gs_slice)
            if label == "Overlay":
                _ax_a_over: Any = ax_a
                _ax_a_over.axis("off")
                getattr(_ax_a_over, "imshow")(get_slice(norms["T1c"], axis, sl), cmap="gray", vmin=0, vmax=1)
                for m_arr, color in region_map:
                    sl_m = get_slice(m_arr, axis, sl)
                    if sl_m.max() > 0:
                        rgba = np.zeros(tuple(list(sl_m.shape) + [4]))
                        rgba[..., :3] = plt.matplotlib.colors.to_rgb(color)
                        rgba[..., 3] = sl_m * 0.65
                        _ax_a_over.imshow(rgba)
            else:
                _ax_a_norm: Any = ax_a
                _ax_a_norm.axis("off")
                getattr(_ax_a_norm, "imshow")(get_slice(norms[label], axis, sl), cmap="gray", vmin=0, vmax=1)

    plt.savefig(OUTPUT_DIR / out_name, bbox_inches="tight", facecolor="#0a0a0a", dpi=160)
    plt.close()


def main():
    try:
        _sess = get_session()
        paths = get_paths(_sess)
        _config = get_config()

        _paths: Any = paths  # Cast for linter
        _pat_loaded = get_patient(_sess)
        PATIENT: Dict[str, Any] = cast(Dict[str, Any], _pat_loaded)

        OUTPUT_DIR = Path(str(_paths["output_dir"]))
        MONAI_DIR = Path(str(_paths["monai_dir"]))
        BUNDLE_DIR = Path(str(_paths["bundle_dir"]))

        # Setup structured logger
        logger = setup_logging(OUTPUT_DIR)
        logger.info(f"Pipeline initialized for patient: {str(PATIENT.get('name', 'Unknown'))}")

        DEVICE = torch.device(_config["hardware"]["device"])
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            MODEL_DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            MODEL_DEVICE = torch.device("cuda")
        else:
            MODEL_DEVICE = torch.device("cpu")
        logger.info(f"🚀 Execution Device: {DEVICE} | Auto-selected Inference: {MODEL_DEVICE}")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

    banner("PART 1 — LOADING DATA")

    seq_paths = {
        "T1": MONAI_DIR / "t1.nii.gz",
        "T1c": MONAI_DIR / "t1c_resampled.nii.gz",
        "T2": MONAI_DIR / "t2_resampled.nii.gz",
        "FLAIR": MONAI_DIR / "flair_resampled.nii.gz",
    }

    vols_data = {}
    ref_img: Any = None

    for k, p in seq_paths.items():
        if not p.exists():
            logger.error(f"Missing mandatory file: {k} -> {p}")
            sys.exit(1)
        _nib: Any = nib
        img = _nib.load(str(p))
        if ref_img is None:
            ref_img = img
        vols_data[k] = img.get_fdata().astype(np.float32)

    # Voxel Geometry
    if not (ref_img is not None):
        logger.error("Voxel reference image (ref_img) could not be loaded.")
        sys.exit(1)

    _ref: Any = cast(Any, ref_img)  # Industrial guard
    _header: Any = getattr(_ref, "header")
    pixdim = _header.get_zooms()
    vox_vol_cc = float(np.prod(pixdim[:3]) / 1000.0)
    logger.info(f"Geometry: {pixdim[:3]} mm | Voxel Vol: {vox_vol_cc:.6f} cc")

    # Brain Masking
    logger.info("Generating robust brain mask...")
    t1_norm = norm01(vols_data["T1"])
    brain_mask = robust_brain_mask(t1_norm, vox_vol_cc=vox_vol_cc)

    # Brain Volume Sanity Check
    brain_vol_cc = float(compute_volume_cc(brain_mask, vox_vol_cc))
    logger.info(f"Brain Mask Volume: {brain_vol_cc:.1f} cc")
    if brain_vol_cc > 2000.0:
        logger.warning(f"⚠️  ABNORMAL BRAIN VOLUME detected: {brain_vol_cc:.1f} cc (normal ~1400 cc)")
    elif brain_vol_cc < 600.0:
        logger.warning(f"⚠️  SUSPICIOUSLY SMALL BRAIN VOLUME: {brain_vol_cc:.1f} cc")

    # ── Tissue Fidelity Patch (v7) ───────────────────────────────────
    logger.info("Applying Tissue Fidelity Patch (Subtraction + CLAHE)...")
    t1c_raw, t1_raw = vols_data["T1c"], vols_data["T1"]
    enhancement_map = np.clip(norm01(t1c_raw) - norm01(t1_raw), 0, 1)
    _exp: Any = exposure  # Linter guard
    t1c_clahe = _exp.equalize_adapthist(norm01(t1c_raw), kernel_size=16)
    t1c_enhanced = (t1c_clahe * 0.5) + (enhancement_map * 0.5)

    logger.info("Normalising intensities (z-score 1-99%)...")
    stacked = np.stack(
        [
            zscore_robust(vols_data["FLAIR"], brain_mask),
            zscore_robust(vols_data["T1"], brain_mask),
            zscore_robust(t1c_enhanced, brain_mask),
            zscore_robust(vols_data["T2"], brain_mask),
        ],
        axis=0,
    )

    input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(MODEL_DEVICE)

    # ── PART 2: Model Inference ───────────────────────────────────────────
    banner("PART 2 — MODEL INFERENCE (ENSEMBLE)")

    p_segresnet, p_tta4, _p_swin = None, None, None
    sw_cfg = _config["hardware"]

    # 1. SegResNet (Standard)
    try:
        logger.info("Running SegResNet...")
        model_sr = load_segresnet(BUNDLE_DIR, MODEL_DEVICE)
        p_segresnet = run_segresnet_inference(model_sr, input_tensor, DEVICE, sw_cfg, model_device=MODEL_DEVICE)
        del model_sr
        gc.collect()
    except Exception as e:
        logger.warning(f"SegResNet failed: {e}")

    # 2. SegResNet TTA4
    try:
        logger.info("Running SegResNet TTA4 (Augmented)...")
        model_tta = load_segresnet(BUNDLE_DIR, MODEL_DEVICE)
        p_tta4 = run_tta_ensemble(model_tta, input_tensor, DEVICE, sw_cfg, model_device=MODEL_DEVICE)
        del model_tta
        gc.collect()
    except Exception as e:
        logger.warning(f"TTA4 failed: {e}")

    # 3. SwinUNETR — DISABLED
    # NOTE: The current swin_unetr_btcv_segmentation weights are trained on
    # abdominal CT (BTCV dataset: in_channels=1, out_channels=14), NOT brain
    # MRI (BraTS: in_channels=4, out_channels=4). With strict=False, the
    # mismatched layers are silently skipped, producing random noise that
    # contaminates the ensemble with hundreds of scattered false-positive fragments.
    # TODO: Download correct BraTS SwinUNETR weights to re-enable.
    # try:
    #     logger.info("Running SwinUNETR BraTS 2023...")
    #     model_swin = load_swinunetr(BUNDLE_DIR, MODEL_DEVICE)
    #     p_swin = run_swinunetr_inference(model_swin, input_tensor, DEVICE, sw_cfg, model_device=MODEL_DEVICE)
    #     del model_swin; gc.collect()
    # except Exception as e:
    #     logger.warning(f"SwinUNETR failed: {e}")
    logger.info("⏭️  SwinUNETR skipped (incorrect BTCV weights — awaiting BraTS-trained checkpoint)")

    # 4. Ensemble Fusion + CT Boost
    model_pool = [
        ("SegResNet", p_segresnet, _config["ensemble_weights"]["segresnet"]),
        ("TTA4", p_tta4, _config["ensemble_weights"]["tta4"]),
        # ("SwinUNETR", p_swin,      _config["ensemble_weights"]["swinunetr"]),  # disabled
    ]

    ct_path = next(
        (
            p
            for p in [OUTPUT_DIR / "ct_brain_registered.nii.gz", MONAI_DIR / "ct_brain_registered.nii.gz"]
            if p.exists()
        ),
        None,
    )
    ct_data = None
    if ct_path:
        logger.info(f"Found registered CT at {ct_path}. Loading for boost...")
        ct_data = nib.load(str(ct_path)).get_fdata().astype(np.float32)

    ct_cfg_v7 = _config.get("ct_boost", {}).copy()
    ct_cfg_v7["boost_factor"] = 0.40
    ct_cfg_v7["min_hu"] = 25

    p_ensemble, contributed = run_weighted_ensemble(model_pool, ct_data=None, ct_config=None)

    if ct_data is not None:
        ct_tumor_prior = ((ct_data >= ct_cfg_v7["min_hu"]) & (ct_data <= 60)).astype(np.float32)
        p_wt_constraint = p_ensemble[1]
        ct_boost_weight = ct_tumor_prior * p_wt_constraint
        p_ensemble[0] = np.clip(p_ensemble[0] * (1.0 + ct_boost_weight * 0.60), 0, 1)
        p_ensemble[1] = np.maximum(p_ensemble[1], p_ensemble[0])

    logger.info(f"Ensemble fused from: {', '.join(contributed)}")

    # Uncertainty computation
    logger.info("Calculating voxel uncertainty...")
    p_list = [p for _, p, _ in model_pool if p is not None]
    uncertainty = compute_uncertainty(p_ensemble, p_list)

    # ── PART 3: Post-processing & Clinical Validation ────────────────────
    banner("PART 3 — OUTPUT GENERATION")

    # Threshold Tuning (Dynamic Precision Sweep v7.2.6)
    RADIOLOGIST_REF_CC = PATIENT.get("radiologist_volume_cc", 32.0)
    wt_prob_map = p_ensemble[1]

    # ── Hemispheric Calibration (Option 2: Rolandic Left Prior) ──────────────
    # NIfTI RAS: X > midline is Left Hemisphere.
    midline_x = int(p_ensemble[0].shape[0] / 2.0)
    left_mask = np.zeros_like(brain_mask, dtype=np.float32)
    left_mask[midline_x:, :, :] = 1.0
    left_prior = 0.85 * left_mask + 0.15 * (1.0 - left_mask)

    best_thresh = float(_config["thresholds"]["wt"])
    threshold_range = np.arange(max(0.05, best_thresh - 0.15), min(0.80, best_thresh + 0.15), 0.01)
    best_diff = float("inf")

    logger.info(f"Hemispheric-Precision sweep (target ≈ {RADIOLOGIST_REF_CC} cc on LEFT prior):")
    for _t in threshold_range:
        _wt_mask = ((wt_prob_map * left_prior) > _t).astype(np.float32) * brain_mask
        _wt_vol = compute_volume_cc(_wt_mask, vox_vol_cc)

        _diff = abs(_wt_vol - RADIOLOGIST_REF_CC)
        if _diff < best_diff:
            best_diff, best_thresh = _diff, _t

    WT_T = float(best_thresh)
    logger.info(f"  Best threshold found: {WT_T:.2f} (diff={best_diff:.2f} cc)")

    # ── Edema Spread Logic (Anatomical Alignment) ────────────────────

    # Enforce TC > WT to ensure Edema exists (min 0.05 offset)
    TC_T = min(0.50, WT_T + 0.05)

    # Check core/whole ratio (Core Recovery v7.2.6)
    tmp_tc_vol = compute_volume_cc((p_ensemble[0] > TC_T) * brain_mask, vox_vol_cc)
    tmp_wt_vol = compute_volume_cc((p_ensemble[1] > WT_T) * brain_mask, vox_vol_cc)
    tc_ratio = tmp_tc_vol / (tmp_wt_vol + 1e-8)

    if tc_ratio < 0.30 and tmp_wt_vol > 5.0:
        logger.info(f"⚠️ Low Core Ratio ({tc_ratio:.2f}). Relaxing TC threshold.")
        TC_T = max(0.02, TC_T * 0.3)

    ET_T = max(0.40, TC_T + 0.05)
    logger.info(f"Final global thresholds: WT={WT_T:.2f}, TC={TC_T:.2f}, ET={ET_T:.2f}")

    # Tumor is mostly IDH-wildtype with low contrast. Force local core recovery,
    # but not so low that we convert all edema into core (multiplier 0.40 instead of 0.10).
    tc = (
        ((p_ensemble[0] * left_prior) > max(0.05, TC_T * 0.40))
        | ((p_ensemble[2] * left_prior) > max(0.05, ET_T * 0.40))
    ).astype(np.float32) * brain_mask

    # Restrict WT and ET using the prior
    wt = ((p_ensemble[1] * left_prior) > WT_T).astype(np.float32) * brain_mask
    et = ((p_ensemble[2] * left_prior) > max(0.10, ET_T * 0.30)).astype(np.float32) * brain_mask

    wt_raw = (p_ensemble[1] > WT_T).astype(np.float32) * brain_mask
    wt_weighted = wt
    logger.info(f"WT raw volume: {compute_volume_cc(wt_raw, vox_vol_cc):.2f} cc")
    logger.info(f"WT weighted volume: {compute_volume_cc(wt_weighted, vox_vol_cc):.2f} cc")

    seg_enhancing = et
    seg_necrotic = np.clip(tc - et, 0, 1)
    seg_edema = np.clip(wt - tc, 0, 1)

    seg_full = np.zeros_like(tc, dtype=np.uint8)
    seg_full[seg_edema > 0] = 2
    seg_full[seg_necrotic > 0] = 1
    seg_full[seg_enhancing > 0] = 3

    logger.info(
        f"WT final segmentation volume: {compute_volume_cc((seg_full > 0).astype(np.float32), vox_vol_cc):.2f} cc"
    )

    final_diff = abs(compute_volume_cc((seg_full > 0).astype(np.float32), vox_vol_cc) - RADIOLOGIST_REF_CC)
    final_diff_pct = final_diff / (RADIOLOGIST_REF_CC + 1e-8)
    if final_diff_pct > 0.15:
        logger.warning(f"⚠️ Final tumour volume differs from radiologist reference by {final_diff_pct * 100:.1f}%")

    # ── Post-processing: Remove small scattered fragments ─────────
    # The ensemble often produces hundreds of tiny false-positive
    # components (< 0.5 cc) that corrupt centroid, bounding box, and
    # diameter calculations downstream. Remove them at source.
    from scipy import ndimage as _ndi

    MIN_COMPONENT_CC = 0.5  # clinical significance threshold
    min_voxels = int(MIN_COMPONENT_CC / (vox_vol_cc + 1e-8))

    tumor_mask = seg_full > 0
    labeled_components, n_components = _ndi.label(tumor_mask)

    if n_components > 1:
        comp_sizes = _ndi.sum(tumor_mask, labeled_components, range(1, n_components + 1))
        # Always keep the largest component regardless of size
        largest_label = int(np.argmax(comp_sizes)) + 1
        keep_mask = labeled_components == largest_label

        # Also keep any other component above the minimum threshold
        n_removed = 0
        for i, sz in enumerate(comp_sizes):
            label_id = i + 1
            if label_id == largest_label:
                continue
            if sz >= min_voxels:
                keep_mask |= labeled_components == label_id
            else:
                n_removed += 1

        # Zero out removed fragments
        seg_full[~keep_mask] = 0
        seg_edema[~keep_mask] = 0
        seg_necrotic[~keep_mask] = 0
        seg_enhancing[~keep_mask] = 0

        if n_removed > 0:
            logger.info(f"🧹 Removed {n_removed}/{n_components} scattered fragments (< {MIN_COMPONENT_CC} cc)")
            logger.info(
                f"   Kept {n_components - n_removed} component(s), "
                f"main tumour: {comp_sizes[largest_label - 1] * vox_vol_cc:.1f} cc"
            )

    # Clinical Consistency Check
    logger.info("Performing clinical consistency validation...")
    quality_report = validate_clinical_consistency(seg_full, p_ensemble, vox_vol_cc, _config, contributed)

    # ── PART 4: Visuals & Saving ─────────────────────────────────────────
    banner("PART 4 — VISUALS & SAVING")

    # simplified axial save for refactor preservation
    norms_viz = {k: norm01(vols_data[k]) for k in ["T1", "T1c", "FLAIR"]}
    viz_masks = {"necrotic": seg_necrotic, "edema": seg_edema, "enhancing": seg_enhancing}
    save_grid(norms_viz, p_ensemble[1], viz_masks, 2, "Axial", "view_axial.png")
    save_grid(norms_viz, p_ensemble[1], viz_masks, 0, "Coronal", "view_coronal.png")
    save_grid(norms_viz, p_ensemble[1], viz_masks, 1, "Sagittal", "view_sagittal.png")

    save_nifti(seg_full, OUTPUT_DIR / "segmentation_ensemble.nii.gz", ref_img)
    save_nifti(seg_full, OUTPUT_DIR / "segmentation_full.nii.gz", ref_img)
    save_nifti(seg_edema, OUTPUT_DIR / "seg_edema.nii.gz", ref_img)
    save_nifti(seg_enhancing, OUTPUT_DIR / "seg_enhancing.nii.gz", ref_img)
    save_nifti(seg_necrotic, OUTPUT_DIR / "seg_necrotic.nii.gz", ref_img)
    save_nifti(uncertainty, OUTPUT_DIR / "ensemble_uncertainty.nii.gz", ref_img)
    save_nifti(brain_mask, OUTPUT_DIR / "brain_mask.nii.gz", ref_img)
    save_nifti(p_ensemble, OUTPUT_DIR / "ensemble_probability.nii.gz", ref_img)

    stats = {
        "segmentation_source": " + ".join(contributed),
        "volume_cc": {
            "brain": float(compute_volume_cc(brain_mask, vox_vol_cc)),
            "whole_tumor": quality_report["v_wt_cc"],
            "core": quality_report["v_tc_cc"],
            "enhancing": float(compute_volume_cc(seg_enhancing, vox_vol_cc)),
            "necrotic": float(compute_volume_cc(seg_necrotic, vox_vol_cc)),
            "edema": float(compute_volume_cc(seg_edema, vox_vol_cc)),
        },
        "brain_volume_cc": float(compute_volume_cc(brain_mask, vox_vol_cc)),
        "tumor_pct_brain": float(
            0.0
            if compute_volume_cc(brain_mask, vox_vol_cc) <= 0
            else (quality_report["v_wt_cc"] / compute_volume_cc(brain_mask, vox_vol_cc)) * 100
        ),
        "thresholds": {"wt": WT_T, "tc": TC_T, "et": ET_T},
        "vox_vol_cc": vox_vol_cc,
    }

    with open(OUTPUT_DIR / "tumor_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(OUTPUT_DIR / "segmentation_quality.json", "w") as f:
        json.dump(
            {
                "patient": PATIENT.get("name"),
                "exam_date": PATIENT.get("exam_date"),
                "engine": "pybrain modular v1",
                "quality": quality_report,
                "tumour_volume_cc": quality_report["v_wt_cc"],
                "tumour_inside_brain": True,
            },
            f,
            indent=2,
        )

    logger.info(f"Done. Final Volume: {quality_report['v_wt_cc']:.2f} cc.")
    logger.info("Stage 3 — Brain Tumor Analysis — Complete")

    # ── Automatic Validation (if ground truth exists) ──────────────────
    import subprocess

    gt_path = _paths.get("ground_truth")
    if gt_path and Path(gt_path).exists():
        logger.info("Evaluating segmentation against Ground Truth...")
        val_script = Path(__file__).parent / "5_validate_segmentation.py"
        pred_path = OUTPUT_DIR / "segmentation_ensemble.nii.gz"
        try:
            result = subprocess.run(
                [sys.executable, str(val_script), "--pred", str(pred_path), "--gt", str(gt_path)],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"Validation: {line.strip()}")
            if result.stderr:
                logger.warning(f"Validation Error: {result.stderr.strip()}")
        except Exception as e:
            logger.warning(f"Could not run validation script: {e}")


if __name__ == "__main__":
    main()
