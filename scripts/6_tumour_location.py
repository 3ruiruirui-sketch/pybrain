#!/usr/bin/env python3
"""
Tumour Location and Ramification Analysis
==========================================
Computes tumour location, lobe involvement, hemisphere,
proximity to key structures, and 3D bounding box
from the segmentation mask — no atlas required.

Can be run standalone or imported by generate_report.py.

Requirements:
  pip install nibabel numpy scipy

Run:
  python3 tumour_location.py
"""

import sys
import json
from pathlib import Path

# ── pybrain Imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from pybrain.io.session import get_session, get_paths, get_patient
    from pybrain.io.config import get_config
    from pybrain.io.logging_utils import setup_logging

    _sess = get_session()
    _paths = get_paths(_sess)
    _config = get_config()
    PATIENT = get_patient(_sess)

    OUTPUT_DIR = _paths["output_dir"]
    MONAI_DIR = _paths["monai_dir"]
    RESULTS_DIR = _paths["results_dir"]

    logger = setup_logging(OUTPUT_DIR)
    logger.info("Stage 6 — Tumour Location Analysis — Initialized")

except Exception as e:
    print(f"❌ Failed to load session: {e}")
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# ⚙️  PATHS
# ─────────────────────────────────────────────────────────────────────────

RESULTS_BASE = RESULTS_DIR
MRI_DIR = Path(str(MONAI_DIR))

# ─────────────────────────────────────────────────────────────────────────


def banner(t):
    print("\n" + "═" * 60)
    print(f"  {t}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────


try:
    import numpy as np
    import nibabel as nib
    from scipy import ndimage
except ImportError as e:
    print(f"❌ {e} — run: pip install nibabel numpy scipy")
    sys.exit(1)


def analyse_location(seg_path: Path, t1_path: Path) -> dict:
    """
    Analyse tumour location and ramification from segmentation mask.
    Returns a dict of findings.
    """

    # ── Load segmentation and T1 ──────────────────────────────────
    seg_nib = nib.load(str(seg_path))

    # --- Quality Check Verification ---
    qual_path = seg_path.parent / "segmentation_quality.json"
    if qual_path.exists():
        with open(qual_path) as f:
            q_data = json.load(f)
            qual = q_data.get("quality", q_data)
            if not qual.get("tumour_inside_brain", True):
                print("  ⚠️  WARNING: Poor MRI quality detected. CT fusion will be more conservative.")

    seg_arr = seg_nib.get_fdata().astype(np.uint8)
    affine = seg_nib.affine
    vox_sizes = seg_nib.header.get_zooms()[:3]
    vox_mm3 = float(vox_sizes[0] * vox_sizes[1] * vox_sizes[2])

    t1_nib = nib.load(str(t1_path))
    t1_arr = t1_nib.get_fdata().astype(np.float32)

    shape = seg_arr.shape  # (X, Y, Z)
    _cx, _cy, _cz = shape[0] // 2, shape[1] // 2, shape[2] // 2  # volume centre

    tumour_any = seg_arr > 0
    if not tumour_any.any():
        return {"error": "No tumour voxels found in segmentation"}

    # ── Filter to largest connected component ──────────────────────
    # The segmentation often contains hundreds of tiny scattered fragments
    # that corrupt centroid, bounding box, and hemisphere calculations.
    # Keep only the largest contiguous tumour mass.
    tumour_labeled, tumour_n_comp = ndimage.label(tumour_any)
    if tumour_n_comp > 1:
        tumour_sizes = ndimage.sum(tumour_any, tumour_labeled, range(1, tumour_n_comp + 1))
        largest_label = int(np.argmax(tumour_sizes)) + 1
        tumour_any = tumour_labeled == largest_label

    # ── Tumour voxel coordinates ───────────────────────────────────
    coords = np.argwhere(tumour_any)  # [N, 3] in voxel space
    x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]

    # ── Centre of mass (voxel + mm) ────────────────────────────────
    com_vox = coords.mean(axis=0)  # [x, y, z] voxel
    # Convert to RAS mm using affine
    com_h = np.array([com_vox[0], com_vox[1], com_vox[2], 1.0])
    com_ras = affine @ com_h  # [R, A, S, 1]

    # ── Hemisphere ────────────────────────────────────────────────
    # In RAS coordinates: R > 0 = right, R < 0 = left
    # In voxel space for typical brain MRI: x < midline = left or right
    # depends on orientation. Use RAS R coordinate directly.
    ras_r = com_ras[0]
    hemisphere = "Left" if ras_r < 0 else "Right"

    # Fraction of tumour voxels on each side
    # Convert all tumour voxels to RAS
    coords_h = np.column_stack([coords, np.ones(len(coords))])  # [N,4]
    ras_all = (affine @ coords_h.T).T  # [N,4]
    r_vals = ras_all[:, 0]
    pct_left = float((r_vals < 0).sum() / len(r_vals) * 100)
    pct_right = 100 - pct_left
    crosses_midline = pct_left > 5 and pct_right > 5

    # ── Bounding box ──────────────────────────────────────────────
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    z_min, z_max = int(z_coords.min()), int(z_coords.max())

    bbox_mm = {
        "x_mm": round((x_max - x_min) * float(vox_sizes[0]), 1),
        "y_mm": round((y_max - y_min) * float(vox_sizes[1]), 1),
        "z_mm": round((z_max - z_min) * float(vox_sizes[2]), 1),
    }

    # ── Lobe estimation (based on voxel position heuristics) ──────
    # MNI152 1mm space rough lobe boundaries (voxel coords for 160x256x256):
    # These are approximate — real atlas parcellation needs registration
    # but gives a reasonable estimate for standard brain MRI orientation.
    #
    # Z axis (axial):  low Z = inferior, high Z = superior
    # Y axis (coronal): low Y = posterior, high Y = anterior
    # X axis:           lateral position
    #
    # Use centre of mass in normalised coordinates [0,1]
    com_vox[0] / shape[0]  # lateral (0=left edge, 1=right edge)
    cy_n = com_vox[1] / shape[1]  # anterior-posterior (0=post, 1=ant)
    cz_n = com_vox[2] / shape[2]  # inferior-superior (0=inf, 1=sup)

    # Anterior-posterior split: >0.45 = frontal/parietal, <0.45 = occipital
    # Superior-inferior split:  >0.50 = parietal, <0.50 = temporal
    #
    # IMPORTANT — Coordinate system correction (H3 fix):
    # Stage 1b applies LPS+ orientation via DICOMOrient("LPS").
    # In LPS+: Z increases Inferior→Superior (LPS Z-axis: low=inferior, high=superior).
    # The original code used RAS+ conventions (Z high = inferior) which inverts
    # the superior/inferior axis — parietal tumors classified as temporal and vice versa.
    # All RAS-based checks (ras_s > 40mm) remain correct in RAS mm space.
    lobes = []
    if cy_n > 0.55:
        lobes.append("Frontal")
    if 0.35 < cy_n <= 0.65 and cz_n < 0.50:  # cz_n < 0.50 = superior in LPS+
        lobes.append("Parietal")
    if cy_n < 0.45 and cz_n > 0.50:  # cz_n > 0.50 = inferior in LPS+
        lobes.append("Temporal")
    if cy_n < 0.35:
        lobes.append("Occipital")
    if not lobes:
        lobes = ["Fronto-parietal"]  # default for central lesions

    # Rolandic / perirolandic check
    # Central sulcus runs mid-AP, superior. RAS S > 40mm above AC-PC (z=0)
    # is reliably in the rolandic cortex for standard MNI space.
    # Broaden AP range: 0.35 < cy_n < 0.70 covers fronto-parietal junction.
    ras_s = float(com_ras[2])  # superior coordinate in mm
    is_rolandic = ((0.35 < cy_n < 0.70) and (cz_n > 0.50)) or (ras_s > 40.0 and cz_n > 0.50)
    if is_rolandic:
        # Fronto-parietal / rolandic lesions always straddle frontal + parietal
        if "Frontal" not in lobes:
            lobes.append("Frontal")
        if "Parietal" not in lobes:
            lobes.append("Parietal")

    lobe_str = " / ".join(lobes)
    if is_rolandic:
        lobe_str += " (Rolandic region — motor cortex proximity)"

    # ── Cortical vs subcortical ────────────────────────────────────
    # Estimate brain surface by finding outer extent of T1 signal
    brain_mask_t1 = t1_arr > (t1_arr.max() * 0.08)
    brain_mask_t1 = ndimage.binary_fill_holes(brain_mask_t1)

    # FIX H — Adaptar erosão ao tamanho do voxel (10mm e 20mm reais)
    # binary_erosion uses a cubic structuring element (equal in all 3 axes).
    # Use min(vox_sizes) so that isotropic-equivalent erosion is applied —
    # e.g. for 2×1×2mm voxels, 10mm requires 5 iterations not 10.
    vox_min = float(min(vox_sizes))
    iter_10mm = int(10 / vox_min) if vox_min > 0 else 10
    iter_20mm = int(20 / vox_min) if vox_min > 0 else 20

    eroded_1cm = ndimage.binary_erosion(brain_mask_t1, iterations=iter_10mm)
    eroded_2cm = ndimage.binary_erosion(brain_mask_t1, iterations=iter_20mm)

    com_i = tuple(com_vox.astype(int))
    try:
        bool(brain_mask_t1[com_i]) and not bool(eroded_1cm[com_i])
        in_subcortical = bool(eroded_1cm[com_i]) and not bool(eroded_2cm[com_i])
        in_deep = bool(eroded_2cm[com_i])
    except IndexError:
        _in_cortical, in_subcortical, in_deep = True, False, False

    if in_deep:
        depth_str = "Deep (>20mm from cortex)"
    elif in_subcortical:
        depth_str = "Subcortical (10-20mm from cortex)"
    else:
        depth_str = "Cortical / juxtacortical (<10mm from cortex)"

    # ── Distance from midline ──────────────────────────────────────
    midline_dist_mm = abs(float(com_ras[0]))

    # ── Distance from ventricles ──────────────────────────────────────
    t1_norm = t1_arr / (t1_arr.max() + 1e-8)
    vent_raw = (t1_norm < 0.08) & brain_mask_t1

    # Keep only the central component (ventricles) vs other dark spots
    labeled, n_comp = ndimage.label(vent_raw)
    if n_comp > 0:
        brain_com = ndimage.center_of_mass(brain_mask_t1)
        min_dist = float("inf")
        best_label = 1
        for lbl in range(1, n_comp + 1):
            comp_mask = labeled == lbl
            if comp_mask.sum() < 200:
                continue  # filter noise
            com = ndimage.center_of_mass(comp_mask)
            dist = np.linalg.norm(np.array(com) - np.array(brain_com))
            if dist < min_dist:
                min_dist, best_label = dist, lbl

        vent_mask = labeled == best_label
        # Dilate to define periventricular zone
        vent_mask_dilated = ndimage.binary_dilation(vent_mask, iterations=5)
        vent_coords = np.argwhere(vent_mask_dilated)

        diffs = vent_coords - com_vox
        dists = np.sqrt((diffs**2 * np.array(vox_sizes) ** 2).sum(axis=1))
        vent_dist_mm = float(dists.min())
    else:
        vent_dist_mm = -1.0

    # ── Sub-region locations ───────────────────────────────────────
    region_info = {}
    for label, name in [(1, "Necrotic"), (2, "Edema"), (3, "Enhancing")]:
        mask_r = seg_arr == label
        if mask_r.any():
            c_r = np.argwhere(mask_r).mean(axis=0)
            c_ras = (affine @ np.array([c_r[0], c_r[1], c_r[2], 1.0]))[:3]
            vol_r = float(mask_r.sum()) * vox_mm3 / 1000.0
            region_info[name] = {
                "volume_cc": round(vol_r, 2),
                "centre_ras_mm": [round(float(c_ras[0]), 1), round(float(c_ras[1]), 1), round(float(c_ras[2]), 1)],
            }

    # ── Oedema spread ─────────────────────────────────────────────
    edema_mask = seg_arr == 2
    if edema_mask.any():
        ed_coords = np.argwhere(edema_mask)
        ed_x_span = (int(ed_coords[:, 0].min()), int(ed_coords[:, 0].max()))
        ed_span_mm = round((ed_x_span[1] - ed_x_span[0]) * float(vox_sizes[0]), 1)
    else:
        ed_span_mm = 0.0

    # ── Eloquent area proximity ────────────────────────────────────
    eloquent = []
    if is_rolandic:
        eloquent.append("Primary motor cortex (M1) — movement control")
    if cy_n > 0.60 and hemisphere == "Left":
        eloquent.append("Broca area (left frontal) — speech production")
    if 0.35 < cy_n < 0.50 and hemisphere == "Left":
        eloquent.append("Wernicke area (left temporal) — speech comprehension")
    if cz_n < 0.40:
        eloquent.append("Visual cortex proximity")
    if vent_dist_mm < 15 and vent_dist_mm > 0:
        eloquent.append(f"Periventricular ({vent_dist_mm:.0f}mm from ventricle)")
    if not eloquent:
        eloquent.append("No major eloquent area immediately adjacent")

    # ── Core-only bounding box (clinical diameter = necrosis + enhancing, NOT edema) ──
    # Use ONLY the largest connected component to exclude scattered false-positive
    # fragments that inflate the bounding box (e.g. 125mm vs radiologist's 45mm)
    core_mask = (seg_arr == 1) | (seg_arr == 4)  # NCR + ET (BraTS 2021: ET = label 4)
    if core_mask.any():
        core_labeled, core_n = ndimage.label(core_mask)
        if core_n > 1:
            core_sizes = ndimage.sum(core_mask, core_labeled, range(1, core_n + 1))
            largest_core_label = int(np.argmax(core_sizes)) + 1
            core_mask_clean = core_labeled == largest_core_label
        else:
            core_mask_clean = core_mask

        core_coords = np.argwhere(core_mask_clean)
        cx_min, cx_max = int(core_coords[:, 0].min()), int(core_coords[:, 0].max())
        cy_min, cy_max = int(core_coords[:, 1].min()), int(core_coords[:, 1].max())
        cz_min, cz_max = int(core_coords[:, 2].min()), int(core_coords[:, 2].max())
        core_bbox_mm = {
            "x_mm": round((cx_max - cx_min) * float(vox_sizes[0]), 1),
            "y_mm": round((cy_max - cy_min) * float(vox_sizes[1]), 1),
            "z_mm": round((cz_max - cz_min) * float(vox_sizes[2]), 1),
        }
        core_max_diam_mm = round(max(core_bbox_mm.values()), 1)
    else:
        core_bbox_mm = bbox_mm
        core_max_diam_mm = round(max(bbox_mm.values()), 1)

    # ── Assemble result ───────────────────────────────────────────
    return {
        "hemisphere": hemisphere,
        "pct_left": round(pct_left, 1),
        "pct_right": round(pct_right, 1),
        "crosses_midline": crosses_midline,
        "lobes": lobe_str,
        "depth": depth_str,
        "is_rolandic": is_rolandic,
        "centre_ras_mm": [round(float(com_ras[0]), 1), round(float(com_ras[1]), 1), round(float(com_ras[2]), 1)],
        "midline_dist_mm": round(midline_dist_mm, 1),
        "vent_dist_mm": round(vent_dist_mm, 1) if vent_dist_mm > 0 else "N/A",
        "bounding_box_mm": bbox_mm,
        "max_diameter_mm": core_max_diam_mm,  # clinical: core only (excl. edema)
        "full_mask_diameter_mm": round(max(bbox_mm.values()), 1),  # whole mask incl. edema
        "core_bbox_mm": core_bbox_mm,
        "eloquent_areas": eloquent,
        "sub_regions": region_info,
        "oedema_span_mm": ed_span_mm,
        "vox_size_mm": [round(float(v), 2) for v in vox_sizes],
    }


def print_location_report(r: dict):
    """Print formatted location report to console."""
    if "error" in r:
        print(f"  ❌ {r['error']}")
        return

    print(f"\n  Hemisphere        : {r['hemisphere']}")
    if r["crosses_midline"]:
        print(f"  Midline crossing  : ⚠️  YES — {r['pct_left']:.0f}% left / {r['pct_right']:.0f}% right")
    else:
        print("  Midline crossing  : No")
    print(f"  Lobe / region     : {r['lobes']}")
    print(f"  Depth             : {r['depth']}")
    print(f"  Centre (RAS mm)   : R={r['centre_ras_mm'][0]}  A={r['centre_ras_mm'][1]}  S={r['centre_ras_mm'][2]}")
    print(f"  From midline      : {r['midline_dist_mm']} mm")
    if r["vent_dist_mm"] != "N/A":
        print(f"  From ventricles   : {r['vent_dist_mm']} mm")

    print("\n  3D bounding box:")
    bb = r["bounding_box_mm"]
    print(f"    X (lateral)   : {bb['x_mm']} mm")
    print(f"    Y (ant-post)  : {bb['y_mm']} mm")
    print(f"    Z (inf-sup)   : {bb['z_mm']} mm")
    print(f"    Max diameter  : {r['max_diameter_mm']} mm")

    print(f"\n  Oedema lateral spread : {r['oedema_span_mm']} mm")

    print("\n  Eloquent area proximity:")
    for e in r["eloquent_areas"]:
        prefix = "  ⚠️ " if "motor" in e.lower() or "speech" in e.lower() or "broca" in e.lower() else "  ℹ️ "
        print(f"  {prefix} {e}")

    print("\n  Sub-region centres (RAS mm):")
    for name, info in r.get("sub_regions", {}).items():
        c = info["centre_ras_mm"]
        print(f"    {name:12s}: vol={info['volume_cc']:.1f}cc  R={c[0]}  A={c[1]}  S={c[2]}")


# ─────────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner("TUMOUR LOCATION AND RAMIFICATION ANALYSIS")

    # Guard: check for previous quality results
    qual_path = OUTPUT_DIR / "segmentation_quality.json"
    if qual_path.exists():
        with open(qual_path) as f:
            q_data = json.load(f)
            qual = q_data.get("quality", q_data)
            vol = qual.get("v_wt_cc", qual.get("tumour_volume_cc", 0))
            non_gbm = qual.get("non_glioblastoma_suspected", False)
            zero_tumor = qual.get("zero_tumor_volume", vol < 0.01)
            if non_gbm:
                print("ℹ️  SKIPPED: Non-glioblastoma suspected (model did not activate).")
                with open(OUTPUT_DIR / "location_results.json", "w") as f:
                    json.dump(
                        {"status": "skipped", "reason": "non_glioblastoma_suspected", "volume_cc": float(vol)},
                        f,
                        indent=2,
                    )
                sys.exit(0)
            if zero_tumor:
                print("ℹ️  SKIPPED: Zero-tumour volume detected.")
                with open(OUTPUT_DIR / "location_results.json", "w") as f:
                    json.dump(
                        {"status": "skipped", "reason": "zero_tumor_volume", "volume_cc": float(vol)}, f, indent=2
                    )
                sys.exit(0)
            if not qual.get("tumour_inside_brain", True) or vol < 0.01:
                print("❌ Fatal: Segmentation quality insufficient for location analysis.")
                print(f"   Tumor volume: {vol:.3f}cc (minimum: 0.01cc)")
                if vol < 0.01:
                    print("   Note: Very small tumor detected. Consider adjusting segmentation parameters.")
                sys.exit(1)

    try:
        # Load segmentation from current session output directory
        target_names = ["segmentation_ct_merged.nii.gz", "segmentation_full.nii.gz", "segmentation_ensemble.nii.gz"]
        seg_path = None
        for name in target_names:
            p = OUTPUT_DIR / name
            if p.exists():
                seg_path = p
                break

        if not seg_path:
            print(f"  ❌ No AI segmentation found in {OUTPUT_DIR}")
            sys.exit(1)

        t1_path = MRI_DIR / "t1.nii.gz"
        print(f"  Segmentation : {seg_path.name}")
        print(f"  T1 reference : {t1_path}")

        if not t1_path.exists():
            print(f"  ❌ T1 not found at {t1_path}")
            sys.exit(1)

        print("\n  Computing location analysis…", end=" ", flush=True)
        result = analyse_location(seg_path, t1_path)
        print("Done")

        banner("LOCATION FINDINGS")
        print_location_report(result)

        # Save to JSON alongside results
        import json

        class _NumpyEncoder(json.JSONEncoder):
            """Handle numpy types that the default encoder can't serialise."""

            def default(self, obj):
                import numpy as _np

                if isinstance(obj, (_np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, (_np.floating,)):
                    return float(obj)
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        out_json = OUTPUT_DIR / "tumour_location.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, cls=_NumpyEncoder)
        print(f"\n  Results saved → {out_json}")

        banner("CLINICAL SUMMARY")
        hemi = result["hemisphere"]
        lobe = result["lobes"]
        diam = result["max_diameter_mm"]
        dist = result["midline_dist_mm"]
        eq = result["eloquent_areas"]

        print(f"""
  The tumour is located in the {hemi} {lobe}.
  Maximum diameter: {diam:.0f} mm ({diam / 10:.1f} cm)
  Distance from midline: {dist:.0f} mm
  Depth: {result["depth"]}

  Eloquent area risk:""")
        for e in eq:
            print(f"    • {e}")

        if result["is_rolandic"]:
            print("""
  ⚠️  ROLANDIC / PERIROLANDIC LOCATION:
     The tumour is in or near the motor cortex (Rolandic area).
     This is clinically critical:
       → Risk of motor deficit (arm/leg weakness) if resected
       → Surgical planning must include functional MRI (fMRI)
         and/or intraoperative neuromonitoring
       → Radiotherapy margins must carefully spare motor cortex
       → The radiologist's report confirms this location
""")

    except FileNotFoundError as e:
        print(f"\n  ❌ {e}")
        print("  Run brain_tumor_analysis.py first")
