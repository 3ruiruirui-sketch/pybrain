# pybrain/clinical/consistency.py
"""
Biological plausibility and clinical consistency checks for brain tumor segmentations.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Any, Tuple, List

def validate_clinical_consistency(
    seg: np.ndarray, 
    p_ensemble: np.ndarray, 
    vox_vol_cc: float,
    config: Dict[str, Any],
    contributed_models: List[str]
) -> Dict[str, Any]:
    """
    Performs clinical sanity checks on the final segmentation.
    Checks for: 
      1. Core/Whole-Tumor ratio (Adaptive re-segmentation recovery).
      2. Isolated edema islands.
      3. Uncertainty/Entropy in tumor center.
    """
    clinical_cfg = config.get("clinical", {})
    wt_tc_min_ratio = clinical_cfg.get("core_min_ratio", 0.05)
    entropy_warn_thresh = clinical_cfg.get("entropy_warn", 0.7)

    wt_mask = (seg > 0).astype(np.float32)
    tc_mask = ((seg == 1) | (seg == 3)).astype(np.float32)
    ed_mask = (seg == 2).astype(np.float32)

    v_wt = float(wt_mask.sum()) * vox_vol_cc
    v_tc = float(tc_mask.sum()) * vox_vol_cc

    results = {
        "status": "OK",
        "v_wt_cc": round(v_wt, 2),
        "v_tc_cc": round(v_tc, 2),
        "tc_pct_of_wt": round((v_tc / (v_wt + 1e-8)) * 100, 1) if v_wt > 0 else 0.0,
        "core_empty_warning": False,
        "adaptive_reseg_applied": False,
        "continuity_warning": False,
        "isolated_edema_cc": 0.0,
        "uncertainty_flag": False,
        "centre_entropy": 0.0,
        "requires_manual_review": False,
        "contributing_models": contributed_models,
    }

    # 1. CORE RATIO CHECK & RECOVERY
    if v_wt > 5.0 and (v_tc / v_wt) < wt_tc_min_ratio:
        results["core_empty_warning"] = True
        results["requires_manual_review"] = True
        
        # Adaptive recovery: relax thresholds
        # Note: This usually involves re-running thresholding logic, 
        # which might be better placed in a higher-level orchestrator.
        # For now, we flag it.
        results["status"] = "WARNING: CORE EMPTY"

    # 2. CONTINUITY CHECK
    if ed_mask.sum() > 0 and tc_mask.sum() > 0:
        tc_dil = ndimage.binary_dilation(tc_mask.astype(bool), iterations=3)
        iso_ed = ed_mask.astype(bool) & ~tc_dil
        iso_vol = float(iso_ed.sum()) * vox_vol_cc
        if iso_vol > 1.0:
            results["continuity_warning"] = True
            results["isolated_edema_cc"] = round(iso_vol, 2)
            results["status"] = "WARNING: ISOLATED EDEMA"

    # 3. ENTROPY CHECK (Binary entropy per channel, then averaged)
    p_safe = np.clip(p_ensemble, 1e-8, 1.0 - 1e-8)
    # H(p) = -[p*log(p) + (1-p)*log(1-p)]
    ent_per_chan = -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))
    ent_map = np.mean(ent_per_chan, axis=0) # Average across TC, WT, ET
    
    if wt_mask.sum() > 0:
        avg_ent = float(ent_map[wt_mask.astype(bool)].mean()) / np.log(2.0)
        results["centre_entropy"] = round(avg_ent, 3)
        if avg_ent > entropy_warn_thresh:
            results["uncertainty_flag"] = True
            results["requires_manual_review"] = True
            results["status"] = "UNCERTAIN: High entropy"

    return results
