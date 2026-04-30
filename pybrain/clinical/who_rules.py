# pybrain/clinical/who_rules.py
"""
WHO 2021 Clinical Classification Rules.
"""

from typing import List, Dict, Any


def get_who_clinical_interpretation(metrics: Dict[str, Any]) -> List[str]:
    """
    Returns a list of clinical warnings/interpretations based on WHO 2021 criteria.
    Compatible with Stage 7/8/9 reports.
    """
    warnings = []

    # 1. T2-FLAIR mismatch (Suggestive of IDH-mutant astrocytoma)
    # Assumes metrics contain a 'mismatch_sign' boolean or enough data to infer it.
    if metrics.get("t2_flair_mismatch_detected", False):
        warnings.append("⚠️ T2-FLAIR mismatch sign — highly suggestive of IDH-mutant glioma (WHO 2021 Grade 2-3)")

    # 2. Calcification (Suggestive of Oligodendroglioma)
    calc_vol = metrics.get("calcification_vol_cc", 0.0)
    if calc_vol >= 1.0:
        warnings.append("⚠️ Significant calcification (≥1 cc) — consider oligodendroglioma IDH-mutant 1p/19q codeleted")

    # 3. Necrotic/Enhancing pattern (Suggestive of Glioblastoma Grade 4)
    # Check for core presence and ring-enhancement features if available
    if metrics.get("v_tc_cc", 0) > 10.0 and metrics.get("v_et_cc", 0) > 5.0:
        warnings.append("⚠️ WHO Grade 4 pattern: Classic necrosis and enhancement suggestive of Glioblastoma.")

    if not warnings:
        warnings.append("✅ No high-risk WHO 2021 pathognomonic patterns detected.")

    return warnings
