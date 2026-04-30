# pybrain/analysis/molecular.py
"""
PY-BRAIN v2 — Molecular Status Prediction (IDH / MGMT)
========================================================
Predicts IDH mutation and MGMT methylation status from tumour
segmentation and radiomic features using imaging correlates.

Current method: Heuristic rules based on published imaging correlates.
Future upgrade: XGBoost trained on TCGA-GBM + BraTS with confirmed labels.

IDH mutation:
  - Correlates with less enhancement, more FLAIR, frontal location
  - AUC ~0.80 with validated radiomics (Zhang et al. 2017)

MGMT methylation:
  - Correlates with less necrosis, homogeneous enhancement
  - AUC ~0.60-0.75 (harder to predict from imaging alone)

Usage:
    from pybrain.analysis.molecular import predict_molecular_status
    result = predict_molecular_status(volumes_cc, morphology, radiomics)
"""

from typing import Any, Optional

import numpy as np

from pybrain.io.logging_utils import get_logger

logger = get_logger("molecular")


def _to_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def predict_molecular_status(
    volumes_cc: dict[str, float],
    morphology: Optional[dict] = None,
    radiomics: Optional[dict] = None,
    patient_info: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Predict IDH mutation and MGMT methylation from imaging features.

    Args:
        volumes_cc:   dict with keys whole_tumour, enhancing, edema, necrotic_core (in cc)
        morphology:   dict from Stage 7 (sphericity, max_diameter, etc.)
        radiomics:    dict from Stage 8 (pyradiomics features)
        patient_info: dict with age, sex

    Returns:
        dict with idh and mgmt predictions, probabilities, and confidence
    """
    wt = float(volumes_cc.get("whole_tumour", 0))
    et = float(volumes_cc.get("enhancing", 0))
    ed = float(volumes_cc.get("edema", 0))
    nc = float(volumes_cc.get("necrotic_core", 0))

    if wt < 0.1:
        logger.warning("No tumour detected — cannot predict molecular status")
        return {
            "idh": {"prediction": "unknown", "probability": 0.5, "method": "none", "note": "No tumour detected"},
            "mgmt": {"prediction": "unknown", "probability": 0.5, "method": "none", "note": "No tumour detected"},
        }

    features = {
        "wt_volume_cc": wt,
        "et_volume_cc": et,
        "ed_volume_cc": ed,
        "nc_volume_cc": nc,
        "et_wt_ratio": et / wt if wt > 0 else 0,
        "nc_wt_ratio": nc / wt if wt > 0 else 0,
        "ed_wt_ratio": ed / wt if wt > 0 else 0,
        "ed_et_ratio": ed / et if et > 0 else 0,
    }

    if morphology:
        features["sphericity"] = float(morphology.get("sphericity", 0.5))
        features["max_diameter_mm"] = float(morphology.get("max_axial_diameter_mm", 0))
        features["surface_volume_ratio"] = float(morphology.get("surface_volume_ratio", 0))

    if patient_info:
        age = patient_info.get("age", 0)
        if age and str(age).isdigit() and int(age) > 0:
            age = int(age)
            features["age"] = age
            if age < 45:
                features["age_factor"] = 0.15
            elif age > 65:
                features["age_factor"] = -0.15
            else:
                features["age_factor"] = 0.0

    idh_result = _predict_idh(features)
    mgmt_result = _predict_mgmt(features)

    idh_conf = abs(idh_result["probability"] - 0.5) * 2
    mgmt_conf = abs(mgmt_result["probability"] - 0.5) * 2

    result = {
        "idh": {
            **{k: _to_serializable(v) for k, v in idh_result.items()},
            "confidence": _to_serializable(round(idh_conf, 2)),
            "confidence_level": ("high" if idh_conf > 0.5 else "moderate" if idh_conf > 0.25 else "low"),
        },
        "mgmt": {
            **{k: _to_serializable(v) for k, v in mgmt_result.items()},
            "confidence": _to_serializable(round(mgmt_conf, 2)),
            "confidence_level": ("high" if mgmt_conf > 0.5 else "moderate" if mgmt_conf > 0.25 else "low"),
        },
        "features_used": {k: _to_serializable(v) for k, v in features.items()},
        "disclaimer": (
            "Molecular predictions are based on imaging patterns only. "
            "Tissue biopsy and molecular testing remain the gold standard. "
            "These predictions should NOT be used for treatment decisions."
        ),
    }

    logger.info(
        f"Molecular prediction: "
        f"IDH={idh_result['prediction']} ({idh_result['probability']:.0%}), "
        f"MGMT={mgmt_result['prediction']} ({mgmt_result['probability']:.0%})"
    )

    return result


def _predict_idh(features: dict) -> dict:
    """Predict IDH mutation status from imaging features."""
    score = 0.5

    et_wt = features.get("et_wt_ratio", 0.5)
    if et_wt < 0.3:
        score += 0.15
    elif et_wt > 0.6:
        score -= 0.15

    nc_wt = features.get("nc_wt_ratio", 0.1)
    if nc_wt < 0.05:
        score += 0.10
    elif nc_wt > 0.15:
        score -= 0.10

    ed_wt = features.get("ed_wt_ratio", 0.3)
    if ed_wt > 0.5:
        score += 0.10
    elif ed_wt < 0.2:
        score -= 0.05

    if features.get("sphericity", 0.5) > 0.7:
        score += 0.05

    score += features.get("age_factor", 0.0)

    wt = features.get("wt_volume_cc", 30)
    if wt > 80:
        score -= 0.10
    elif wt < 15:
        score += 0.05

    score = max(0.05, min(0.95, score))

    return {
        "prediction": "mutant" if score > 0.5 else "wildtype",
        "probability": round(score, 3),
        "method": "radiomics_heuristic_v1",
    }


def _predict_mgmt(features: dict) -> dict:
    """Predict MGMT methylation status from imaging features."""
    score = 0.5

    nc_wt = features.get("nc_wt_ratio", 0.1)
    if nc_wt < 0.05:
        score += 0.10
    elif nc_wt > 0.20:
        score -= 0.10

    et_cc = features.get("et_volume_cc", 10)
    if 5 < et_cc < 25:
        score += 0.05
    elif et_cc > 40:
        score -= 0.05

    ed_et = features.get("ed_et_ratio", 1.0)
    if ed_et > 1.5:
        score += 0.05

    wt = features.get("wt_volume_cc", 30)
    if wt > 60:
        score -= 0.05

    score = max(0.05, min(0.95, score))

    return {
        "prediction": "methylated" if score > 0.5 else "unmethylated",
        "probability": round(score, 3),
        "method": "radiomics_heuristic_v1",
    }
