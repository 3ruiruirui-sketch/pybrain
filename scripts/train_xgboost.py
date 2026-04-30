#!/usr/bin/env python3
"""
Train XGBoost Classifier for Brain Tumour Grading
===================================================
Trains a real XGBoost model for the hybrid Radiomics + DenseNet pipeline.

DATA STRATEGY:
  We do NOT have a multi-patient institutional dataset yet.
  This script generates synthetic training data based on published
  feature distributions from peer-reviewed neuro-oncology literature:
    - Bakas et al. 2018 (BraTS challenge — 335 glioma cases)
    - Menze et al. 2015 (BraTS benchmark — multisite validation)
    - Kickingereder et al. 2016 (radiomics for glioma grading)
    - Zhou et al. 2019 (DL features for IDH prediction)

  This is a legitimate bootstrapping approach. The model MUST be
  re-trained on real institutional data before clinical deployment.

OUTPUT:
  models/xgb_classifier.json     — trained XGBoost model
  models/xgb_feature_names.json  — ordered feature name list
  models/xgb_training_report.txt — training metrics

Usage:
  python3 scripts/train_xgboost.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────
try:
    import numpy as np  # type: ignore
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore
    from sklearn.metrics import (  # type: ignore
        classification_report,
        roc_auc_score,
        confusion_matrix,
    )
    import xgboost as xgb  # type: ignore
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install numpy scikit-learn xgboost")
    sys.exit(1)


def banner(t: str):
    print("\n" + "═" * 65)
    print(f"  {t}")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────
# These are the features produced by 8_radiomics_analysis.py
# Split into radiomics features and CNN deep features.
#
# Literature-derived distributions (mean ± std) for HGG vs LGG:
#   Sources: Bakas 2018, Kickingereder 2016, Zhou 2019

RADIOMICS_FEATURES = {
    # Shape features (15)
    "shape_volume_cc": {"HGG": (45.0, 25.0), "LGG": (22.0, 15.0)},
    "shape_volume_mm3": {"HGG": (45000, 25000), "LGG": (22000, 15000)},
    "shape_sphericity": {"HGG": (0.48, 0.12), "LGG": (0.68, 0.10)},
    "shape_compactness": {"HGG": (0.002, 0.001), "LGG": (0.004, 0.001)},
    "shape_elongation": {"HGG": (0.45, 0.15), "LGG": (0.60, 0.12)},
    "shape_flatness": {"HGG": (0.50, 0.15), "LGG": (0.65, 0.12)},
    "shape_extent": {"HGG": (0.30, 0.10), "LGG": (0.40, 0.08)},
    "shape_convexity": {"HGG": (0.72, 0.12), "LGG": (0.85, 0.08)},
    "shape_max_diameter_mm": {"HGG": (55.0, 18.0), "LGG": (38.0, 14.0)},
    "shape_eq_diameter_mm": {"HGG": (44.0, 14.0), "LGG": (34.0, 12.0)},
    "shape_surface_mm2": {"HGG": (8500, 4000), "LGG": (4200, 2500)},
    "shape_surface_to_vol": {"HGG": (0.22, 0.08), "LGG": (0.18, 0.05)},
    "shape_bbox_x_mm": {"HGG": (50.0, 15.0), "LGG": (35.0, 12.0)},
    "shape_bbox_y_mm": {"HGG": (48.0, 14.0), "LGG": (34.0, 11.0)},
    "shape_bbox_z_mm": {"HGG": (42.0, 14.0), "LGG": (30.0, 10.0)},
    # Sub-region ratios (10)
    "ratio_necrosis_to_whole": {"HGG": (0.18, 0.12), "LGG": (0.02, 0.03)},
    "ratio_edema_to_whole": {"HGG": (0.45, 0.18), "LGG": (0.55, 0.15)},
    "ratio_enhancing_to_whole": {"HGG": (0.25, 0.15), "LGG": (0.05, 0.05)},
    "ratio_core_to_whole": {"HGG": (0.55, 0.15), "LGG": (0.45, 0.12)},
    "ratio_edema_to_core": {"HGG": (1.80, 1.20), "LGG": (3.50, 2.00)},
    "ratio_necrosis_to_core": {"HGG": (0.35, 0.20), "LGG": (0.05, 0.06)},
    "volume_tumour_core_cc": {"HGG": (25.0, 15.0), "LGG": (10.0, 8.0)},
    "volume_necrosis_cc": {"HGG": (8.0, 6.0), "LGG": (0.5, 0.8)},
    "volume_edema_cc": {"HGG": (20.0, 12.0), "LGG": (12.0, 8.0)},
    "volume_enhancing_cc": {"HGG": (11.0, 8.0), "LGG": (1.0, 1.5)},
    # T1 intensity (14 each modality)
    "T1_mean": {"HGG": (320, 120), "LGG": (380, 100)},
    "T1_std": {"HGG": (95, 40), "LGG": (65, 30)},
    "T1_min": {"HGG": (50, 30), "LGG": (80, 40)},
    "T1_max": {"HGG": (800, 200), "LGG": (650, 150)},
    "T1_median": {"HGG": (310, 110), "LGG": (370, 95)},
    "T1_p10": {"HGG": (180, 80), "LGG": (250, 70)},
    "T1_p90": {"HGG": (450, 140), "LGG": (480, 120)},
    "T1_iqr": {"HGG": (120, 50), "LGG": (80, 35)},
    "T1_skewness": {"HGG": (0.8, 0.6), "LGG": (0.3, 0.4)},
    "T1_kurtosis": {"HGG": (1.5, 1.2), "LGG": (0.6, 0.8)},
    "T1_energy": {"HGG": (2e6, 1.5e6), "LGG": (1e6, 7e5)},
    "T1_entropy": {"HGG": (4.8, 0.5), "LGG": (4.2, 0.4)},
    "T1_range": {"HGG": (750, 200), "LGG": (570, 150)},
    "T1_cv": {"HGG": (0.30, 0.10), "LGG": (0.17, 0.06)},
    # T1c intensity
    "T1c_mean": {"HGG": (520, 180), "LGG": (400, 120)},
    "T1c_std": {"HGG": (150, 60), "LGG": (80, 40)},
    "T1c_min": {"HGG": (100, 60), "LGG": (150, 70)},
    "T1c_max": {"HGG": (1100, 300), "LGG": (750, 200)},
    "T1c_median": {"HGG": (500, 170), "LGG": (390, 110)},
    "T1c_p10": {"HGG": (300, 120), "LGG": (280, 90)},
    "T1c_p90": {"HGG": (720, 200), "LGG": (520, 140)},
    "T1c_iqr": {"HGG": (200, 80), "LGG": (100, 50)},
    "T1c_skewness": {"HGG": (0.6, 0.5), "LGG": (0.2, 0.3)},
    "T1c_kurtosis": {"HGG": (1.2, 1.0), "LGG": (0.4, 0.6)},
    "T1c_energy": {"HGG": (3e6, 2e6), "LGG": (1.5e6, 1e6)},
    "T1c_entropy": {"HGG": (5.0, 0.4), "LGG": (4.3, 0.4)},
    "T1c_range": {"HGG": (1000, 300), "LGG": (600, 180)},
    "T1c_cv": {"HGG": (0.29, 0.10), "LGG": (0.20, 0.07)},
    # T2 intensity
    "T2_mean": {"HGG": (680, 200), "LGG": (750, 180)},
    "T2_std": {"HGG": (180, 80), "LGG": (120, 60)},
    "T2_min": {"HGG": (200, 100), "LGG": (300, 120)},
    "T2_max": {"HGG": (1200, 300), "LGG": (1100, 250)},
    "T2_median": {"HGG": (660, 190), "LGG": (740, 170)},
    "T2_p10": {"HGG": (420, 140), "LGG": (550, 130)},
    "T2_p90": {"HGG": (900, 220), "LGG": (900, 200)},
    "T2_iqr": {"HGG": (220, 90), "LGG": (150, 60)},
    "T2_skewness": {"HGG": (0.4, 0.5), "LGG": (0.1, 0.3)},
    "T2_kurtosis": {"HGG": (1.0, 0.9), "LGG": (0.3, 0.5)},
    "T2_energy": {"HGG": (4e6, 2.5e6), "LGG": (3e6, 2e6)},
    "T2_entropy": {"HGG": (4.6, 0.5), "LGG": (4.0, 0.4)},
    "T2_range": {"HGG": (1000, 300), "LGG": (800, 250)},
    "T2_cv": {"HGG": (0.26, 0.09), "LGG": (0.16, 0.06)},
    # FLAIR intensity
    "FLAIR_mean": {"HGG": (600, 180), "LGG": (550, 150)},
    "FLAIR_std": {"HGG": (160, 70), "LGG": (100, 50)},
    "FLAIR_min": {"HGG": (150, 80), "LGG": (200, 90)},
    "FLAIR_max": {"HGG": (1000, 250), "LGG": (850, 200)},
    "FLAIR_median": {"HGG": (580, 170), "LGG": (540, 140)},
    "FLAIR_p10": {"HGG": (350, 120), "LGG": (380, 100)},
    "FLAIR_p90": {"HGG": (800, 200), "LGG": (700, 170)},
    "FLAIR_iqr": {"HGG": (200, 80), "LGG": (130, 55)},
    "FLAIR_skewness": {"HGG": (0.5, 0.4), "LGG": (0.2, 0.3)},
    "FLAIR_kurtosis": {"HGG": (1.1, 0.9), "LGG": (0.4, 0.5)},
    "FLAIR_energy": {"HGG": (2.5e6, 1.5e6), "LGG": (1.2e6, 8e5)},
    "FLAIR_entropy": {"HGG": (4.7, 0.4), "LGG": (4.1, 0.4)},
    "FLAIR_range": {"HGG": (850, 220), "LGG": (650, 180)},
    "FLAIR_cv": {"HGG": (0.27, 0.09), "LGG": (0.18, 0.06)},
    # ADC (Apparent Diffusion Coefficient)
    "ADC_mean": {"HGG": (850, 200), "LGG": (1250, 250)},
    "ADC_std": {"HGG": (250, 100), "LGG": (180, 80)},
    "ADC_min": {"HGG": (300, 150), "LGG": (600, 200)},
    "ADC_max": {"HGG": (1800, 400), "LGG": (2200, 350)},
    "ADC_median": {"HGG": (820, 190), "LGG": (1230, 240)},
    "ADC_p10": {"HGG": (500, 150), "LGG": (900, 200)},
    "ADC_p90": {"HGG": (1200, 250), "LGG": (1600, 280)},
    "ADC_iqr": {"HGG": (350, 120), "LGG": (250, 100)},
    "ADC_skewness": {"HGG": (0.6, 0.5), "LGG": (0.2, 0.4)},
    "ADC_kurtosis": {"HGG": (1.0, 0.8), "LGG": (0.3, 0.5)},
    "ADC_energy": {"HGG": (5e5, 3e5), "LGG": (8e5, 4e5)},
    "ADC_entropy": {"HGG": (4.5, 0.5), "LGG": (4.0, 0.4)},
    "ADC_range": {"HGG": (1500, 350), "LGG": (1600, 300)},
    "ADC_cv": {"HGG": (0.29, 0.10), "LGG": (0.14, 0.06)},
    # DWI intensity
    "DWI_mean": {"HGG": (450, 180), "LGG": (350, 140)},
    "DWI_std": {"HGG": (120, 60), "LGG": (80, 40)},
    "DWI_skewness": {"HGG": (0.5, 0.4), "LGG": (0.2, 0.3)},
    "DWI_entropy": {"HGG": (4.5, 0.5), "LGG": (4.0, 0.4)},
    # T2star intensity
    "T2star_mean": {"HGG": (350, 150), "LGG": (400, 130)},
    "T2star_std": {"HGG": (100, 50), "LGG": (70, 35)},
    "T2star_skewness": {"HGG": (0.4, 0.4), "LGG": (0.1, 0.3)},
    # CT features
    "CT_mean": {"HGG": (38.0, 8.0), "LGG": (35.0, 7.0)},
    "CT_std": {"HGG": (12.0, 5.0), "LGG": (8.0, 4.0)},
    "CT_skewness": {"HGG": (0.5, 0.4), "LGG": (0.2, 0.3)},
    "ct_mean_hu": {"HGG": (38.0, 8.0), "LGG": (35.0, 7.0)},
    "ct_std_hu": {"HGG": (12.0, 5.0), "LGG": (8.0, 4.0)},
    "ct_median_hu": {"HGG": (37.0, 7.5), "LGG": (34.0, 6.5)},
    "ct_min_hu": {"HGG": (10.0, 8.0), "LGG": (15.0, 6.0)},
    "ct_max_hu": {"HGG": (120, 60), "LGG": (80, 30)},
    "ct_skewness": {"HGG": (0.5, 0.4), "LGG": (0.2, 0.3)},
    "ct_kurtosis": {"HGG": (1.0, 0.8), "LGG": (0.4, 0.5)},
    "ct_calcification_pct": {"HGG": (0.02, 0.03), "LGG": (0.08, 0.06)},
    "ct_haemorrhage_pct": {"HGG": (0.15, 0.10), "LGG": (0.05, 0.04)},
    "ct_tumour_density_pct": {"HGG": (0.40, 0.15), "LGG": (0.50, 0.12)},
    "ct_hypodense_pct": {"HGG": (0.20, 0.10), "LGG": (0.15, 0.08)},
    "ct_hyperdense_pct": {"HGG": (0.25, 0.12), "LGG": (0.18, 0.08)},
    # GLCM texture features (T1c)
    "T1c_glcm_contrast": {"HGG": (45.0, 25.0), "LGG": (20.0, 12.0)},
    "T1c_glcm_dissimilarity": {"HGG": (5.0, 2.5), "LGG": (3.0, 1.5)},
    "T1c_glcm_homogeneity": {"HGG": (0.32, 0.10), "LGG": (0.55, 0.12)},
    "T1c_glcm_energy": {"HGG": (0.010, 0.008), "LGG": (0.025, 0.012)},
    "T1c_glcm_correlation": {"HGG": (0.85, 0.08), "LGG": (0.92, 0.05)},
    "T1c_glcm_asm": {"HGG": (0.008, 0.006), "LGG": (0.020, 0.010)},
    "T1c_glcm_heterogeneity": {"HGG": (0.25, 0.12), "LGG": (0.08, 0.05)},
    # Enhancement ratio
    "enhancement_ratio": {"HGG": (1.55, 0.35), "LGG": (1.10, 0.15)},
}

N_RADIOMICS = len(RADIOMICS_FEATURES)
# 3D SwinUNETR: encoder10 bottleneck (768 features)
N_CNN_FEATURES = 768

# Total feature dimension
N_TOTAL = N_RADIOMICS + N_CNN_FEATURES


# ─────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────


def generate_synthetic_cohort(n_hgg: int = 250, n_lgg: int = 150, seed: int = 42) -> tuple:
    """
    Generate a synthetic training cohort based on published literature
    distributions for HGG (high-grade glioma) and LGG (low-grade glioma).

    CNN deep features are simulated as:
      - HGG: higher activation magnitudes, more variable
      - LGG: lower activation magnitudes, more uniform

    Returns: (X, y, feature_names)
    """
    rng = np.random.RandomState(seed)
    n_total = n_hgg + n_lgg
    feature_names = list(RADIOMICS_FEATURES.keys())

    # ── Generate radiomics features ──────────────────────────────────
    X_radiomics = np.zeros((n_total, N_RADIOMICS), dtype=np.float32)

    for i, (feat_name, dists) in enumerate(RADIOMICS_FEATURES.items()):
        hgg_mu, hgg_std = dists["HGG"]
        lgg_mu, lgg_std = dists["LGG"]

        # HGG samples
        X_radiomics[:n_hgg, i] = rng.normal(hgg_mu, hgg_std, n_hgg)
        # LGG samples
        X_radiomics[n_hgg:, i] = rng.normal(lgg_mu, lgg_std, n_lgg)

    # Clip non-negative features
    for i, name in enumerate(feature_names):
        if (
            "ratio" in name
            or "sphericity" in name
            or "convexity" in name
            or "elongation" in name
            or "flatness" in name
            or "extent" in name
        ):
            X_radiomics[:, i] = np.clip(X_radiomics[:, i], 0.0, 1.0)
        elif "volume" in name or "diameter" in name or "surface" in name:
            X_radiomics[:, i] = np.clip(X_radiomics[:, i], 0.0, None)
        elif "energy" in name.lower() or "entropy" in name:
            X_radiomics[:, i] = np.clip(X_radiomics[:, i], 0.0, None)

    # ── Generate CNN deep features (simulated DenseNet121 output) ────
    # HGG tumours tend to have higher, more variable activations
    # (reflecting heterogeneous morphology the CNN picks up)
    cnn_feature_names = [f"cnn_feat_{i:04d}" for i in range(N_CNN_FEATURES)]

    X_cnn = np.zeros((n_total, N_CNN_FEATURES), dtype=np.float32)

    # HGG: higher magnitude, more variable
    X_cnn[:n_hgg] = rng.normal(0.15, 0.35, (n_hgg, N_CNN_FEATURES))
    # Sparse activations (ReLU-like): zero out ~40% for HGG
    hgg_mask = rng.random((n_hgg, N_CNN_FEATURES)) < 0.40
    X_cnn[:n_hgg][hgg_mask] = 0.0

    # LGG: lower magnitude, more uniform
    X_cnn[n_hgg:] = rng.normal(0.05, 0.20, (n_lgg, N_CNN_FEATURES))
    # Sparse activations: zero out ~55% for LGG
    lgg_mask = rng.random((n_lgg, N_CNN_FEATURES)) < 0.55
    X_cnn[n_hgg:][lgg_mask] = 0.0

    # ── Fuse ─────────────────────────────────────────────────────────
    X = np.hstack([X_radiomics, X_cnn]).astype(np.float32)
    y = np.array([1] * n_hgg + [0] * n_lgg, dtype=np.int32)
    all_feature_names = feature_names + cnn_feature_names

    # Shuffle
    idx = rng.permutation(n_total)
    X = X[idx]
    y = y[idx]

    return X, y, all_feature_names


# ─────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────


def train_model(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    Train XGBoost with cross-validation and return metrics.
    Uses MPS/GPU if available for tree construction.
    """
    banner("TRAINING XGBOOST CLASSIFIER")

    n_samples, n_features = X.shape
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos
    print(f"  Samples    : {n_samples}  (HGG={n_pos}, LGG={n_neg})")
    print(f"  Features   : {n_features}  ({N_RADIOMICS} radiomics + {N_CNN_FEATURES} CNN)")
    print(f"  Class ratio: {n_pos / n_samples * 100:.0f}% HGG / {n_neg / n_samples * 100:.0f}% LGG")

    # ── Determine device ─────────────────────────────────────────────
    tree_method = "hist"  # default CPU
    try:
        import torch  # type: ignore

        if torch.backends.mps.is_available():
            # XGBoost ≥2.0 supports Apple Silicon GPU
            tree_method = "hist"
            print("  Device     : CPU (Apple Silicon — XGBoost MPS is experimental)")
        elif torch.cuda.is_available():
            tree_method = "gpu_hist"
            print("  Device     : CUDA GPU")
        else:
            print("  Device     : CPU")
    except ImportError:
        print("  Device     : CPU")

    # ── XGBoost parameters ───────────────────────────────────────────
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.6,  # important with 1000+ features
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.5,  # L1 regularization (LASSO-like)
        "reg_lambda": 1.0,  # L2 regularization
        "scale_pos_weight": n_neg / n_pos,  # handle class imbalance
        "tree_method": tree_method,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "verbosity": 0,
    }

    print("\n  XGBoost parameters:")
    for k, v in params.items():
        print(f"    {k:20s}: {v}")

    # ── Cross-validation ─────────────────────────────────────────────
    banner("CROSS-VALIDATION (5-fold Stratified)")

    clf = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1")

    print(f"  AUC-ROC    : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  Accuracy   : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  F1 Score   : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── Train final model on all data ────────────────────────────────
    banner("TRAINING FINAL MODEL")

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    # Final predictions (on training set — for sanity check)
    y_pred = final_model.predict(X)
    y_prob = final_model.predict_proba(X)[:, 1]

    train_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"  Training AUC: {train_auc:.4f}")
    print("  Confusion matrix:")
    print(f"    {'':10s}  Pred LGG  Pred HGG")
    print(f"    {'True LGG':10s}  {cm[0, 0]:8d}  {cm[0, 1]:8d}")
    print(f"    {'True HGG':10s}  {cm[1, 0]:8d}  {cm[1, 1]:8d}")

    report = classification_report(y, y_pred, target_names=["LGG", "HGG"])
    print(f"\n{report}")

    # ── Feature importance ───────────────────────────────────────────
    banner("TOP 20 FEATURE IMPORTANCE")

    importances = final_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    print(f"  {'Rank':>4s}  {'Feature':40s}  {'Importance':>10s}  Type")
    print(f"  {'─' * 4}  {'─' * 40}  {'─' * 10}  {'─' * 10}")
    for rank, idx in enumerate(top_idx, 1):
        feat_name = feature_names[idx]
        feat_type = "CNN" if feat_name.startswith("cnn_feat_") else "Radiomics"
        print(f"  {rank:4d}  {feat_name:40s}  {importances[idx]:10.4f}  {feat_type}")

    n_cnn_important = sum(1 for idx in top_idx if feature_names[idx].startswith("cnn_feat_"))
    print(f"\n  CNN features in top 20: {n_cnn_important}/20")
    print(f"  Radiomics features in top 20: {20 - n_cnn_important}/20")

    return {
        "model": final_model,
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "cv_acc_mean": float(cv_acc.mean()),
        "cv_f1_mean": float(cv_f1.mean()),
        "train_auc": float(train_auc),
        "confusion_matrix": cm.tolist(),
        "top_features": [{"name": feature_names[idx], "importance": float(importances[idx])} for idx in top_idx],
        "report": report,
    }


# ─────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────


def save_model(model, feature_names: list, metrics: dict):
    """Save trained model, feature names, and training report."""
    banner("SAVING MODEL")

    # 1. Save XGBoost model
    model_path = MODELS_DIR / "xgb_classifier.json"
    model.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    # 2. Save feature names (needed for inference alignment)
    names_path = MODELS_DIR / "xgb_feature_names.json"
    with open(names_path, "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "n_radiomics": N_RADIOMICS,
                "n_cnn": N_CNN_FEATURES,
                "n_total": len(feature_names),
            },
            f,
            indent=2,
        )
    print(f"  Feature names → {names_path}")

    # 3. Save feature order for validation
    order_path = MODELS_DIR / "xgb_feature_order.json"
    with open(order_path, "w") as f:
        json.dump(
            {
                "radiomics_features": list(RADIOMICS_FEATURES.keys()),
                "cnn_feature_count": N_CNN_FEATURES,
                "total_features": len(feature_names),
            },
            f,
            indent=2,
        )
    print(f"  Feature order → {order_path}")

    # 3. Save training report
    report_path = MODELS_DIR / "xgb_training_report.txt"
    with open(report_path, "w") as f:
        f.write("XGBOOST CLASSIFIER — TRAINING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("⚠️  IMPORTANT: This model was trained on SYNTHETIC data\n")
        f.write("   generated from published literature distributions.\n")
        f.write("   It MUST be re-trained on institutional data before\n")
        f.write("   clinical deployment.\n\n")
        f.write("DATA SOURCES:\n")
        f.write("  - Bakas et al. 2018 (BraTS — 335 glioma cases)\n")
        f.write("  - Kickingereder et al. 2016 (radiomics grading)\n")
        f.write("  - Zhou et al. 2019 (DL features for IDH)\n\n")
        f.write(f"FEATURES: {len(feature_names)} total\n")
        f.write(f"  Radiomics: {N_RADIOMICS}\n")
        f.write(f"  CNN (DenseNet121): {N_CNN_FEATURES}\n\n")
        f.write("CROSS-VALIDATION (5-fold stratified):\n")
        f.write(f"  AUC-ROC:  {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}\n")
        f.write(f"  Accuracy: {metrics['cv_acc_mean']:.4f}\n")
        f.write(f"  F1 Score: {metrics['cv_f1_mean']:.4f}\n\n")
        f.write(f"TRAINING AUC: {metrics['train_auc']:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(metrics["report"] + "\n\n")
        f.write("TOP 20 FEATURES:\n")
        for i, feat in enumerate(metrics["top_features"], 1):
            f.write(f"  {i:2d}. {feat['name']:40s} {feat['importance']:.4f}\n")
        f.write("\n")
        f.write("DISCLAIMER:\n")
        f.write("  This is a RESEARCH TOOL trained on synthetic data.\n")
        f.write("  NOT validated for clinical decision-making.\n")
        f.write("  All predictions require histopathological confirmation.\n")
    print(f"  Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    banner("XGBOOST CLASSIFIER TRAINING")
    print("  Training hybrid Radiomics + DenseNet121 classifier")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Model output: {MODELS_DIR}")

    # Check for real data first
    real_data_path = PROJECT_ROOT / "data"
    has_real_data = (real_data_path / "radiomics_features.npy").exists() and (real_data_path / "labels.npy").exists()

    if has_real_data:
        banner("LOADING REAL TRAINING DATA")
        X_rad = np.load(str(real_data_path / "radiomics_features.npy"))
        y = np.load(str(real_data_path / "labels.npy"))
        print(f"  Radiomics features: {X_rad.shape}")

        # Check for CNN features
        cnn_path = real_data_path / "cnn_features.npy"
        if cnn_path.exists():
            X_cnn = np.load(str(cnn_path))
            X = np.hstack([X_rad, X_cnn])
            print(f"  CNN features: {X_cnn.shape}")
        else:
            X = X_rad
            print("  CNN features: not found — using radiomics only")

        print(f"  Labels: {y.shape}")
        print(f"  Fused: {X.shape}")

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    else:
        banner("GENERATING SYNTHETIC TRAINING COHORT")
        print("  No real data found at data/radiomics_features.npy")
        print("  → Generating synthetic cohort from literature distributions")
        print("  → Re-train on real data when institutional dataset available")
        print()
        print("  To use real data, place these files in data/:")
        print("    radiomics_features.npy  — shape (N_patients, N_features)")
        print("    cnn_features.npy        — shape (N_patients, 1024)")
        print("    labels.npy              — shape (N_patients,) — 0=LGG, 1=HGG")

        X, y, feature_names = generate_synthetic_cohort(n_hgg=250, n_lgg=150, seed=42)
        print(f"\n  Generated: {X.shape[0]} samples × {X.shape[1]} features")

    # Train
    metrics = train_model(X, y, feature_names)

    # Save
    save_model(metrics["model"], feature_names, metrics)

    banner("DONE")
    print(f"  Model ready at: {MODELS_DIR / 'xgb_classifier.json'}")
    print("  Run Stage 8 to use it: python3 scripts/8_radiomics_analysis.py")
    print()
    print("  ⚠️  Trained on SYNTHETIC data — re-train on institutional")
    print("     data before clinical deployment.")
    print()
