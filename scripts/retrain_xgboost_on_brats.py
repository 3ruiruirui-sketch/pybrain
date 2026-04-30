#!/usr/bin/env python3
"""
Retrain XGBoost on Real BraTS2021 Data
======================================
Professional workflow to retrain the radiomics classifier on REAL patient data
from the BraTS2021 dataset, replacing the synthetic training.

Supports all 1250 BraTS2021 cases — automatically selects cases with grade labels.

PHASES:
  1. Scan all BraTS2021 cases, discover which have grade labels
  2. Run pipeline on labeled cases (stages 1-8)
  3. Extract radiomics + CNN features from completed cases
  4. Retrain XGBoost with proper 5-fold stratified CV
  5. Evaluate, save, and generate validation report

REQUIREMENTS:
  - BraTS2021 survival_data.csv (WHO grade labels)
  - Download from: https://www.med.upenn.edu/cbica/brats2021/data.html
  - OR provide your own labels CSV (case_id, grade columns)

USAGE:
  python3 scripts/retrain_xgboost_on_brats.py
"""

import sys
import json
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
BRATS_DATA = PROJECT_ROOT / "data/datasets/BraTS2021/raw/BraTS2021_Training_Data"

# ─────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        confusion_matrix,
    )
    import xgboost as xgb
except ImportError as e:
    print(f"❌ Missing: {e}")
    print("Run: pip install numpy pandas scikit-learn xgboost")
    sys.exit(1)


def banner(t: str):
    print("\n" + "=" * 65)
    print(f"  {t}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────
# PHASE 1: Discover All BraTS2021 Cases and Match with Grade Labels
# ─────────────────────────────────────────────────────────────────────────


def discover_all_brats_cases() -> List[str]:
    """Find all BraTS2021 case directories."""
    if not BRATS_DATA.exists():
        print(f"❌ BraTS data not found at {BRATS_DATA}")
        sys.exit(1)

    cases = sorted([d.name for d in BRATS_DATA.iterdir() if d.is_dir() and d.name.startswith("BraTS2021_")])
    print(f"  Found {len(cases)} BraTS2021 cases")
    return cases


def find_survival_data(args) -> Optional[Path]:
    """Find survival_data.csv or labels CSV in multiple possible locations."""
    # First check if user provided a custom labels CSV
    if args.labels_csv:
        custom_path = Path(args.labels_csv)
        if custom_path.exists():
            print(f"  Using custom labels CSV: {custom_path}")
            return custom_path
        else:
            print(f"  ⚠️  Custom labels CSV not found: {custom_path}")
            return None

    # Otherwise search for survival_data.csv
    possible_paths = [
        BRATS_DATA / "survival_data.csv",
        BRATS_DATA.parent / "survival_data.csv",
        DATA_DIR / "survival_data.csv",
        DATA_DIR / "BraTS2021_survival_data.csv",
        Path("~/Downloads/BraTS2021_survival_data.csv").expanduser(),
        Path("~/Downloads/survival_data.csv").expanduser(),
    ]

    for p in possible_paths:
        if p.exists():
            print(f"  Found: {p}")
            return p

    return None


def load_survival_data(path: Path) -> pd.DataFrame:
    """Load and parse survival_data.csv."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"  Loaded: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def find_grade_column(df: pd.DataFrame) -> Optional[str]:
    """Find the WHO grade column in survival data."""
    grade_candidates = ["grade", "who_grade", "who_grade_or_histology", "histology", "tumor_grade", "mgmt", "idh"]
    for col in df.columns:
        col_lower = col.lower()
        if any(cand in col_lower for cand in grade_candidates):
            # Make sure it has grade-like values
            sample_vals = df[col].dropna().astype(str).str.upper().unique()[:10]
            print(f"  Grade candidate: '{col}' → values: {list(sample_vals)}")
            return col
    return None


def match_cases_with_labels(case_ids: List[str], df: pd.DataFrame, grade_col: str) -> dict:
    """
    Match BraTS cases to grade labels.

    Returns dict: {case_id: label}  where label = 0 (LGG, Grade I-II) or 1 (HGG, Grade III-IV)
    """
    # Normalize column names for ID matching
    id_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ["id", "name", "case", "subject"]):
            id_col = col
            break

    if id_col is None:
        print("  ⚠️  No ID column found in survival data")
        return {}

    print(f"\n  ID column: '{id_col}', Grade column: '{grade_col}'")

    # Build label map from survival data
    raw_label_map = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        grade_raw = str(row[grade_col]).strip().upper()

        # Parse grade values
        label = None
        # LGG = Grade I, II
        if grade_raw in [
            "I",
            "1",
            "II",
            "2",
            "GRADE I",
            "GRADE 1",
            "GRADE_1",
            "GRADE II",
            "GRADE 2",
            "GRADE_2",
            "LOW GRADE",
            "LGG",
            "ASTROCYTOMA",
            "ASTROCYTOMA I",
            "ASTROCYTOMA II",
            "OLIGODENDROGLIOMA",
            "OLIGODENDROGLIOMA I",
            "OLIGODENDROGLIOMA II",
            "PA",
            "DNET",
        ]:
            label = 0
        # HGG = Grade III, IV
        elif grade_raw in [
            "III",
            "3",
            "GRADE III",
            "GRADE 3",
            "GRADE_3",
            "IV",
            "4",
            "GRADE IV",
            "GRADE 4",
            "GRADE_4",
            "GLIOBLASTOMA",
            "GBM",
            "AA",
            "AO",
            "AOA",
            "GB",
            "ANAPLASTIC ASTROCYTOMA",
            "ANAPLASTIC OLIGODENDROGLIOMA",
        ]:
            label = 1

        if label is not None:
            raw_label_map[raw_id] = label

    print(f"  Labeled cases in survival CSV: {len(raw_label_map)}")

    # Match our case IDs to the survival data
    label_map = {}
    unmapped = []
    grade_dist = {0: 0, 1: 0}

    for case_id in case_ids:
        matched = False

        # Try exact match
        if case_id in raw_label_map:
            label_map[case_id] = raw_label_map[case_id]
            grade_dist[raw_label_map[case_id]] += 1
            matched = True
        else:
            # Try BraTS21_XXX vs BraTS2021_XXX format
            alt1 = case_id.replace("BraTS2021_", "BraTS21_")
            alt2 = case_id.replace("BraTS2021_", "")
            case_id.replace("BraTS2021_", "BraTS21_00")

            if alt1 in raw_label_map:
                label_map[case_id] = raw_label_map[alt1]
                grade_dist[raw_label_map[alt1]] += 1
                matched = True
            elif alt2 in raw_label_map:
                label_map[case_id] = raw_label_map[alt2]
                grade_dist[raw_label_map[alt2]] += 1
                matched = True

        if not matched:
            unmapped.append(case_id)

    print(f"\n  Matched to grade labels: {len(label_map)} cases")
    print(f"    LGG (Grade I-II):  {grade_dist[0]}")
    print(f"    HGG (Grade III-IV): {grade_dist[1]}")
    print(f"    Unmapped: {len(unmapped)} cases (no grade in survival CSV)")

    return label_map


def filter_labeled_cases(case_ids: List[str], label_map: dict) -> tuple:
    """Return only cases that have grade labels."""
    labeled = [(c, label_map[c]) for c in case_ids if c in label_map]
    unlabeled = [c for c in case_ids if c not in label_map]
    return labeled, unlabeled


# ─────────────────────────────────────────────────────────────────────────
# PHASE 2: Process Labeled Cases Through Pipeline
# ─────────────────────────────────────────────────────────────────────────


def ensure_pipeline_output_for_case(case_id: str, session_path: Path) -> bool:
    """
    Ensure all pipeline stages (1-8) have been run for a case.

    Checks if radiomics_features.json exists (stage 8 complete).
    Returns True if case is fully processed.
    """
    case_output = RESULTS_DIR / "validation_runs" / case_id
    radiomics_file = case_output / "radiomics_features.json"
    return radiomics_file.exists()


def process_single_case(case_id: str) -> bool:
    """
    Run full pipeline (stages 1-8) for a single case.

    Sets up session.json then runs the pipeline.
    Returns True if successful.
    """
    import subprocess

    case_dir = RESULTS_DIR / "validation_runs" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    session_file = case_dir / "session.json"

    # Create minimal session.json for this case
    session_data = {
        "case_id": case_id,
        "patient": {"name": case_id, "age": "0", "sex": "M"},
        "output_dir": str(case_dir),
        "monai_dir": str(case_dir),
        "results_dir": str(RESULTS_DIR / "validation_runs"),
        "extra_dir": str(BRATS_DATA / case_id),
        "stages": {
            "stage_0_validate": False,
            "stage_1_dicom": True,
            "stage_1b_prep": True,
            "stage_1_register": True,
            "stage_2_ct": True,
            "stage_3_segment": True,
            "stage_4_register": True,
            "stage_5_qc": True,
            "stage_6_location": True,
            "stage_7_morphology": True,
            "stage_8_radiomics": True,
            "use_hdbet": True,
        },
    }

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    # Run pipeline
    env = {
        **dict(__import__("os").environ),
        "PYBRAIN_SESSION": str(session_file),
    }

    try:
        result = subprocess.run(
            [sys.executable, "run_pipeline.py", "--session", str(session_file)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per case
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Timeout for {case_id}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def process_all_labeled_cases(
    labeled_cases: List[tuple], skip_existing: bool = True, max_cases: int = None, balance_lgg: int = 0
) -> tuple:
    """
    Process labeled cases through stages 1-8.

    Args:
        labeled_cases: List of (case_id, grade) tuples
        skip_existing: Skip cases with existing radiomics
        max_cases: Maximum total cases to process
        balance_lgg: If > 0, take equal LGG/HGG samples (balance_lgg each)

    Returns: (success_cases, failed_cases, skipped_cases)
    """
    banner("PHASE 2: Processing Labeled Cases Through Pipeline")

    # Balanced sampling: equal LGG and HGG
    if balance_lgg > 0:
        lgg_cases = [(c, g) for c, g in labeled_cases if g == 0]
        hgg_cases = [(c, g) for c, g in labeled_cases if g == 1]

        # Take up to balance_lgg from each class, prioritizing those needing processing
        def needs_processing(case_id):
            rf = RESULTS_DIR / "validation_runs" / case_id / "radiomics_features.json"
            if not rf.exists():
                return True
            try:
                import json as j

                with open(rf) as f:
                    d = j.load(f)
                return not (isinstance(d.get("all_features"), dict) and len(d.get("all_features", {})) >= 40)
            except:
                return True

        lgg_needed = [x for x in lgg_cases if needs_processing(x[0])]
        hgg_needed = [x for x in hgg_cases if needs_processing(x[0])]
        lgg_done = [x for x in lgg_cases if not needs_processing(x[0])]
        hgg_done = [x for x in hgg_cases if not needs_processing(x[0])]

        # Prioritize: needed LGG > needed HGG > done LGG > done HGG
        balanced = (
            lgg_needed[:balance_lgg]
            + hgg_needed[:balance_lgg]
            + lgg_done[: max(0, balance_lgg - len(lgg_needed))]
            + hgg_done[: max(0, balance_lgg - len(hgg_needed))]
        )
        labeled_cases = balanced[: max_cases or 99999]
        print(
            f"  Balanced sampling: {len(labeled_cases)} cases "
            f"({sum(g == 0 for _, g in labeled_cases)} LGG, {sum(g == 1 for _, g in labeled_cases)} HGG)"
        )
    elif max_cases:
        labeled_cases = labeled_cases[:max_cases]
        print(f"  Processing first {max_cases} labeled cases (per user request)")

    total = len(labeled_cases)
    success = []
    failed = []
    skipped = []

    print(f"  Total labeled cases to process: {total}")
    print(f"  Pipeline output dir: {RESULTS_DIR / 'validation_runs'}")
    print(f"  Skip existing: {skip_existing}\n")

    for i, (case_id, grade) in enumerate(labeled_cases, 1):
        output_dir = RESULTS_DIR / "validation_runs" / case_id
        already_done = (output_dir / "radiomics_features.json").exists()

        print(f"[{i}/{total}] {case_id} (Grade {'HGG' if grade else 'LGG'})...", end=" ")

        if skip_existing and already_done:
            print("✅ (already done)")
            success.append(case_id)
            skipped.append(case_id)
            continue

        ok = process_single_case(case_id)
        if ok:
            print("✅")
            success.append(case_id)
        else:
            print("❌")
            failed.append(case_id)

        # Rate limit to avoid overwhelming the system
        if i % 10 == 0:
            time.sleep(2)

    print(f"\n  Processed: {len(success)} | Failed: {len(failed)} | Skipped: {len(skipped)}")
    if failed:
        print(f"  Failed cases: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return success, failed, skipped


# ─────────────────────────────────────────────────────────────────────────
# PHASE 3: Extract Features from Processed Cases
# ─────────────────────────────────────────────────────────────────────────


def extract_features_for_cases(case_ids: List[str]) -> tuple:
    """
    Extract radiomics features from processed cases.

    Returns: (X_features, feature_names, cases_with_features)
    """
    banner("PHASE 3: Extracting Radiomics Features")

    all_features = []
    success_cases = []
    failed = []

    for i, case_id in enumerate(case_ids, 1):
        case_dir = RESULTS_DIR / "validation_runs" / case_id
        radiomics_file = case_dir / "radiomics_features.json"

        print(f"[{i}/{len(case_ids)}] {case_id}...", end=" ")

        if not radiomics_file.exists():
            print("⚠️  (no radiomics)")
            failed.append(case_id)
            continue

        try:
            with open(radiomics_file) as f:
                raw_radio = json.load(f)

            # Radiomics: actual scalar features live in the "all_features" sub-dict
            # Top-level keys like "shape", "classification" are metadata dicts
            features = {}
            if isinstance(raw_radio.get("all_features"), dict):
                features.update(raw_radio["all_features"])
            else:
                # Fallback: use top-level scalars (older format)
                for k, v in raw_radio.items():
                    if isinstance(v, (int, float)):
                        features[k] = v

            # Add morphology if available
            morph_file = case_dir / "morphology_results.json"
            if morph_file.exists():
                with open(morph_file) as f:
                    morph = json.load(f)
                for k, v in morph.items():
                    if isinstance(v, (int, float)):
                        features[f"morph_{k}"] = v
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float)):
                                features[f"morph_{sub_k}"] = sub_v

            # Add location if available
            loc_file = case_dir / "tumour_location.json"
            if loc_file.exists():
                with open(loc_file) as f:
                    loc = json.load(f)
                for k, v in loc.items():
                    if isinstance(v, (int, float)):
                        features[f"loc_{k}"] = v
                    elif isinstance(v, (list, tuple)) and len(v) == 3:
                        # Unpack [x,y,z] centre into separate features
                        for axis, val in zip(["x", "y", "z"], v):
                            if isinstance(val, (int, float)):
                                features[f"loc_{k}_{axis}"] = float(val)

            all_features.append(features)
            success_cases.append(case_id)
            print("✅")

        except Exception as e:
            print(f"⚠️  (error: {e})")
            failed.append(case_id)

    if not all_features:
        print("❌ No features extracted.")
        sys.exit(1)

    # Align all features to common feature space
    all_keys = set()
    for f in all_features:
        all_keys.update(f.keys())

    feature_names = sorted(all_keys)
    n_features = len(feature_names)

    X = np.zeros((len(all_features), n_features), dtype=np.float32)
    for i, f in enumerate(all_features):
        for j, key in enumerate(feature_names):
            val = f.get(key, 0.0)
            # Skip dict values — only use scalar floats
            if isinstance(val, (int, float)):
                X[i, j] = float(val)
            elif isinstance(val, np.floating):
                X[i, j] = float(val)
            elif isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                # Unpack list/tuple like [x,y,z] into scalar features
                X[i, j] = float(val[0]) if len(val) == 1 else 0.0
            # else leave as 0.0

    print(f"\n  Extracted: {len(success_cases)} cases × {n_features} features")
    if failed:
        print(f"  Failed: {len(failed)}")

    return X, feature_names, success_cases


# ─────────────────────────────────────────────────────────────────────────
# PHASE 4: Retrain XGBoost with Proper Cross-Validation
# ─────────────────────────────────────────────────────────────────────────


def retrain_xgboost(X, y, feature_names, case_ids):
    """
    Retrain XGBoost with 5-fold stratified CV on real data.
    """
    banner("PHASE 4: Retraining XGBoost on Real BraTS2021 Data")

    n_samples, n_features = X.shape
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos

    print("\n  Dataset:")
    print(f"    Cases: {n_samples}")
    print(f"    LGG (Grade I-II): {(y == 0).sum()}")
    print(f"    HGG (Grade III-IV): {(n_pos)}")
    print(f"    Features: {n_features}")
    print(f"    Class ratio: {n_pos / n_samples * 100:.0f}% HGG")

    if n_samples < 50:
        print(f"\n  ⚠️  Small dataset ({n_samples} cases) — results will be noisy")
        print("     Target: n≥200 for stable model, n≥500 for clinical use")

    # Conservative params for small-to-medium real data
    params = {
        "n_estimators": 150 if n_samples >= 200 else 80,
        "max_depth": 4 if n_samples >= 200 else 3,
        "learning_rate": 0.03,
        "subsample": 0.8 if n_samples >= 200 else 0.7,
        "colsample_bytree": 0.6 if n_samples >= 200 else 0.5,
        "min_child_weight": 3 if n_samples >= 200 else 5,
        "gamma": 0.1 if n_samples >= 200 else 0.3,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "scale_pos_weight": n_neg / max(n_pos, 1),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "verbosity": 0,
    }

    print("\n  XGBoost parameters:")
    for k, v in params.items():
        print(f"    {k:20s}: {v}")

    # 5-fold stratified CV
    banner("5-FOLD STRATIFIED CROSS-VALIDATION")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = xgb.XGBClassifier(**params)

    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)
    y_prob_cv = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    cv_auc = roc_auc_score(y, y_prob_cv)
    cv_acc = (y_pred_cv == y).mean()
    cm = confusion_matrix(y, y_pred_cv)

    print("\n  Cross-Validated Results:")
    print(f"    AUC-ROC:  {cv_auc:.4f}")
    print(f"    Accuracy: {cv_acc:.4f}")

    print("\n  Confusion Matrix:")
    print("              Pred LGG  Pred HGG")
    print(f"    True LGG     {cm[0, 0]:4d}     {cm[0, 1]:4d}")
    print(f"    True HGG     {cm[1, 0]:4d}     {cm[1, 1]:4d}")

    print("\n  Classification Report:")
    print(classification_report(y, y_pred_cv, target_names=["LGG", "HGG"]))

    # Per-fold
    print("\n  Per-Fold Metrics:")
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        clf_fold = xgb.XGBClassifier(**params)
        clf_fold.fit(X[train_idx], y[train_idx])
        y_prob_fold = clf_fold.predict_proba(X[val_idx])[:, 1]
        fold_auc = roc_auc_score(y[val_idx], y_prob_fold)
        fold_acc = (clf_fold.predict(X[val_idx]) == y[val_idx]).mean()
        fold_metrics.append({"fold": fold, "auc": fold_auc, "acc": fold_acc})
        print(f"    Fold {fold}: AUC={fold_auc:.4f}  Acc={fold_acc:.4f}")

    mean_auc = np.mean([m["auc"] for m in fold_metrics])
    std_auc = np.std([m["auc"] for m in fold_metrics])
    print(f"\n  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Feature importance
    clf.fit(X, y)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]

    banner("TOP 20 FEATURE IMPORTANCE")
    print(f"  {'Rank':>4s}  {'Feature':45s}  {'Importance':>10s}")
    print(f"  {'─' * 4}  {'─' * 45}  {'─' * 10}")
    for rank, idx in enumerate(top_idx, 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"  {rank:4d}  {fname:45s}  {importances[idx]:10.4f}")

    return {
        "model": clf,
        "cv_auc": cv_auc,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "cv_acc": cv_acc,
        "confusion_matrix": cm.tolist(),
        "fold_metrics": fold_metrics,
        "feature_importance": [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"feat_{i}",
                "importance": float(importances[i]),
            }
            for i in np.argsort(importances)[::-1]
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
# PHASE 5: Save Model + Generate Report
# ─────────────────────────────────────────────────────────────────────────


def save_model_and_report(
    model, feature_names, metrics, X, y, all_labeled_cases, processed_cases, success_cases, failed_cases
):
    """Save retrained model with full documentation."""
    banner("PHASE 5: Saving Model and Report")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. XGBoost model
    model_path = MODELS_DIR / "xgb_classifier.json"
    model.save_model(str(model_path))
    print(f"  Model: {model_path}")

    # 2. Feature names
    names_path = MODELS_DIR / "xgb_feature_names.json"
    with open(names_path, "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "trained_on": "real_barts2021",
                "n_cases_trained": len(y),
                "n_cases_total_labeled": len(all_labeled_cases),
                "n_cases_processed": len(processed_cases),
            },
            f,
            indent=2,
        )

    # 3. Feature order
    order_path = MODELS_DIR / "xgb_feature_order.json"
    with open(order_path, "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "total_features": len(feature_names),
                "data_source": "BraTS2021 pipeline (stages 6-8)",
            },
            f,
            indent=2,
        )

    # 4. JSON metrics
    json_report = {
        "timestamp": timestamp,
        "data_source": "BraTS2021_training_data",
        "n_total_cases_available": 1251,
        "n_cases_with_grade_labels": len(all_labeled_cases),
        "n_cases_processed": len(processed_cases),
        "n_cases_used_in_training": len(y),
        "n_features": len(feature_names),
        "class_distribution": {
            "LGG": int((y == 0).sum()),
            "HGG": int((y == 1).sum()),
        },
        "cv_auc": float(metrics["cv_auc"]),
        "cv_acc": float(metrics["cv_acc"]),
        "mean_auc": float(metrics["mean_auc"]),
        "std_auc": float(metrics["std_auc"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "fold_metrics": metrics["fold_metrics"],
        "top_20_features": metrics["feature_importance"][:20],
        "cases_used": sorted(success_cases),
        "cases_failed": failed_cases,
        "model_parameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
        },
        "validation_status": "research_only",
        "clinical_status": "NOT VALIDATED — external institutional validation required",
    }
    json_path = MODELS_DIR / "xgb_retrain_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"  JSON report: {json_path}")

    # 5. TXT Report
    report_path = MODELS_DIR / "xgb_training_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST CLASSIFIER — BRATS2021 REAL DATA RETRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Data Source: BraTS2021 Training Data ({len(all_labeled_cases)} labeled cases)\n")
        f.write(f"Cases processed: {len(processed_cases)}/{len(all_labeled_cases)}\n")
        f.write(f"Cases used in training: {len(y)}\n")
        f.write(f"Features: {len(feature_names)}\n\n")

        f.write("CLASS DISTRIBUTION:\n")
        f.write(f"  LGG (Grade I-II):  {(y == 0).sum()}\n")
        f.write(f"  HGG (Grade III-IV): {(y == 1).sum()}\n\n")

        f.write("CROSS-VALIDATION (5-fold Stratified):\n")
        f.write(f"  AUC-ROC:  {metrics['cv_auc']:.4f}\n")
        f.write(f"  Accuracy: {metrics['cv_acc']:.4f}\n")
        f.write(f"  Mean AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}\n\n")

        f.write("PER-FOLD METRICS:\n")
        for m in metrics["fold_metrics"]:
            f.write(f"  Fold {m['fold']}: AUC={m['auc']:.4f}  Acc={m['acc']:.4f}\n")
        f.write("\n")

        f.write("CONFUSION MATRIX:\n")
        cm = metrics["confusion_matrix"]
        f.write("               Pred LGG  Pred HGG\n")
        f.write(f"  True LGG      {cm[0][0]:4d}     {cm[0][1]:4d}\n")
        f.write(f"  True HGG      {cm[1][0]:4d}     {cm[1][1]:4d}\n\n")

        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y, model.predict(X), target_names=["LGG", "HGG"]) + "\n")

        f.write("\nTOP 30 FEATURES:\n")
        for i, feat in enumerate(metrics["feature_importance"][:30], 1):
            f.write(f"  {i:2d}. {feat['feature']:45s} {feat['importance']:.4f}\n")

        f.write("\nCASES USED:\n")
        for c in sorted(success_cases):
            f.write(f"  {c}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("STATUS & REQUIRED NEXT STEPS:\n\n")
        f.write("  Current status: RESEARCH MODEL — NOT FOR CLINICAL USE\n\n")
        f.write("  Completed:\n")
        f.write("    ✅ Retrained on REAL BraTS2021 patient data\n")
        f.write("    ✅ 5-fold stratified cross-validation\n")
        f.write("    ✅ Feature importance analysis\n\n")
        f.write("  Required before clinical deployment:\n")
        f.write("    ⏳ External validation on held-out institutional data\n")
        f.write("    ⏳ Minimum 200+ cases for stable model\n")
        f.write("    ⏳ IRB approval\n")
        f.write("    ⏳ Prospective validation study\n\n")
        f.write(f"  Model CV AUC: {metrics['cv_auc']:.4f} (expected range: 0.75-0.90 for real data)\n")
        f.write("=" * 70 + "\n")
    print(f"  TXT report: {report_path}")

    return model_path, json_report


# ─────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW
# ─────────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrain XGBoost on BraTS2021 real data")
    parser.add_argument("--max-cases", type=int, default=None, help="Max labeled cases to process (default: all)")
    parser.add_argument(
        "--skip-existing", action="store_true", default=True, help="Skip cases with existing pipeline output"
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false", help="Re-process all labeled cases"
    )
    parser.add_argument(
        "--labels-csv", type=str, default=None, help="Path to custom labels CSV (case_id, grade columns)"
    )
    parser.add_argument("--balance-lgg", type=int, default=0, help="Take N LGG + N HGG cases (balanced sampling)")
    args = parser.parse_args()

    banner("XGBOOST RETRAINING ON REAL BRATS2021 DATA")
    print("  Professional workflow — 1250 BraTS2021 cases\n")

    # ── Step 1: Discover cases ─────────────────────────────────────────
    all_cases = discover_all_brats_cases()

    # ── Step 2: Find survival data ──────────────────────────────────────
    survival_path = find_survival_data(args)

    if survival_path is None:
        print("\n❌ Labels CSV not found (survival_data.csv or custom --labels-csv).")
        print("\n  OPTIONS TO GET LABELS:")
        print("  ─────────────────────────────────────────────────────")
        print("  OPTION A — Download BraTS2021 survival data:")
        print("    1. Register at: https://www.med.upenn.edu/cbica/brats2021/data.html")
        print("    2. Download survival_data.csv")
        print("    3. Place in: /Users/ssoares/Downloads/PY-BRAIN/data/datasets/BraTS2021/")
        print("\n  OPTION B — Provide your own labels CSV:")
        print("    Columns: case_id, grade")
        print("    grade values: I, II, III, IV (or 1,2,3,4)")
        print("    Run: python3 scripts/retrain_xgboost_on_brats.py --labels-csv /path/to/labels.csv")
        print("\n  OPTION C — Create labels from BraTS segmentation:")
        print("    Use BraTS2021_XXXXX_seg.nii.gz to infer grade:")
        print("    - Volume > 30cc + necrotic core → likely HGG (Grade IV)")
        print("    - Volume < 15cc + no necrosis → likely LGG (Grade II)")
        print("    ⚠️  This is approximate and less reliable\n")
        sys.exit(1)

    survival_df = load_survival_data(survival_path)

    # ── Step 3: Find grade column ────────────────────────────────────────
    grade_col = find_grade_column(survival_df)
    if grade_col is None:
        print("\n❌ Cannot find grade column in survival CSV.")
        print(f"  Available columns: {list(survival_df.columns)}")
        sys.exit(1)

    # ── Step 4: Match cases to labels ───────────────────────────────────
    label_map = match_cases_with_labels(all_cases, survival_df, grade_col)
    labeled_cases, unlabeled = filter_labeled_cases(all_cases, label_map)

    if not labeled_cases:
        print("\n❌ No cases matched to grade labels.")
        print("   Check the ID column format in your survival CSV.")
        sys.exit(1)

    print("\n  Summary:")
    print(f"    Total BraTS2021 cases: {len(all_cases)}")
    print(f"    Cases with grade labels: {len(labeled_cases)}")
    print(f"    Cases without labels: {len(unlabeled)}")

    # ── Step 5: Process labeled cases ───────────────────────────────────
    success_cases, failed_cases, skipped_cases = process_all_labeled_cases(
        labeled_cases,
        skip_existing=args.skip_existing,
        max_cases=args.max_cases,
        balance_lgg=args.balance_lgg,
    )

    if not success_cases:
        print("\n❌ No cases successfully processed.")
        sys.exit(1)

    # ── Step 6: Extract features ────────────────────────────────────────
    X, feature_names, cases_with_features = extract_features_for_cases(success_cases)

    # Build y array aligned with feature extraction order
    y = np.array([label_map[c] for c in cases_with_features], dtype=np.int32)

    # ── Step 7: Retrain ─────────────────────────────────────────────────
    metrics = retrain_xgboost(X, y, feature_names, cases_with_features)

    # ── Step 8: Save ────────────────────────────────────────────────────
    model_path, json_report = save_model_and_report(
        model=metrics["model"],
        feature_names=feature_names,
        metrics=metrics,
        X=X,
        y=y,
        all_labeled_cases=labeled_cases,
        processed_cases=success_cases,
        success_cases=cases_with_features,
        failed_cases=failed_cases,
    )

    # ── Final Summary ───────────────────────────────────────────────────
    banner("RETRAINING COMPLETE")
    print(f"\n  ✅ XGBoost retrained on {len(y)} REAL BraTS2021 cases")
    print(f"\n  Model: {model_path}")
    print(f"  Class distribution: LGG={(y == 0).sum()}, HGG={(y == 1).sum()}")
    print(f"  CV AUC: {metrics['cv_auc']:.4f}")
    print(f"  Mean AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
    print("\n  ⚠️  REQUIRED NEXT STEPS:")
    print("     1. External validation on held-out institutional data")
    print(f"     2. Collect more cases (current: {len(y)}, target: n≥200)")
    print("     3. IRB approval before any clinical deployment")
    print("\n  To use: python3 scripts/8_radiomics_analysis.py")


if __name__ == "__main__":
    main()
