#!/usr/bin/env python3
"""
predict_grade.py — Inference for LGG/HGG Grade Prediction
===========================================================
Takes a BraTS2021 case_id and returns:
  1. XGBoost prediction (radiomics features → grade)
  2. Ensemble prediction (radiomics + DualHead bottleneck)
  3. DualHead MLP prediction (standalone encoder features)

Usage:
  python3 scripts/predict_grade.py --case-id BraTS2021_00074
  python3 scripts/predict_grade.py --case-id BraTS2021_00074 --verbose

  # Run on all cases with features and print per-case accuracy
  python3 scripts/predict_grade.py --all-cases

Author: PY-BRAIN pipeline — Brain Tumor MRI Analysis
"""

import sys, json, argparse, os

os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_radiomics_features(case_id, results_dir):
    """Load and flatten radiomics + morphology + location features."""
    case_dir = results_dir / case_id
    features = {}

    # Radiomics
    rf = case_dir / "radiomics_features.json"
    if rf.exists():
        with open(rf) as f:
            d = json.load(f)
        if isinstance(d.get("all_features"), dict):
            features.update(d["all_features"])

    # Morphology
    mf = case_dir / "morphology_results.json"
    if mf.exists():
        with open(mf) as f:
            m = json.load(f)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                features[f"morph_{k}"] = v
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)):
                        features[f"morph_{sk}"] = sv

    # Location
    lf = case_dir / "tumour_location.json"
    if lf.exists():
        with open(lf) as f:
            l = json.load(f)
        for k, v in l.items():
            if isinstance(v, (int, float)):
                features[f"loc_{k}"] = v

    return features


def load_dualhead_bottleneck(case_id, results_dir):
    """Load cached DualHeadSegResNet encoder bottleneck features."""
    p = results_dir / case_id / "dualhead_bottleneck.npy"
    if p.exists():
        return np.load(p)
    return None


def load_feature_matrix(case_ids, feature_names, results_dir, include_dualhead=True):
    """Build feature matrix for a list of case IDs."""
    X_radio = []
    X_dh = []
    for cid in case_ids:
        feats = load_radiomics_features(cid, results_dir)
        row = np.array([float(feats.get(k, 0.0)) for k in feature_names], dtype=np.float32)
        X_radio.append(row)

        if include_dualhead:
            dh = load_dualhead_bottleneck(cid, results_dir)
            X_dh.append(dh if dh is not None else np.zeros(128, dtype=np.float32))

    X = np.array(X_radio, dtype=np.float32)
    if include_dualhead:
        X_dh = np.array(X_dh, dtype=np.float32)
        return X, X_dh
    return X


def load_xgb_model(model_path, feature_names_path):
    """Load XGBoost model and feature names."""
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(feature_names_path) as f:
        meta = json.load(f)
    return model, meta.get("feature_names", [])


def load_dualhead_mlp(fold_dir, input_dim=128):
    """Load all 5 DualHead MLP fold models for ensemble prediction."""
    import torch.nn as nn

    class ClassifierMLP(nn.Module):
        def __init__(self, input_dim=128, hidden=64, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden // 2, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    models = []
    for i in range(1, 6):
        m = ClassifierMLP(input_dim=input_dim)
        state = torch.load(fold_dir / f"fold_{i}.pt", map_location="cpu", weights_only=True)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


def predict_single(model, feature_names, case_id, results_dir, has_dualhead=False):
    """Run prediction for a single case."""
    feats = load_radiomics_features(case_id, results_dir)
    set(feature_names)

    X = np.zeros((1, len(feature_names)), dtype=np.float32)
    for j, k in enumerate(feature_names):
        X[0, j] = float(feats.get(k, 0.0))

    prob = model.predict_proba(X)[0]
    return {"p_lgg": float(prob[0]), "p_hgg": float(prob[1])}


def predict_ensemble(model, feature_names, case_id, results_dir, fold_models, all_keys):
    """Ensemble: XGBoost (radiomics) + DualHead MLP (bottleneck)."""
    import torch.nn.functional as F

    feats = load_radiomics_features(case_id, results_dir)
    X = np.zeros((1, len(feature_names)), dtype=np.float32)
    for j, k in enumerate(feature_names):
        X[0, j] = float(feats.get(k, 0.0))

    # XGBoost prediction
    xgb_prob = model.predict_proba(X)[0]

    # DualHead MLP ensemble
    dh = load_dualhead_bottleneck(case_id, results_dir)
    if dh is None:
        return {
            "xgb_p_lgg": float(xgb_prob[0]),
            "xgb_p_hgg": float(xgb_prob[1]),
            "dh_p_lgg": None,
            "dh_p_hgg": None,
            "ensemble_p_hgg": None,
        }

    dh_t = torch.from_numpy(dh).float().unsqueeze(0)
    dh_probs = []
    with torch.no_grad():
        for m in fold_models:
            logits = m(dh_t)
            probs = F.softmax(logits, dim=1)[0].numpy()
            dh_probs.append(probs)
    dh_prob_avg = np.mean(dh_probs, axis=0)

    # Weighted ensemble: 70% XGBoost + 30% DualHead
    ens_prob = 0.7 * xgb_prob + 0.3 * dh_prob_avg

    return {
        "xgb_p_lgg": float(xgb_prob[0]),
        "xgb_p_hgg": float(xgb_prob[1]),
        "dh_p_lgg": float(dh_prob_avg[0]),
        "dh_p_hgg": float(dh_prob_avg[1]),
        "ensemble_p_lgg": float(ens_prob[0]),
        "ensemble_p_hgg": float(ens_prob[1]),
    }


def predict_all():
    """Run on all cases that have radiomics features and evaluate."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

    results_dir = PROJECT_ROOT / "results" / "validation_runs"
    labels_df = pd.read_csv(PROJECT_ROOT / "data" / "auto_generated_labels.csv")
    labels_df["grade_bin"] = labels_df["grade"].apply(
        lambda g: 0 if str(g) in ["I", "II", "2"] or "II" in str(g) else 1
    )

    # Load XGBoost model
    model, feature_names = load_xgb_model(
        PROJECT_ROOT / "models" / "xgb_classifier.json", PROJECT_ROOT / "models" / "xgb_feature_names.json"
    )

    # Find cases with radiomics
    cases = []
    for _, row in labels_df.iterrows():
        c = row["case_id"]
        if (results_dir / c / "radiomics_features.json").exists():
            cases.append(c)

    results = []
    for cid in sorted(cases):
        row = labels_df[labels_df.case_id == cid]
        true_label = row["grade_bin"].values[0]
        true_grade = str(row["grade"].values[0])

        try:
            pred = predict_single(model, feature_names, cid, results_dir)
            pred_class = 1 if pred["p_hgg"] > 0.5 else 0
            correct = int(pred_class == true_label)
            results.append(
                {
                    "case_id": cid,
                    "true_grade": true_grade,
                    "true_label": int(true_label),
                    "p_hgg": float(pred["p_hgg"]),
                    "predicted": int(pred_class),
                    "correct": bool(correct),
                }
            )
        except Exception:
            pass

    y_true = [int(r["true_label"]) for r in results]
    y_prob = [float(r["p_hgg"]) for r in results]
    y_pred = [int(r["predicted"]) for r in results]

    auc = roc_auc_score(y_true, y_prob)
    acc = sum(1 for r in results if r["correct"]) / len(results)
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 65)
    print("  XGBOOST GRADE PREDICTION — ALL VALIDATION CASES")
    print("=" * 65)
    print(f"  Cases evaluated: {len(results)}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print("\n  Confusion Matrix:")
    print("                Pred LGG  Pred HGG")
    print(f"    True LGG     {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"    True HGG     {cm[1][0]:4d}     {cm[1][1]:4d}")
    print()
    print(classification_report(y_true, y_pred, target_names=["LGG", "HGG"]))

    # Per-case breakdown
    print("  DETAILED RESULTS (first 30):")
    print(f"  {'Case':<20} {'True':>6} {'p(HGG)':>8} {'Pred':>6} {'OK':>4}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 4}")
    for r in results[:30]:
        print(
            f"  {r['case_id']:<20} {r['true_grade']:>6} {r['p_hgg']:>8.3f} {'HGG' if r['predicted'] else 'LGG':>6} {'✓' if r['correct'] else '✗':>4}"
        )

    # Save full results
    out = PROJECT_ROOT / "results" / "prediction_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Predict LGG/HGG grade from BraTS case")
    parser.add_argument("--case-id", type=str, help="BraTS case ID (e.g. BraTS2021_00074)")
    parser.add_argument("--all-cases", action="store_true", help="Run on all cases with features")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-feature breakdown")
    parser.add_argument("--ensemble", action="store_true", help="Also run ensemble (XGBoost + DualHead MLP)")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / "validation_runs"

    if args.all_cases:
        predict_all()
        return

    if not args.case_id:
        print("Error: --case-id required (or use --all-cases)")
        print("Example: python3 scripts/predict_grade.py --case-id BraTS2021_00074")
        sys.exit(1)

    case_id = args.case_id

    # Check case exists
    if not (results_dir / case_id).exists():
        print(f"Error: Case {case_id} not found in {results_dir}")
        sys.exit(1)

    # Load models
    print(f"\n{'=' * 60}")
    print(f"  GRADE PREDICTION: {case_id}")
    print(f"{'=' * 60}")

    # XGBoost (radiomics)
    print("\n[1] XGBoost (radiomics + morphology + location)")
    try:
        model, feature_names = load_xgb_model(
            PROJECT_ROOT / "models" / "xgb_classifier.json", PROJECT_ROOT / "models" / "xgb_feature_names.json"
        )
        pred = predict_single(model, feature_names, case_id, results_dir)
        print(f"    p(LGG) = {pred['p_lgg']:.3f}")
        print(f"    p(HGG) = {pred['p_hgg']:.3f}")
        print(f"    → Predicted: {'HGG' if pred['p_hgg'] > 0.5 else 'LGG'} (p>0.5 threshold)")
    except Exception as e:
        print(f"    XGBoost prediction failed: {e}")

    # Ensemble (XGBoost + DualHead MLP)
    if args.ensemble:
        print("\n[2] Ensemble (XGBoost 70% + DualHead MLP 30%)")
        try:
            fold_models = load_dualhead_mlp(PROJECT_ROOT / "models" / "dualhead_classifier_folds")
            # Get feature names from ensemble metrics
            with open(PROJECT_ROOT / "models" / "ensemble_metrics.json") as f:
                json.load(f)
            # Use the same feature names from xgb_feature_names.json
            _, feature_names = load_xgb_model(
                PROJECT_ROOT / "models" / "xgb_classifier.json", PROJECT_ROOT / "models" / "xgb_feature_names.json"
            )
            pred = predict_ensemble(model, feature_names, case_id, results_dir, fold_models, feature_names)
            if pred["ensemble_p_hgg"] is not None:
                print(f"    XGBoost: p(HGG)={pred['xgb_p_hgg']:.3f}")
                print(f"    DualHead MLP: p(HGG)={pred['dh_p_hgg']:.3f}")
                print(f"    Ensemble: p(HGG)={pred['ensemble_p_hgg']:.3f}")
                print(f"    → Predicted: {'HGG' if pred['ensemble_p_hgg'] > 0.5 else 'LGG'}")
            else:
                print("    DualHead features not available for this case")
                print(f"    XGBoost alone: p(HGG)={pred['xgb_p_hgg']:.3f}")
        except Exception as e:
            print(f"    Ensemble prediction failed: {e}")
            import traceback

            traceback.print_exc()

    # Check what files are available for this case
    print("\n[INFO] Available pipeline outputs:")
    case_dir = results_dir / case_id
    outputs = {
        "radiomics_features.json": "✅" if (case_dir / "radiomics_features.json").exists() else "❌",
        "dualhead_bottleneck.npy": "✅" if (case_dir / "dualhead_bottleneck.npy").exists() else "❌",
        "cnn_deep_features.npy": "✅" if (case_dir / "cnn_deep_features.npy").exists() else "❌",
        "morphology_results.json": "✅" if (case_dir / "morphology_results.json").exists() else "❌",
        "tumour_location.json": "✅" if (case_dir / "tumour_location.json").exists() else "❌",
        "seg_enhancing.nii.gz": "✅" if (case_dir / "seg_enhancing.nii.gz").exists() else "❌",
        "brain_mask.nii.gz": "✅" if (case_dir / "brain_mask.nii.gz").exists() else "❌",
    }
    for fname, status in outputs.items():
        print(f"    {status} {fname}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
