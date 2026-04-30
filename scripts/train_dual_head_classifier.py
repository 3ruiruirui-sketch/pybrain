#!/usr/bin/env python3
"""
Train DualHeadSegResNet Classification Head (End-to-End)
========================================================
Properly trains the classification head on top of the SegResNet encoder
using extracted bottleneck features + labels from BraTS2021 cases.

Two phases:
  Phase 1: Extract bottleneck features for all eligible cases (if not cached)
  Phase 2: Train classifier head with 5-fold CV

Usage:
  python3 scripts/train_dual_head_classifier.py --max-cases 200
"""

import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pybrain.io.logging_utils import setup_logging

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SEQUENCES = ["t1", "t1c", "t2", "flair"]


class BottleneckDataset(Dataset):
    """Dataset for training classifier head from cached bottleneck features."""

    def __init__(self, case_ids, labels_df, results_dir):
        self.case_ids = case_ids
        self.labels_df = labels_df
        self.results_dir = results_dir

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        dh = np.load(self.results_dir / cid / "dualhead_bottleneck.npy")
        row = self.labels_df[self.labels_df.case_id == cid]
        g = str(row["grade"].values[0])
        grade_bin = 0 if g in ["I", "II", "2"] or "II" in g else 1
        return torch.from_numpy(dh).float(), torch.tensor(grade_bin).long(), cid


class VolumeDataset(Dataset):
    """Full 3D volume dataset — for end-to-end training with fresh inference."""

    def __init__(self, case_ids, labels_df, results_dir):
        self.case_ids = case_ids
        self.labels_df = labels_df
        self.results_dir = results_dir

    def __len__(self):
        return len(self.case_ids)

    def _load_volume(self, case_id):
        import nibabel as nib

        case_dir = self.results_dir / case_id
        volumes = {}
        for seq in SEQUENCES:
            p = case_dir / f"{seq}_resampled.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
            try:
                img = nib.load(str(p), mmap=False)
                arr = np.asanyarray(img.dataobj)
            except Exception as e:
                raise IOError(f"Corrupt: {p}: {e}")
            mask_p = case_dir / "brain_mask.nii.gz"
            if mask_p.exists():
                try:
                    mask = np.asanyarray(nib.load(str(mask_p), mmap=False).dataobj) > 0
                    arr = arr * mask.astype(np.float32)
                except Exception:
                    pass
            # Z-score normalize
            brain = arr > 0
            if brain.sum() > 0:
                arr[brain] = (arr[brain] - arr[brain].mean()) / max(arr[brain].std(), 1e-6)
            volumes[seq] = arr

        channel_order = ["t1c", "t1", "t2", "flair"]
        vol = np.stack([volumes[s] for s in channel_order], axis=0)
        return torch.from_numpy(vol.astype(np.float32))

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        try:
            vol = self._load_volume(cid)
        except IOError:
            return None
        row = self.labels_df[self.labels_df.case_id == cid]
        g = str(row["grade"].values[0])
        grade_bin = 0 if g in ["I", "II", "2"] or "II" in g else 1
        return vol, torch.tensor(grade_bin).long(), cid


class ClassifierMLP(nn.Module):
    """Simple MLP for classification from bottleneck features."""

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


def train_classifier_head(bottleneck_dir, labels_df, results_dir, num_epochs=30, batch_size=16, lr=1e-3):
    """
    Train classifier head from cached bottleneck features using simple MLP.
    """
    # Collect cases with bottleneck features
    labels_df = labels_df.copy()
    labels_df["grade_bin"] = labels_df["grade"].apply(
        lambda g: 0 if str(g) in ["I", "II", "2"] or "II" in str(g) else 1
    )

    cases_with_bottleneck = []
    for cid in labels_df["case_id"]:
        p = bottleneck_dir / cid / "dualhead_bottleneck.npy"
        if p.exists():
            cases_with_bottleneck.append(cid)

    print(f"  Training on {len(cases_with_bottleneck)} cases with cached bottleneck features")

    X = []
    y = []
    for cid in cases_with_bottleneck:
        feat = np.load(bottleneck_dir / cid / "dualhead_bottleneck.npy")
        label = labels_df[labels_df.case_id == cid]["grade_bin"].values[0]
        X.append(feat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"  Feature matrix: {X.shape} | LGG={int((y == 0).sum())} HGG={int((y == 1).sum())}")

    # Simple MLP classifier on bottleneck features
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = np.zeros(len(y))
    fold_aucs = []
    models = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = ClassifierMLP(input_dim=X.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()

        # Class balance weight for CrossEntropyLoss
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        weight = torch.tensor([1.0, float(n_neg / max(n_pos, 1))], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        # Training loop
        X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
        y_tr_t = torch.from_numpy(y_tr).long().to(DEVICE)
        X_val_t = torch.from_numpy(X_val).float().to(DEVICE)

        best_val_auc = 0
        best_state = None
        patience = 7
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            # Mini-batch
            perm = torch.randperm(len(X_tr_t))
            epoch_loss = 0
            for i in range(0, len(X_tr_t), batch_size):
                batch_idx = perm[i : i + batch_size]
                logits = model(X_tr_t[batch_idx])
                loss = criterion(logits, y_tr_t[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
                val_auc = roc_auc_score(y_val, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Load best and evaluate
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        y_prob_cv[val_idx] = val_probs
        fold_aucs.append(best_val_auc)
        models.append(model.cpu())
        print(f"  Fold {fold}: AUC={best_val_auc:.4f} (epoch {epoch - no_improve + 1})")

    y_pred = (y_prob_cv > 0.5).astype(int)
    cv_auc = roc_auc_score(y, y_prob_cv)
    cv_acc = (y_pred == y).mean()
    cm = confusion_matrix(y, y_pred)

    print(f"\n  Overall AUC: {cv_auc:.4f} | Accuracy: {cv_acc:.4f}")
    print(f"  Mean fold AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print("  Confusion Matrix:")
    print("               Pred LGG  Pred HGG")
    print(f"    True LGG     {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"    True HGG     {cm[1][0]:4d}     {cm[1][1]:4d}")
    print()
    print(classification_report(y, y_pred, target_names=["LGG", "HGG"]))

    # Save all fold models
    models_dir = PROJECT_ROOT / "models" / "dualhead_classifier_folds"
    models_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), models_dir / f"fold_{i + 1}.pt")

    return {
        "method": "MLP on DualHeadSegResNet bottleneck features (5-fold CV)",
        "n_cases": len(y),
        "lgg": int((y == 0).sum()),
        "hgg": int((y == 1).sum()),
        "cv_auc": float(cv_auc),
        "cv_acc": float(cv_acc),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "confusion_matrix": cm.tolist(),
        "fold_aucs": [float(a) for a in fold_aucs],
        "models_dir": str(models_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    setup_logging()
    from pybrain.io.session import get_session, get_paths

    sess = get_session()
    paths = get_paths(sess)
    results_dir = Path(paths["results_dir"]) / "validation_runs"
    bottleneck_dir = results_dir

    labels_df = __import__("pandas").read_csv(PROJECT_ROOT / "data" / "auto_generated_labels.csv")

    print("=" * 60)
    print("  DUALHEAD CLASSIFIER HEAD TRAINING (End-to-End)")
    print("=" * 60)

    t0 = time.time()
    metrics = train_classifier_head(
        bottleneck_dir=bottleneck_dir,
        labels_df=labels_df,
        results_dir=results_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    elapsed = time.time() - t0

    # Save metrics
    out_path = PROJECT_ROOT / "models" / "dualhead_classifier_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Metrics: {out_path}")
    print(f"  Fold models: {metrics['models_dir']}")

    # Ensemble: combine XGBoost trained on radiomics+dualhead features
    try:
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        import xgboost as xgb

        # Load radiomics features
        labels_df["grade_bin"] = labels_df["grade"].apply(
            lambda g: 0 if str(g) in ["I", "II", "2"] or "II" in str(g) else 1
        )

        # Cases with both radiomics + dualhead
        cases = []
        for _, row in labels_df.iterrows():
            c = row["case_id"]
            rd = results_dir / c
            rf = rd / "radiomics_features.json"
            dh = rd / "dualhead_bottleneck.npy"
            if rf.exists() and dh.exists():
                try:
                    with open(rf) as f:
                        d = json.load(f)
                    if isinstance(d.get("all_features"), dict) and len(d["all_features"]) >= 40:
                        cases.append(c)
                except:
                    pass

        def load_features(c):
            features = {}
            rd = results_dir / c
            with open(rd / "radiomics_features.json") as f:
                r = json.load(f)
            if isinstance(r.get("all_features"), dict):
                features.update(r["all_features"])
            dh = np.load(rd / "dualhead_bottleneck.npy")
            for i, v in enumerate(dh):
                features[f"dh_{i}"] = v
            return features

        all_feats = [load_features(c) for c in cases]
        all_keys = sorted(set().union(*[set(f.keys()) for f in all_feats]))
        X = np.zeros((len(all_feats), len(all_keys)), dtype=np.float32)
        for i, f in enumerate(all_feats):
            for j, k in enumerate(all_keys):
                X[i, j] = float(f.get(k, 0.0))
        y = np.array([labels_df[labels_df.case_id == c]["grade_bin"].values[0] for c in cases], dtype=np.int32)

        print(f"  Ensemble dataset: {X.shape} | LGG={int((y == 0).sum())} HGG={int((y == 1).sum())}")

        # XGBoost on radiomics + dualhead combined features (5-fold CV)
        xgb_params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "min_child_weight": 3,
            "gamma": 0.15,
            "reg_alpha": 0.4,
            "reg_lambda": 1.2,
            "scale_pos_weight": (y == 1).sum() / max((y == 0).sum(), 1),
            "tree_method": "hist",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        xgb.XGBClassifier(**xgb_params)
        cv_ens = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob_ens = np.zeros(len(y))
        fold_aucs_ens = []

        for fold, (tr_idx, val_idx) in enumerate(cv_ens.split(X, y), 1):
            clf_fold = xgb.XGBClassifier(**xgb_params)
            clf_fold.fit(X[tr_idx], y[tr_idx])
            prob = clf_fold.predict_proba(X[val_idx])[:, 1]
            y_prob_ens[val_idx] = prob
            auc = roc_auc_score(y[val_idx], prob)
            fold_aucs_ens.append(auc)
            print(f"  Fold {fold} XGB Ensemble AUC: {auc:.4f}")

        y_pred_ens = (y_prob_ens > 0.5).astype(int)
        ens_auc = roc_auc_score(y, y_prob_ens)
        ens_acc = (y_pred_ens == y).mean()
        cm_ens = confusion_matrix(y, y_pred_ens)

        print(f"\n  XGB Ensemble AUC: {ens_auc:.4f} | Accuracy: {ens_acc:.4f}")
        print("  Confusion Matrix:")
        print("               Pred LGG  Pred HGG")
        print(f"    True LGG     {cm_ens[0][0]:4d}     {cm_ens[0][1]:4d}")
        print(f"    True HGG     {cm_ens[1][0]:4d}     {cm_ens[1][1]:4d}")
        print()
        print(classification_report(y, y_pred_ens, target_names=["LGG", "HGG"]))

        ens_metrics = {
            "method": "XGBoost on radiomics+dualhead bottleneck combined features",
            "n_cases": len(y),
            "lgg": int((y == 0).sum()),
            "hgg": int((y == 1).sum()),
            "cv_auc": float(ens_auc),
            "cv_acc": float(ens_acc),
            "mean_fold_auc": float(np.mean(fold_aucs_ens)),
            "std_fold_auc": float(np.std(fold_aucs_ens)),
            "confusion_matrix": cm_ens.tolist(),
            "fold_aucs": [float(a) for a in fold_aucs_ens],
        }
        with open(PROJECT_ROOT / "models" / "ensemble_metrics.json", "w") as f:
            json.dump(ens_metrics, f, indent=2)
        print("  Ensemble metrics: models/ensemble_metrics.json")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"  Ensemble skipped: {e}")

    print("\n  Training complete!")


if __name__ == "__main__":
    main()
