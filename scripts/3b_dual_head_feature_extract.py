#!/usr/bin/env python3
"""
Stage 3b — DualHeadSegResNet Feature Extraction
=================================================
Runs DualHeadSegResNet inference on preprocessed BraTS volumes
to extract encoder bottleneck features (128-dim after GAP) for
training the classification head.

For each case:
  1. Load 4 preprocessed modalities + brain mask
  2. Create tumour mask (ET + ED region)
  3. Normalize per modality (z-score within brain mask)
  4. Permute channels → MONAI order [T1c, T1, T2, FLAIR]
  5. Run through DualHeadSegResNet (return_seg=False)
  6. Save bottleneck features + class logits to result dir

Usage:
  python3 scripts/3b_dual_head_feature_extract.py --max-cases 50
"""

import sys, json, time, argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.models.dual_head import DualHeadSegResNet
from pybrain.io.session import get_session, get_paths
from pybrain.io.logging_utils import setup_logging

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Channel order for MONAI SegResNet: [T1c, T1, T2, FLAIR]
SEQUENCES = ["t1", "t1c", "t2", "flair"]


class DualHeadDataset(Dataset):
    """Wraps preprocessed BraTS volumes for DualHeadSegResNet inference."""

    def __init__(self, case_ids, results_dir, labels_df):
        self.case_ids = case_ids
        self.results_dir = results_dir
        self.labels_df = labels_df

    def __len__(self):
        return len(self.case_ids)

    def _load_case(self, case_id):
        case_dir = self.results_dir / case_id

        # Load 4 modalities
        volumes = {}
        for seq in SEQUENCES:
            p = case_dir / f"{seq}_resampled.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
            try:
                img = nib.load(str(p), mmap=False)
                arr = np.asanyarray(img.dataobj)
            except Exception as e:
                raise IOError(f"Corrupt file {p}: {e}") from e
            # Apply brain mask
            mask_p = case_dir / "brain_mask.nii.gz"
            if mask_p.exists():
                try:
                    mask_img = nib.load(str(mask_p), mmap=False)
                    mask = np.asanyarray(mask_img.dataobj) > 0
                except Exception:
                    mask = None
                if mask is not None:
                    arr = arr * mask.astype(np.float32)
            volumes[seq] = arr
            affine = img.affine

        # Normalize each modality (z-score within brain region)
        for seq in SEQUENCES:
            vol = volumes[seq]
            brain_mask = vol > 0
            if brain_mask.sum() > 0:
                mu = vol[brain_mask].mean()
                sigma = vol[brain_mask].std()
                if sigma > 1e-6:
                    vol[brain_mask] = (vol[brain_mask] - mu) / sigma
            volumes[seq] = vol

        # Stack in MONAI channel order: [T1c, T1, T2, FLAIR]
        channel_order = ["t1c", "t1", "t2", "flair"]
        vol_stack = np.stack([volumes[seq] for seq in channel_order], axis=0)
        return torch.from_numpy(vol_stack.astype(np.float32)), case_id, affine

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        try:
            vol, cid, affine = self._load_case(case_id)
        except (IOError, OSError) as e:
            # Corrupt NIfTI — skip this case
            print(f"  ⚠️  Skipping corrupt case {case_id}: {e}")
            return None

        row = self.labels_df[self.labels_df.case_id == cid]
        grade_str = str(row["grade"].values[0])
        grade_bin = 0 if grade_str in ["I", "II", "2"] or "II" in grade_str else 1

        return {
            "volume": vol,
            "case_id": cid,
            "grade": grade_str,
            "grade_bin": grade_bin,
            "affine": affine,
        }


def collate_fn(batch):
    """Pad variable-size volumes to a fixed grid; skip None (corrupt) cases."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    affines = []
    volumes, case_ids, grade_bins = [], [], []

    max_d, max_h, max_w = 0, 0, 0
    for item in batch:
        v = item["volume"]
        max_d = max(max_d, v.shape[1])
        max_h = max(max_h, v.shape[2])
        max_w = max(max_w, v.shape[3])

    for item in batch:
        v = item["volume"]
        # Pad to max shape
        d, h, w = v.shape[1:]
        pd = max_d - d
        ph = max_h - h
        pw = max_w - w
        v_pad = torch.nn.functional.pad(v, (0, pw, 0, ph, 0, pd))
        volumes.append(v_pad)
        case_ids.append(item["case_id"])
        grade_bins.append(item["grade_bin"])
        affines.append(item["affine"])

    volumes = torch.stack(volumes, dim=0)
    grade_bins = torch.tensor(grade_bins, dtype=torch.long)

    return {
        "volume": volumes,
        "case_id": case_ids,
        "grade_bin": grade_bins,
        "affine": affines,
    }


def extract_bottleneck_features(model, dataloader, results_dir):
    """Run inference, save bottleneck features + logits for each case."""
    model.eval()
    results = []
    skipped = 0

    for batch in dataloader:
        if batch is None:
            continue
        volumes = batch["volume"].to(DEVICE)
        case_ids = batch["case_id"]
        grade_bins = batch["grade_bin"].numpy()

        try:
            with torch.no_grad():
                out = model(volumes, return_seg=False, return_class_logits=True)
            # Raw encoder bottleneck: (B, 128, D', H', W') — average pool to save
            bottleneck = out["bottleneck"]  # (B, 128, D', H', W')
            out["class_logits"]
            class_probs = out["class_probs"]
        except Exception as e:
            print(f"  ⚠️  Inference error for batch {case_ids}: {e}")
            skipped += len(case_ids)
            continue

        for i, cid in enumerate(case_ids):
            feat_dir = results_dir / cid
            feat_dir.mkdir(parents=True, exist_ok=True)
            # Pooled: (B, 128, D', H', W') → per-sample (128,) = mean over spatial dims only
            pooled = bottleneck[i].mean(dim=(1, 2, 3)).cpu().numpy()  # (128,)
            np.save(feat_dir / "dualhead_bottleneck.npy", pooled)
            prob = class_probs[i].cpu().numpy()
            grade_bin = grade_bins[i]
            true_label = "LGG" if grade_bin == 0 else "HGG"
            print(f"  {cid} (True: {true_label}) → p(HGG)={prob[1]:.3f}")
            results.append(
                {
                    "case_id": cid,
                    "grade_bin": int(grade_bin),
                    "p_lgg": float(prob[0]),
                    "p_hgg": float(prob[1]),
                }
            )

    print(f"  Skipped (error): {skipped}")
    return results


def main():
    parser = argparse.ArgumentParser(description="DualHeadSegResNet feature extraction")
    parser.add_argument("--max-cases", type=int, default=0, help="Max cases to process (0=all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if dualhead_bottleneck.npy exists")
    args = parser.parse_args()

    setup_logging()
    sess = get_session()
    paths = get_paths(sess)
    results_dir = Path(paths["results_dir"]) / "validation_runs"

    labels_df = __import__("pandas").read_csv(PROJECT_ROOT / "data" / "auto_generated_labels.csv")

    # Find all cases with preprocessed volumes and brain mask
    all_cases = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    eligible = []
    for case_id in all_cases:
        cd = results_dir / case_id
        has_all = all((cd / f"{s}_resampled.nii.gz").exists() for s in SEQUENCES)
        has_mask = (cd / "brain_mask.nii.gz").exists()
        if has_all and has_mask:
            # Check skip-existing
            if args.skip_existing and (cd / "dualhead_bottleneck.npy").exists():
                continue
            eligible.append(case_id)

    if args.max_cases > 0:
        eligible = eligible[: args.max_cases]

    print(f"Found {len(all_cases)} total cases | Eligible: {len(eligible)}")
    if not eligible:
        print("No eligible cases found.")
        return

    # Load dual-head model
    print(f"\nLoading DualHeadSegResNet on {DEVICE}...")
    model = DualHeadSegResNet(
        num_classes=2,
        dropout_rate=0.4,
        pretrained_path=str(
            PROJECT_ROOT / "models" / "brats_bundle" / "brats_mri_segmentation" / "models" / "model.pt"
        ),
    ).to(DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {total:,} total params, {trainable:,} trainable (classifier head only)")

    # Dataset + DataLoader
    dataset = DualHeadDataset(eligible, results_dir, labels_df)
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # 2 cases per batch (each ~240x240x155 = ~35MB)
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"\nProcessing {len(eligible)} cases...")
    t0 = time.time()
    results = extract_bottleneck_features(model, dataloader, results_dir)
    elapsed = time.time() - t0

    # Save inference results
    out_path = results_dir.parent / "dualhead_inference_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "cases": results,
                "n_cases": len(results),
                "elapsed_seconds": round(elapsed, 1),
                "cases_per_second": round(len(results) / elapsed, 2),
            },
            f,
            indent=2,
        )

    print(f"\nDone: {len(results)} cases in {elapsed:.1f}s ({len(results) / elapsed:.2f} cases/s)")
    print(f"Results saved: {out_path}")

    # Quick accuracy check
    if results:
        correct = sum(1 for r in results if (r["p_hgg"] > 0.5) == r["grade_bin"])
        acc = correct / len(results)
        print(f"Accuracy (p>0.5 threshold): {acc:.1%} ({correct}/{len(results)})")


if __name__ == "__main__":
    main()
