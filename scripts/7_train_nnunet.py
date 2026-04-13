#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7_train_nnunet.py — Train DynUNet (nnU-Net) on BraTS2021 dataset.
================================================================
Fine-tunes a DynUNet model on the BraTS2021 training data and saves
the best weights to models/brats_bundle/nnunet_weights.pth.

Supports:
  - Full BraTS2021_Training_Data (1251 cases)
  - Label remapping: BraTS2021 uses 0,1,2,4 → pipeline uses 0,1,2,3
  - Multi-fold cross-validation (default: 1 fold for speed)
  - Mixed precision (MPS/CUDA)
  - Transfer learning from any existing nnunet_weights.pth

Usage:
  python3 scripts/7_train_nnunet.py
  python3 scripts/7_train_nnunet.py --epochs 20 --fold 0 --lr 1e-4
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.io.logging_utils import setup_logging
from pybrain.models.nnunet import load_nnunet

from monai.networks.nets import DynUNet
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset, partition_dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandFlipd, RandRotate90d, ToTensord, CastToTyped,
    ResizeWithPadOrCropd
)

# ── Device ──────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── BraTS2021 label remapping ───────────────────────────────────────────────
# BraTS2021:       0=background, 1=NET(NCR), 2=ED, 4=ET
# PY-BRAIN/Brats:  0=background, 1=necrotic, 2=edema,  3=enhancing
LABEL_MAP_BRATS_TO_PYBRAIN = {0: 0, 1: 1, 2: 2, 4: 3}
# For loss computation we use original BraTS labels 0,1,2,4
# then remap output to pipeline convention 0,1,2,3


def remap_labels(label: np.ndarray) -> np.ndarray:
    """Remap BraTS2021 labels (0,1,2,4) → pipeline labels (0,1,2,3)."""
    out = np.zeros_like(label)
    for src, dst in LABEL_MAP_BRATS_TO_PYBRAIN.items():
        out[label == src] = dst
    return out


# ── Dataset discovery ───────────────────────────────────────────────────────
def discover_cases(data_root: Path):
    """
    Find all BraTS2021 training cases.
    Returns list of dicts: {"t1": ..., "t1ce": ..., "t2": ..., "flair": ..., "seg": ...}
    """
    cases = []
    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("BraTS2021_"):
            continue
        prefix = case_dir.name
        seg = case_dir / f"{prefix}_seg.nii.gz"
        t1   = case_dir / f"{prefix}_t1.nii.gz"
        t1ce = case_dir / f"{prefix}_t1ce.nii.gz"
        t2   = case_dir / f"{prefix}_t2.nii.gz"
        flair= case_dir / f"{prefix}_flair.nii.gz"
        if all(p.exists() for p in [seg, t1, t1ce, t2, flair]):
            cases.append({
                "t1":    str(t1),
                "t1ce":  str(t1ce),
                "t2":    str(t2),
                "flair": str(flair),
                "seg":   str(seg),
                "id":    prefix,
            })
    return cases


# ── Transforms ──────────────────────────────────────────────────────────────
NNUNET_PATCH_SIZE = [128, 128, 128]  # Standard nnU-Net input size


def get_train_transforms():
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        CastToTyped(keys=["t1", "t1ce", "t2", "flair", "seg"], dtype=torch.float32),
        NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True),
        # Resize all volumes (images + seg) to 128^3 (standard nnU-Net input)
        ResizeWithPadOrCropd(keys=["t1", "t1ce", "t2", "flair", "seg"], spatial_size=NNUNET_PATCH_SIZE),
        # Per-key RandFlipd on MPS avoids MONAI #6384 crash with multi-key spatial transforms.
        # Each modality gets independent flip — acceptable for co-registered brain MRI.
        RandFlipd(keys=["t1"],    prob=0.5, spatial_axis=0),
        RandFlipd(keys=["t1ce"],  prob=0.5, spatial_axis=0),
        RandFlipd(keys=["t2"],    prob=0.5, spatial_axis=0),
        RandFlipd(keys=["flair"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["t1"],    prob=0.5, spatial_axis=1),
        RandFlipd(keys=["t1ce"],  prob=0.5, spatial_axis=1),
        RandFlipd(keys=["t2"],    prob=0.5, spatial_axis=1),
        RandFlipd(keys=["flair"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["t1", "t1ce", "t2", "flair", "seg"], prob=0.5),
        ToTensord(keys=["t1", "t1ce", "t2", "flair", "seg"]),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        CastToTyped(keys=["t1", "t1ce", "t2", "flair", "seg"], dtype=torch.float32),
        NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True),
        ResizeWithPadOrCropd(keys=["t1", "t1ce", "t2", "flair", "seg"], spatial_size=NNUNET_PATCH_SIZE),
        ToTensord(keys=["t1", "t1ce", "t2", "flair", "seg"]),
    ])


# ── Collate function ───────────────────────────────────────────────────────
def _resize_seg_nearest(seg: torch.Tensor, target: list) -> torch.Tensor:
    """Resize seg (B,1,D,H,W) → (B,target[0],target[1],target[2]) with nearest."""
    import torch.nn.functional as F
    # Convert MetaTensor → numpy → plain torch.Tensor to bypass torch_function
    seg_np = seg.detach().cpu().numpy()
    seg_plain = torch.from_numpy(seg_np)
    out = F.interpolate(seg_plain.float(), size=target, mode="nearest")
    return out.squeeze(1).long()  # (B,D,H,W)


def brats_collate(batch):
    """Stack 4 MRI channels + label into (B, C, D, H, W) at 128^3 resolution."""
    t1    = torch.stack([b["t1"]    for b in batch], dim=0)
    t1ce  = torch.stack([b["t1ce"]  for b in batch], dim=0)
    t2    = torch.stack([b["t2"]    for b in batch], dim=0)
    flair = torch.stack([b["flair"] for b in batch], dim=0)
    # seg from dataset is already (B, 1, D, H, W) — no extra unsqueeze needed
    seg_raw = torch.stack([b["seg"] for b in batch], dim=0)  # (B,1,D,H,W)
    seg_resized = _resize_seg_nearest(seg_raw, NNUNET_PATCH_SIZE)  # (B,128,128,128)

    # Images: (B, 1, D, H, W) → (B, 4, D, H, W)
    images = torch.cat([t1, t1ce, t2, flair], dim=1)

    # Remap BraTS labels 0,1,2,4 → 0,1,2,3
    seg_tensor = torch.from_numpy(remap_labels(seg_resized.cpu().numpy())).long()

    return {"image": images, "label": seg_tensor}


def val_collate(batch):
    t1    = torch.stack([b["t1"]    for b in batch], dim=0)
    t1ce  = torch.stack([b["t1ce"]  for b in batch], dim=0)
    t2    = torch.stack([b["t2"]    for b in batch], dim=0)
    flair = torch.stack([b["flair"] for b in batch], dim=0)
    seg_raw = torch.stack([b["seg"] for b in batch], dim=0)  # (B,1,D,H,W)
    seg_resized = _resize_seg_nearest(seg_raw, NNUNET_PATCH_SIZE)

    images = torch.cat([t1, t1ce, t2, flair], dim=1)
    seg_tensor = torch.from_numpy(remap_labels(seg_resized.cpu().numpy())).long()

    return {"image": images, "label": seg_tensor}


# ── Dice score per class ─────────────────────────────────────────────────────
def dice_per_class(pred, target, num_classes=4):
    """Compute Dice for each class (0-3)."""
    dice_scores = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = (2 * intersection / (denom + 1e-8)).item()
        dice_scores.append(dice)
    return dice_scores


# ── Training loop ────────────────────────────────────────────────────────────
def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    logger,
    output_path,
    fold=0,
    val_freq=5,
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Dice + CE combined loss
    dice_loss = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    ce_loss   = nn.CrossEntropyLoss()

    # AMP: only CUDA supports FP16 conv transpose on MPS we fall back to FP32
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu")

    best_val_dice = 0.0
    best_epoch     = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches  = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                outputs = model(images)           # (B, 4, D, H, W) logits
                d_loss  = dice_loss(outputs, labels.unsqueeze(1))  # DiceLoss needs (B,1,D,H,W)
                c_loss  = ce_loss(outputs, labels)
                loss    = d_loss + c_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                dice  = dice_per_class(preds, labels)
                epoch_dice += np.mean(dice)
            n_batches += 1

        scheduler.step()
        avg_loss  = epoch_loss  / n_batches
        avg_dice  = epoch_dice  / n_batches

        logger.info(
            f"Fold {fold} | Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Train Dice: {avg_dice:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation
        if (epoch + 1) % val_freq == 0 or epoch == epochs - 1:
            model.eval()
            val_dice_scores = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                        outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    dice  = dice_per_class(preds, labels, num_classes=4)
                    val_dice_scores.append(dice)

            val_dice = np.mean(val_dice_scores, axis=0)
            # Mean over classes 1,2,3 (ignore background 0)
            val_dice_mean = np.mean(val_dice[1:])
            logger.info(
                f"  Validation | WT Dice: {val_dice[1]:.4f} | "
                f"TC Dice: {np.mean(val_dice[1:3]):.4f} | "
                f"ET Dice: {val_dice[3]:.4f}"
            )

            if val_dice_mean > best_val_dice:
                best_val_dice = val_dice_mean
                best_epoch     = epoch + 1
                torch.save(model.state_dict(), output_path)
                logger.info(
                    f"  ✅ New best model saved (Dice={val_dice_mean:.4f})"
                )

    logger.info(
        f"Fold {fold} complete. Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch}"
    )
    return best_val_dice


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train DynUNet on BraTS2021")
    parser.add_argument("--data",        default=None,
                        help="Path to BraTS2021_Training_Data folder")
    parser.add_argument("--epochs",       type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size",   type=int,   default=1,
                        help="Batch size (default: 1, keep 1 for 3D)")
    parser.add_argument("--fold",        type=int,   default=0,
                        help="Fold index for cross-validation (default: 0)")
    parser.add_argument("--output",      default=None,
                        help="Output weights path (default: models/brats_bundle/nnunet_weights.pth)")
    parser.add_argument("--transfer",    default=None,
                        help="Path to weights to transfer from (e.g. fold1_swin_unetr.pth)")
    parser.add_argument("--max_train",  type=int, default=0,
                        help="Max training cases to use (0 = all, for quick test use e.g. 100)")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    logger.info(f"Device: {device}")

    # Paths
    data_root = Path(args.data) if args.data else (
        PROJECT_ROOT / "data" / "datasets" / "BraTS2021" / "raw" / "BraTS2021_Training_Data"
    )
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "models" / "brats_bundle" / "nnunet_weights.pth"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover cases
    logger.info(f"Scanning {data_root} ...")
    cases = discover_cases(data_root)
    logger.info(f"Found {len(cases)} valid BraTS2021 cases")

    if len(cases) < 10:
        logger.error("Not enough training cases. Check data path.")
        sys.exit(1)

    # Shuffle and split: 90% train, 10% val
    random.seed(42)
    random.shuffle(cases)
    n_val  = max(1, int(0.1 * len(cases)))
    n_train = len(cases) - n_val
    train_cases = cases[:n_train]
    val_cases   = cases[n_train:]
    if args.max_train > 0:
        train_cases = train_cases[:args.max_train]
        logger.info(f"Limiting train to {args.max_train} cases (use --max_train 0 for full)")
    logger.info(f"Train: {len(train_cases)} | Val: {len(val_cases)}")

    # Create datasets
    train_ds = Dataset(train_cases, transform=get_train_transforms())
    val_ds   = Dataset(val_cases,   transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=brats_collate, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=val_collate, pin_memory=(device.type == "cuda")
    )

    # Build model
    logger.info("Building DynUNet model...")
    model = load_nnunet(device, {
        "filters": [32, 64, 128, 256, 512],
        "deep_supervision": False,
    })

    # Transfer learning: load existing nnunet_weights if present
    transfer_path = args.transfer
    if transfer_path:
        tp = Path(transfer_path)
        if tp.exists():
            logger.info(f"Transfer learning: loading {tp}")
            ckpt = torch.load(tp, map_location=device, weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            msg = model.load_state_dict(sd, strict=False)
            logger.info(f"  Loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        else:
            logger.warning(f"Transfer path {tp} not found — starting from scratch")

    model = model.to(device)

    # Train
    logger.info(
        f"Starting training: {args.epochs} epochs | LR={args.lr} | "
        f"BS={args.batch_size} | Fold={args.fold}"
    )
    best_dice = train_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        logger=logger,
        output_path=output_path,
        fold=args.fold,
        val_freq=1,
    )

    # Save training report
    report = {
        "model": "DynUNet (nnU-Net)",
        "fold": args.fold,
        "best_val_dice": float(best_dice),
        "epochs": args.epochs,
        "lr": args.lr,
        "n_train": len(train_cases),
        "n_val": len(val_cases),
        "output_weights": str(output_path),
    }
    report_path = output_path.with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Training report → {report_path}")
    logger.info(f"✅ Training complete. Weights → {output_path}")
    logger.info(f"   To use: copy {output_path} into the pipeline's models/brats_bundle/")


if __name__ == "__main__":
    main()
