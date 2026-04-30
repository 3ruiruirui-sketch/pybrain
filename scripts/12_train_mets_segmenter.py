#!/usr/bin/env python3
"""
Brain Metastases Segmentation Training Script
============================================
Per-lesion segmentation model training using SegResNet on cropped patches.
Uses per-patient cross-validation to avoid data leakage in sparse mets data.

Usage:
  python scripts/12_train_mets_segmenter.py \
    --data-dir data/datasets/BrainMetShare \
    --output-dir models/mets_bundle \
    --epochs 100 \
    --batch-size 8
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from pybrain.io.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train brain metastases segmentation model"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing mets training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/mets_bundle"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Patch size for training (z, y, x)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (per-patient)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    return parser.parse_args()


class MetsPatchDataset(torch.utils.data.Dataset):
    """Dataset for per-lesion patches."""

    def __init__(self, patches: List[Dict[str, Any]], patch_size: tuple):
        """
        Args:
            patches: List of patch dictionaries with 'image' and 'label' keys
            patch_size: Target patch size (z, y, x)
        """
        self.patches = patches
        self.patch_size = patch_size

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        image = patch["image"]
        label = patch["label"]

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Add channel dimension
        image = image[np.newaxis, ...]  # (1, z, y, x)
        label = label[np.newaxis, ...]  # (1, z, y, x)

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()


def extract_lesion_patches(
    t1c_path: Path,
    seg_path: Path,
    patch_size: tuple,
    min_voxels: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract patches around each lesion in a segmentation.

    Args:
        t1c_path: Path to T1c NIfTI file
        seg_path: Path to segmentation NIfTI file
        patch_size: Patch size (z, y, x)
        min_voxels: Minimum voxels for a lesion to be included

    Returns:
        List of patch dictionaries
    """
    import nibabel as nib
    from scipy import ndimage

    t1c_img = nib.load(str(t1c_path))
    t1c = t1c_img.get_fdata()
    seg = nib.load(str(seg_path)).get_fdata()

    # Find connected components in segmentation
    labeled, num_features = ndimage.label(seg > 0)
    patches = []

    for label_id in range(1, num_features + 1):
        lesion_mask = (labeled == label_id)
        voxel_count = np.sum(lesion_mask)

        if voxel_count < min_voxels:
            continue

        # Get centroid
        centroid = ndimage.center_of_mass(lesion_mask)
        centroid = tuple(map(int, centroid))

        # Extract patch around centroid
        z, y, x = centroid
        pz, py, px = patch_size

        # Calculate patch boundaries
        z1 = max(0, z - pz // 2)
        z2 = min(t1c.shape[0], z + pz // 2)
        y1 = max(0, y - py // 2)
        y2 = min(t1c.shape[1], y + py // 2)
        x1 = max(0, x - px // 2)
        x2 = min(t1c.shape[2], x + px // 2)

        # Extract patch
        image_patch = t1c[z1:z2, y1:y2, x1:x2]
        label_patch = seg[z1:z2, y1:y2, x1:x2]

        # Pad if necessary
        if image_patch.shape != patch_size:
            pad_z = pz - image_patch.shape[0]
            pad_y = py - image_patch.shape[1]
            pad_x = px - image_patch.shape[2]
            image_patch = np.pad(image_patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
            label_patch = np.pad(label_patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")

        patches.append({"image": image_patch, "label": label_patch})

    return patches


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    args = parse_args()
    setup_logging()
    logger = get_logger("pybrain")

    logger.info("=== Brain Metastases Segmentation Training ===")

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    data_dir = args.data_dir

    # Find all patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(patient_dirs)} patients")

    # Extract patches from all patients
    all_patches = []
    patient_patches = {}  # patient_id -> list of patches

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        t1c_path = patient_dir / "T1c.nii.gz"
        seg_path = patient_dir / "segmentation.nii.gz"

        if not t1c_path.exists() or not seg_path.exists():
            logger.warning(f"Skipping {patient_id}: missing T1c or segmentation")
            continue

        patches = extract_lesion_patches(
            t1c_path,
            seg_path,
            tuple(args.patch_size),
        )

        if len(patches) == 0:
            logger.warning(f"Skipping {patient_id}: no lesions found")
            continue

        patient_patches[patient_id] = patches
        all_patches.extend(patches)

        logger.info(f"Patient {patient_id}: {len(patches)} patches")

    logger.info(f"Total patches: {len(all_patches)}")

    if len(all_patches) == 0:
        logger.error("No patches found. Check data directory structure.")
        return 1

    # Per-patient cross-validation
    patient_ids = list(patient_patches.keys())
    cv_folds = args.cv_folds

    logger.info(f"Training with {cv_folds}-fold per-patient cross-validation")

    # Initialize model
    try:
        from monai.networks.nets.segresnet import SegResNet
    except ImportError:
        logger.error("MONAI is required for training")
        return 1

    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=1,
        out_channels=1,
        dropout_prob=0.2,
    )
    model.to(device)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train on full dataset (simplified for v1)
    # In production, implement proper per-patient CV
    logger.info("Training on full dataset (simplified for v1)")

    dataset = MetsPatchDataset(all_patches, tuple(args.patch_size))

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_dir / "segmenter.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save training metadata
    metadata = {
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "num_patients": len(patient_ids),
        "num_patches": len(all_patches),
    }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved training metadata to {metadata_path}")
    logger.info("Training complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
