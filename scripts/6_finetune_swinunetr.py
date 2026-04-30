#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_finetune_swinunetr.py: Decoder-only fine-tuning for Swin-UNETR.
Enables local adaptation of weights with a low learning rate.
"""

import sys
import torch
from pathlib import Path

from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, RandFlipd, ToTensord
from monai.data import DataLoader, Dataset

# Adjust path for pybrain imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pybrain.io.logging_utils import setup_logging


def fine_tune_model(
    bundle_dir: Path,
    data_list: list,  # List of dicts: {"image": path, "label": path}
    output_weights: Path,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "mps",
):
    # Dynamic device selection — respects MPS on Apple Silicon, CUDA on NVIDIA,
    # and falls back to CPU. Use caller's explicit device if provided, otherwise auto-detect.
    if device is None or device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    dev = torch.device(device)
    logger = setup_logging()
    logger.info(f"Fine-tuning starting: {epochs} epochs | LR: {lr} | Device: {device}")
    # Architecture: f48, in/out = 4/3
    model = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True).to(dev)

    checkpoint_path = bundle_dir / "fold1_swin_unetr.pth"  # Usually fold1 is the strongest
    if checkpoint_path.exists():
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=dev)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        model.load_state_dict(state_dict, strict=False)

    # 2. Freeze Encoder (Swin Transformer Blocks)
    # The SwinUNETR in MONAI uses 'swinViT' (capital T) as the encoder module name.
    # Freezing the encoder allows fast fine-tuning of only the decoder/bottleneck.
    if hasattr(model, "swinViT"):
        logger.info("Freezing Swin-ViT Encoder...")
        for param in model.swinViT.parameters():
            param.requires_grad = False
    elif hasattr(model, "swin_vit"):
        # Fallback for older MONAI versions that may use underscore variant
        logger.info("Freezing Swin-ViT Encoder (swin_vit fallback)...")
        for param in model.swin_vit.parameters():
            param.requires_grad = False
    else:
        logger.warning("Could not find Swin-ViT encoder module to freeze. Fine-tuning all parameters.")

    # 3. Training Utilities
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # include_background=False: BraTS out_channels=3 maps to {NCR,ED,ET}.
    # The background class (label=0) is never a target — excluding it prevents
    # the loss being diluted by large background regions and avoids the model
    # learning a trivial background-vs-tumour binary segmentation instead of
    # fine-grained multi-label tumour sub-region delineation.
    loss_fn = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)

    # Simple Mock Transforms/DataLoader (Assuming raw NIfTIs for now)
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # Use dummy data if local dataset list is empty (for validation of script)
    if not data_list:
        logger.warning("No data found for fine-tuning. Script ready but needs data list.")
        return

    train_ds = Dataset(data=data_list, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"].to(dev), batch["label"].to(dev)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

    # 5. Save Fine-tuned weights
    torch.save(model.state_dict(), output_weights)
    logger.info(f"Fine-tuned weights saved to {output_weights}")


if __name__ == "__main__":
    # Example usage:
    # python 6_finetune_swinunetr.py
    project_root = Path(__file__).resolve().parent.parent
    bundle_dir = project_root / "models" / "brats_bundle"
    output_w = bundle_dir / "finetuned_swin_unetr.pth"

    # Placeholder for local fine-tuning data
    local_data = []

    fine_tune_model(bundle_dir, local_data, output_w, epochs=5)
