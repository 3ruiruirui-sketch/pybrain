"""
Brain metastases segmentation module.
Stage B of the 2-stage mets pipeline: per-lesion segmentation on cropped patches.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from pathlib import Path
import logging

from pybrain.io.logging_utils import get_logger

logger = get_logger("pybrain")


def segment_lesion(
    patch: np.ndarray,
    model: torch.nn.Module,
    device: str = "cuda",
) -> np.ndarray:
    """
    Segment a single lesion patch using a 3D U-Net model.

    Args:
        patch: 3D image patch (z, y, x), typically 64³
        model: Trained 3D segmentation model (SegResNet)
        device: Device to run inference on ("cuda" or "cpu")

    Returns:
        Binary segmentation mask (z, y, x) for the lesion
    """
    model.eval()
    model.to(device)

    # Prepare input
    # Add batch and channel dimensions: (1, 1, z, y, x)
    patch_tensor = torch.from_numpy(patch).float()
    patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # Normalize
    patch_tensor = (patch_tensor - patch_tensor.min()) / (patch_tensor.max() - patch_tensor.min() + 1e-8)

    with torch.no_grad():
        # Forward pass
        output = model(patch_tensor)

        # Apply sigmoid and threshold
        probs = torch.sigmoid(output)
        segmentation = (probs > 0.5).squeeze().cpu().numpy()

    return segmentation


def segment_all_lesions(
    candidates: List,  # List of LesionCandidate from detector
    t1c: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    Segment all detected lesions using the per-lesion model.

    Args:
        candidates: List of LesionCandidate objects from detection stage
        t1c: Full T1c image (z, y, x)
        t1: Full T1 image (z, y, x)
        t2: Full T2 image (z, y, x)
        flair: Full FLAIR image (z, y, x)
        model: Trained 3D segmentation model
        patch_size: Patch size for per-lesion segmentation (z, y, x)
        device: Device to run inference on

    Returns:
        List of binary segmentation masks, one per lesion
    """
    segmentations = []

    for candidate in candidates:
        # Extract patch around centroid
        centroid = candidate.centroid
        patch = _extract_patch(t1c, centroid, patch_size)

        # Segment the patch
        seg = segment_lesion(patch, model, device)
        segmentations.append(seg)

    return segmentations


def _extract_patch(
    image: np.ndarray,
    centroid: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Extract a patch centered on a centroid from the image.

    Args:
        image: Full image (z, y, x)
        centroid: Centroid (z, y, x)
        patch_size: Desired patch size (z, y, x)

    Returns:
        Extracted patch (z, y, x). Padded with zeros if near boundaries.
    """
    z, y, x = centroid
    pz, py, px = patch_size

    # Calculate patch boundaries
    z1 = max(0, z - pz // 2)
    z2 = min(image.shape[0], z + pz // 2)
    y1 = max(0, y - py // 2)
    y2 = min(image.shape[1], y + py // 2)
    x1 = max(0, x - px // 2)
    x2 = min(image.shape[2], x + px // 2)

    # Extract patch
    patch = image[z1:z2, y1:y2, x1:x2]

    # Pad if necessary
    pad_z = pz - patch.shape[0]
    pad_y = py - patch.shape[1]
    pad_x = px - patch.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        # Pad symmetrically
        pad_z_before = pad_z // 2
        pad_z_after = pad_z - pad_z_before
        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before
        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before

        patch = np.pad(
            patch,
            ((pad_z_before, pad_z_after), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode="constant",
            constant_values=0,
        )

    return patch


def load_mets_segmenter(model_path: Path, device: str = "cuda") -> torch.nn.Module:
    """
    Load a pretrained mets segmentation model.

    Args:
        model_path: Path to model weights (.pth file)
        device: Device to load model on

    Returns:
        Loaded model
    """
    try:
        from monai.networks.nets.segresnet import SegResNet
    except ImportError:
        raise ImportError("MONAI is required for mets segmentation")

    # Initialize SegResNet model
    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=1,
        out_channels=1,
        dropout_prob=0.2,
    )

    # Load weights
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded mets segmenter from {model_path}")
    else:
        logger.warning(f"Model path {model_path} not found, using untrained model")

    model.to(device)
    return model


def combine_lesion_segmentations(
    segmentations: List[np.ndarray],
    candidates: List,
    image_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Combine per-lesion segmentations into a full-volume segmentation.

    Args:
        segmentations: List of binary segmentation masks
        candidates: List of LesionCandidate objects with centroid info
        image_shape: Target image shape (z, y, x)

    Returns:
        Combined binary segmentation (z, y, x)
    """
    combined = np.zeros(image_shape, dtype=np.uint8)
    patch_size = segmentations[0].shape

    for seg, candidate in zip(segmentations, candidates):
        centroid = candidate.centroid
        z, y, x = centroid
        pz, py, px = patch_size

        # Calculate where to place the patch in the full volume
        z1 = max(0, z - pz // 2)
        z2 = min(image_shape[0], z + pz // 2)
        y1 = max(0, y - py // 2)
        y2 = min(image_shape[1], y + py // 2)
        x1 = max(0, x - px // 2)
        x2 = min(image_shape[2], x + px // 2)

        # Calculate patch boundaries (accounting for padding)
        pad_z_before = (pz // 2) - (z - z1)
        pad_y_before = (py // 2) - (y - y1)
        pad_x_before = (px // 2) - (x - x1)

        pad_z_after = (z + pz // 2) - z2
        pad_y_after = (y + py // 2) - y2
        pad_x_after = (x + px // 2) - x2

        # Extract the actual patch region (removing padding)
        seg_patch = seg[
            pad_z_before : pz - pad_z_after if pad_z_after > 0 else pz,
            pad_y_before : py - pad_y_after if pad_y_after > 0 else py,
            pad_x_before : px - pad_x_after if pad_x_after > 0 else px,
        ]

        # Place in combined volume
        combined[z1:z2, y1:y2, x1:x2] = np.maximum(
            combined[z1:z2, y1:y2, x1:x2], seg_patch
        )

    return combined
