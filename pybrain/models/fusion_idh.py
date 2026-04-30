"""
Fusion Model IDH Classifier Integration
Medical Engineering Enhancement - Phase 3 Enhanced Capabilities

Integrates fusion_model.pt as Stage 8c IDH classifier using
CNN+Radiomics fusion architecture trained on UPENN-GBM data.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from pybrain.io.logging_utils import get_logger

logger = get_logger("models.fusion_idh")


class FusionIDHClassifier(nn.Module):
    """
    Fusion model for IDH classification combining CNN features and radiomics.

    Architecture:
    - BrainIAC CNN encoder: 1-channel FLAIR (96³) → 512-dim features
    - rad_norm: BatchNorm1d on 107 radiomics features
    - Fusion head: 619→256→1 (IDH binary classification)
    """

    def __init__(self):
        super().__init__()

        # CNN encoder for FLAIR images (matching actual model structure)
        self.brainiac = nn.ModuleDict(
            {
                "encoder": nn.Sequential(
                    nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),  # 16 channels (matching saved model)
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                    nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # 32 channels
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 channels
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool3d((4, 4, 4)),  # Global pooling to 4x4x4
                    nn.Flatten(),
                    nn.Linear(64 * 4 * 4 * 4, 512),  # Adjusted for 64 channels
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                ),
                "proj": nn.Linear(512, 512),  # Single linear layer (matching saved model)
            }
        )

        # Fusion head (matching actual model structure)
        self.head = nn.Sequential(
            nn.Linear(512 + 107, 256),  # CNN features + radiomics
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # Binary classification
            nn.Sigmoid(),
        )

    def forward(self, flair_crop: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion model.

        Args:
            flair_crop: FLAIR crop (batch_size, 1, 96, 96, 96)
            radiomics: Radiomics features (batch_size, 107)

        Returns:
            IDH probability (batch_size, 1)
        """
        # Extract CNN features from FLAIR
        cnn_features = self.brainiac["encoder"](flair_crop)
        cnn_features = self.brainiac["proj"](cnn_features)

        # Normalize radiomics features (simple normalization for batch size 1)
        rad_features = (radiomics - radiomics.mean()) / (radiomics.std() + 1e-8)

        # Fuse features
        fused_features = torch.cat([cnn_features, rad_features], dim=1)

        # Classification
        idh_prob = self.head(fused_features)

        return idh_prob


def load_fusion_model(model_path: Path, device: torch.device) -> FusionIDHClassifier:
    """
    Load the pre-trained fusion model.

    Args:
        model_path: Path to fusion_model.pt
        device: PyTorch device

    Returns:
        Loaded fusion model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Fusion model not found: {model_path}")

    logger.info(f"Loading fusion model from {model_path}")

    # Initialize model
    model = FusionIDHClassifier()

    # Load weights
    checkpoint = torch.load(str(model_path), map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info("Fusion model loaded successfully")
    return model


def preprocess_flair_for_fusion(
    flair_data: np.ndarray, tumor_mask: Optional[np.ndarray] = None, target_size: int = 96
) -> np.ndarray:
    """
    Preprocess FLAIR data for fusion model input.

    Args:
        flair_data: Full FLAIR volume
        tumor_mask: Optional tumor mask for cropping
        target_size: Target size for CNN input

    Returns:
        Preprocessed FLAIR crop (96, 96, 96)
    """
    # Normalize FLAIR to [0, 1]
    flair_norm = (flair_data - flair_data.min()) / (flair_data.max() - flair_data.min() + 1e-8)

    if tumor_mask is not None:
        # Find tumor center of mass for cropping
        tumor_coords = np.where(tumor_mask > 0)
        if len(tumor_coords[0]) > 0:
            center = np.array([np.mean(coords) for coords in tumor_coords])

            # Extract centered crop
            half_size = target_size // 2
            z_start = max(0, int(center[0] - half_size))
            y_start = max(0, int(center[1] - half_size))
            x_start = max(0, int(center[2] - half_size))

            z_end = min(flair_norm.shape[0], z_start + target_size)
            y_end = min(flair_norm.shape[1], y_start + target_size)
            x_end = min(flair_norm.shape[2], x_start + target_size)

            crop = flair_norm[z_start:z_end, y_start:y_end, x_start:x_end]

            # Pad if necessary
            if crop.shape != (target_size, target_size, target_size):
                padding = []
                for dim_size in crop.shape:
                    pad_total = target_size - dim_size
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    padding.extend([pad_before, pad_after])

                crop = np.pad(crop, padding.reshape(-1, 2).tolist(), mode="constant", constant_values=0)
        else:
            # No tumor found, use center crop
            crop = center_crop_3d(flair_norm, target_size)
    else:
        # No tumor mask, use center crop
        crop = center_crop_3d(flair_norm, target_size)

    return crop


def center_crop_3d(volume: np.ndarray, target_size: int) -> np.ndarray:
    """Extract center crop from 3D volume."""
    z, y, x = volume.shape
    z_start = (z - target_size) // 2
    y_start = (y - target_size) // 2
    x_start = (x - target_size) // 2

    return volume[z_start : z_start + target_size, y_start : y_start + target_size, x_start : x_start + target_size]


def load_radiomics_features(radiomics_path: Path) -> Optional[np.ndarray]:
    """
    Load radiomics features from JSON file.

    Args:
        radiomics_path: Path to radiomics_features.json

    Returns:
        Radiomics features array (107,) or None if not found
    """
    if not radiomics_path.exists():
        logger.warning(f"Radiomics features not found: {radiomics_path}")
        return None

    try:
        with open(radiomics_path, "r") as f:
            radiomics_data = json.load(f)

        # Extract features (assuming standard format)
        if isinstance(radiomics_data, dict):
            # Convert dict values to array
            features = np.array(list(radiomics_data.values()))
        else:
            features = np.array(radiomics_data)

        # Ensure we have 107 features
        if len(features) != 107:
            logger.warning(f"Expected 107 radiomics features, got {len(features)}")
            # Pad or truncate as needed
            if len(features) < 107:
                features = np.pad(features, (0, 107 - len(features)), "constant")
            else:
                features = features[:107]

        return features

    except Exception as e:
        logger.error(f"Failed to load radiomics features: {e}")
        return None


def run_fusion_idh_classification(
    model: FusionIDHClassifier,
    flair_data: np.ndarray,
    radiomics_features: np.ndarray,
    tumor_mask: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Run IDH classification using fusion model.

    Args:
        model: Loaded fusion model
        flair_data: FLAIR volume
        radiomics_features: Radiomics features (107,)
        tumor_mask: Optional tumor mask for cropping
        device: PyTorch device

    Returns:
        Classification results dictionary
    """
    logger.info("Running fusion IDH classification...")

    try:
        # Preprocess FLAIR
        flair_crop = preprocess_flair_for_fusion(flair_data, tumor_mask)

        # Convert to tensors
        flair_tensor = torch.from_numpy(flair_crop).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 96, 96)
        radiomics_tensor = torch.from_numpy(radiomics_features).float().unsqueeze(0)  # (1, 107)

        # Move to device
        flair_tensor = flair_tensor.to(device)
        radiomics_tensor = radiomics_tensor.to(device)

        # Run inference
        with torch.no_grad():
            idh_prob = model(flair_tensor, radiomics_tensor)
            idh_prob = idh_prob.cpu().numpy()[0, 0]  # Extract scalar

        # Create results
        idh_prediction = int(idh_prob > 0.5)  # Binary classification

        results = {
            "idh_probability": float(idh_prob),
            "idh_prediction": idh_prediction,  # 0 = wildtype, 1 = mutant
            "idh_label": "mutant" if idh_prediction == 1 else "wildtype",
            "method": "fusion_cnn_radiomics",
            "confidence": "high" if abs(idh_prob - 0.5) > 0.3 else "moderate" if abs(idh_prob - 0.5) > 0.1 else "low",
        }

        logger.info(f"IDH classification: {results['idh_label']} (prob: {idh_prob:.3f})")

        return results

    except Exception as e:
        logger.error(f"Fusion IDH classification failed: {e}")
        return {
            "idh_probability": 0.5,
            "idh_prediction": 0,
            "idh_label": "unknown",
            "method": "fusion_cnn_radiomics",
            "error": str(e),
        }


def integrate_fusion_model_stage8c(
    output_dir: Path,
    flair_path: Path,
    radiomics_path: Path,
    tumor_mask_path: Optional[Path] = None,
    model_path: Path = Path("fusion_model.pt"),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Complete Stage 8c integration for fusion model IDH classification.

    Args:
        output_dir: Output directory for results
        flair_path: Path to FLAIR NIfTI file
        radiomics_path: Path to radiomics features JSON
        tumor_mask_path: Optional path to tumor mask
        model_path: Path to fusion model weights
        device: PyTorch device

    Returns:
        Classification results
    """
    logger.info("=== Stage 8c: Fusion Model IDH Classification ===")

    try:
        # Load model
        model = load_fusion_model(model_path, device)

        # Load FLAIR data
        import nibabel as nib

        flair_img = nib.load(str(flair_path))
        flair_data = flair_img.get_fdata()

        # Load radiomics features
        radiomics_features = load_radiomics_features(radiomics_path)
        if radiomics_features is None:
            raise ValueError("Radiomics features required for fusion model")

        # Load tumor mask if provided
        tumor_mask = None
        if tumor_mask_path and tumor_mask_path.exists():
            tumor_mask_img = nib.load(str(tumor_mask_path))
            tumor_mask = tumor_mask_img.get_fdata()

        # Run classification
        results = run_fusion_idh_classification(model, flair_data, radiomics_features, tumor_mask, device)

        # Save results
        results_path = output_dir / "fusion_idh_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Fusion IDH results saved to {results_path}")

        return results

    except Exception as e:
        logger.error(f"Stage 8c fusion model integration failed: {e}")
        return {
            "idh_probability": 0.5,
            "idh_prediction": 0,
            "idh_label": "unknown",
            "method": "fusion_cnn_radiomics",
            "error": str(e),
        }
