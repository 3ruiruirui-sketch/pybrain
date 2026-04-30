"""
Enhanced Test-Time Augmentation for Brain Tumor Segmentation
Implements advanced TTA with rotation, scaling, and intensity variations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pybrain.io.logging_utils import get_logger

logger = get_logger("models.enhanced_tta")


class EnhancedTTA:
    """
    Enhanced Test-Time Augmentation with multiple transformation types.
    Includes flips, rotations, scaling, and intensity variations.
    """

    def __init__(
        self,
        enable_flips: bool = True,
        enable_rotations: bool = True,
        enable_scaling: bool = True,
        enable_intensity: bool = True,
        rotation_angles: List[int] = [90, 180, 270],
        scale_factors: List[float] = [0.9, 1.1],
    ):
        """
        Initialize enhanced TTA.

        Args:
            enable_flips: Enable axial, coronal, sagittal flips
            enable_rotations: Enable in-plane rotations
            enable_scaling: Enable scaling transformations
            enable_intensity: Enable intensity augmentations
            rotation_angles: List of rotation angles in degrees
            scale_factors: List of scaling factors
        """
        self.enable_flips = enable_flips
        self.enable_rotations = enable_rotations
        self.enable_scaling = enable_scaling
        self.enable_intensity = enable_intensity
        self.rotation_angles = rotation_angles
        self.scale_factors = scale_factors

        # Generate transformation list
        self.transforms = self._generate_transforms()

    def _generate_transforms(self) -> List[Dict]:
        """Generate list of transformations to apply."""
        transforms = []

        # Original (no transformation)
        transforms.append({"name": "original", "transform": lambda x: x, "inverse": lambda x: x, "weight": 1.0})

        # Flips
        if self.enable_flips:
            # Axial flip
            transforms.append(
                {
                    "name": "flip_axial",
                    "transform": lambda x: torch.flip(x, dims=[2]),
                    "inverse": lambda x: torch.flip(x, dims=[-3]),
                    "weight": 0.9,
                }
            )

            transforms.append(
                {
                    "name": "flip_coronal",
                    "transform": lambda x: torch.flip(x, dims=[3]),
                    "inverse": lambda x: torch.flip(x, dims=[-2]),
                    "weight": 0.9,
                }
            )

            transforms.append(
                {
                    "name": "flip_sagittal",
                    "transform": lambda x: torch.flip(x, dims=[4]),
                    "inverse": lambda x: torch.flip(x, dims=[-1]),
                    "weight": 0.9,
                }
            )

        # Rotations (in-plane, axial view)
        if self.enable_rotations:
            for angle in self.rotation_angles:
                transforms.append(
                    {
                        "name": f"rot_{angle}",
                        "transform": lambda x, a=angle: self._rotate_3d(x, a),
                        "inverse": lambda x, a=angle: (
                            self._rotate_3d(x.unsqueeze(0) if x.ndim == 4 else x, -a).squeeze(0)
                            if (x.ndim == 4)
                            else self._rotate_3d(x, -a)
                        ),
                        "weight": 0.8,
                    }
                )

        # Scaling
        if self.enable_scaling:
            for scale in self.scale_factors:
                transforms.append(
                    {
                        "name": f"scale_{scale}",
                        "transform": lambda x, s=scale: self._scale_3d(x, s),
                        "inverse": lambda x, s=scale: self._scale_3d(x, 1.0 / s),
                        "weight": 0.7,
                    }
                )

        # Intensity variations
        # INPUT DOMAIN: tensor is z-scored MRI (mean≈0, std≈1, range≈[-4,+4]).
        # Do NOT clamp to [0,1] — that destroys negative-valued background voxels.
        # Brightness = additive DC shift (±0.1 in z-score units).
        # Contrast   = multiplicative std scaling (×1.2 / ×0.8); centre is 0.
        if self.enable_intensity:
            transforms.append(
                {"name": "brightness_up", "transform": lambda x: x + 0.1, "inverse": lambda x: x - 0.1, "weight": 0.6}
            )

            transforms.append(
                {"name": "brightness_down", "transform": lambda x: x - 0.1, "inverse": lambda x: x + 0.1, "weight": 0.6}
            )

            transforms.append(
                {"name": "contrast_up", "transform": lambda x: x * 1.2, "inverse": lambda x: x / 1.2, "weight": 0.6}
            )

            transforms.append(
                {"name": "contrast_down", "transform": lambda x: x * 0.8, "inverse": lambda x: x / 0.8, "weight": 0.6}
            )

        logger.info(f"Enhanced TTA: Generated {len(transforms)} transformations")
        return transforms

    def _rotate_3d(self, x: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate 3D tensor around the axial (D) axis by a multiple of 90 degrees.

        Input/output shape: (1, C, D, H, W).
        rot90 on the H-W plane is only lossless when H == W; for BraTS volumes
        the axial plane is 240×240, so this holds.  Non-square inputs raise.
        """
        # Normalise to [0, 360) so that -90 → 270, -180 → 180 etc.
        angle_norm = angle % 360
        k = angle_norm // 90  # number of 90-degree CCW turns (0,1,2,3)

        if k == 0:
            return x

        H, W = x.shape[3], x.shape[4]
        if H != W:
            raise ValueError(
                f"_rotate_3d requires square H×W slices, got H={H}, W={W}. Disable rotations for non-square inputs."
            )

        # Iterate over D slices: x[0, :, i, :, :] → (C, H, W)
        # rot90 on dims [-2,-1]: (C, H, W) → (C, H, W) when H==W
        # stack on dim=1 rebuilds D: list of n*(C,H,W) → (C, D, H, W)
        # unsqueeze(0) restores batch: (1, C, D, H, W)
        rotated_slices = []
        for i in range(x.shape[2]):
            rotated_slices.append(torch.rot90(x[0, :, i, :, :], k=k, dims=[-2, -1]))

        return torch.stack(rotated_slices, dim=1).unsqueeze(0)

    def _scale_3d(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale 3D volume by `scale` then crop/pad to restore original shape.

        Shape contract: input (1, C, D, H, W) → output (1, C, D, H, W).
        Scaling is trilinear; symmetric crop (scale>1) or zero-pad (scale<1)
        restores the original spatial extent so ensemble accumulation stays
        spatially aligned.
        """
        if scale == 1.0:
            return x

        # F.interpolate trilinear requires 5D (N,C,D,H,W).
        # _scale_3d is called on both 5D forward inputs and 4D inverse outputs
        # (after apply_tta squeezes the batch dim).  Normalise here.
        squeezed = x.ndim == 4
        x5 = x.unsqueeze(0) if squeezed else x

        original_shape = x5.shape[2:]  # (D, H, W)
        scaled = torch.nn.functional.interpolate(
            x5,
            scale_factor=scale,
            mode="trilinear",
            align_corners=False,
        )
        scaled_shape = scaled.shape[2:]  # may differ from original after int rounding

        out = torch.zeros_like(x5)  # always 5D to match scaled

        # Compute crop/pad slices for each spatial dim
        slices_out = [slice(None), slice(None)]  # batch + channel
        slices_scaled = [slice(None), slice(None)]
        for orig, sc in zip(original_shape, scaled_shape):
            if sc >= orig:
                # crop: take centre `orig` voxels from scaled
                start = (sc - orig) // 2
                slices_scaled.append(slice(start, start + orig))
                slices_out.append(slice(0, orig))
            else:
                # pad: place scaled in centre of zero output
                start = (orig - sc) // 2
                slices_scaled.append(slice(0, sc))
                slices_out.append(slice(start, start + sc))

        out[tuple(slices_out)] = scaled[tuple(slices_scaled)]
        return out.squeeze(0) if squeezed else out

    def apply_tta(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: torch.device,
        inference_fn: Callable,
        sw_config: Optional[Dict] = None,
    ) -> Tuple[Optional[torch.Tensor], List[str]]:
        """
        Apply enhanced test-time augmentation.

        Args:
            model: PyTorch model
            input_tensor: Input tensor (1, C, D, H, W)
            device: Device for computation
            inference_fn: Function to run inference
            sw_config: Sliding window configuration

        Returns:
            ensemble_output: TTA ensemble output
            applied_transforms: List of applied transformation names
        """
        model.eval()
        ensemble_output = None
        total_weight = 0.0
        applied_transforms = []

        logger.info(f"Applying enhanced TTA with {len(self.transforms)} transforms...")

        with torch.no_grad():
            for i, transform_info in enumerate(self.transforms):
                try:
                    # Apply transformation
                    transform_fn = transform_info["transform"]
                    weight = transform_info["weight"]

                    transformed_input = transform_fn(input_tensor.to(device))

                    # Run inference
                    if sw_config:
                        output = inference_fn(model, transformed_input, device, sw_config)
                    else:
                        output = inference_fn(model, transformed_input, device)

                    # TYPE CONTRACT: inference_fn may return np.ndarray or torch.Tensor.
                    # Inverse lambdas (torch.flip, _rotate_3d) require a torch.Tensor.
                    # Normalise to tensor here; remove batch dim if present.
                    if isinstance(output, np.ndarray):
                        output = torch.from_numpy(output)
                    if isinstance(output, torch.Tensor) and output.ndim == 5:
                        output = output.squeeze(0)  # (1,C,D,H,W) → (C,D,H,W)

                    # Apply inverse transformation
                    inverse_fn = transform_info["inverse"]
                    output = inverse_fn(output)

                    # Add to ensemble
                    if ensemble_output is None:
                        ensemble_output = weight * output
                    else:
                        ensemble_output += weight * output

                    total_weight += weight
                    applied_transforms.append(transform_info["name"])

                    if (i + 1) % 5 == 0:
                        logger.debug(f"TTA progress: {i + 1}/{len(self.transforms)} completed")

                except Exception as e:
                    logger.warning(f"TTA transform {transform_info['name']} failed: {e}")
                    continue

        # Normalize by total weight
        if ensemble_output is not None and total_weight > 0:
            ensemble_output = ensemble_output / total_weight

        logger.info(f"Enhanced TTA completed: {len(applied_transforms)}/{len(self.transforms)} successful")

        return ensemble_output, applied_transforms


def run_enhanced_tta_ensemble(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    sw_config: Dict,
    tta_config: Optional[Dict] = None,
) -> np.ndarray:
    """
    Run enhanced TTA ensemble for a model.

    Args:
        model: PyTorch model
        input_tensor: Input tensor
        device: Device for computation
        sw_config: Sliding window configuration
        tta_config: TTA configuration

    Returns:
        TTA ensemble probability map
    """
    logger = get_logger("models.enhanced_tta")

    if tta_config is None:
        tta_config = {}

    # Initialize enhanced TTA
    tta = EnhancedTTA(
        enable_flips=tta_config.get("enable_flips", True),
        enable_rotations=tta_config.get("enable_rotations", True),
        enable_scaling=tta_config.get("enable_scaling", False),  # Disabled by default due to complexity
        enable_intensity=tta_config.get("enable_intensity", False),  # Disabled by default
        rotation_angles=tta_config.get("rotation_angles", [90, 180, 270]),
        scale_factors=tta_config.get("scale_factors", [0.9, 1.1]),
    )

    # Define inference function
    def inference_fn(model, input_tensor, device, sw_config):
        from monai.inferers.inferer import SlidingWindowInferer
        from typing import cast as _cast

        # SegResNet bundle expects [T1c, T1, T2, FLAIR].
        # Pipeline stacks [FLAIR, T1, T1c, T2] → permute indices (2, 1, 3, 0).
        _input = input_tensor[:, (2, 1, 3, 0), :, :, :].clone()

        roi_size = tuple(sw_config.get("roi_size", (128, 128, 128)))
        sw_batch_size = sw_config.get("sw_batch_size", 1)
        overlap = sw_config.get("overlap", 0.5)

        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian")

        logits = _cast(torch.Tensor, inferer(_input, model))
        probs = torch.sigmoid(logits.cpu())
        return probs

    # Apply TTA
    ensemble_output, applied_transforms = tta.apply_tta(model, input_tensor, device, inference_fn, sw_config)

    if ensemble_output is not None:
        # ensemble_output is (C, D, H, W) — batch dim already removed in apply_tta.
        # Do NOT index [0] here: that would select only channel-0 (TC), silently
        # discarding WT (ch-1) and ET (ch-2).
        from monai.utils.type_conversion import convert_to_numpy

        result = convert_to_numpy(ensemble_output)
        logger.info(f"Enhanced TTA ensemble completed with transforms: {applied_transforms}")
        return result
    else:
        logger.error("Enhanced TTA failed completely")
        raise RuntimeError("Enhanced TTA failed")


def validate_tta_config(config: Dict) -> bool:
    """
    Validate TTA configuration.

    Args:
        config: TTA configuration

    Returns:
        bool: True if configuration is valid
    """
    if not config.get("enabled", False):
        return True

    # Check rotation angles
    rotation_angles = config.get("rotation_angles", [90, 180, 270])
    for angle in rotation_angles:
        if angle not in [90, 180, 270]:
            logger.warning(f"Rotation angle {angle} may not be supported efficiently")

    # Check scale factors
    scale_factors = config.get("scale_factors", [0.9, 1.1])
    for scale in scale_factors:
        if scale < 0.5 or scale > 2.0:
            logger.warning(f"Scale factor {scale} is extreme, may cause issues")

    return True
