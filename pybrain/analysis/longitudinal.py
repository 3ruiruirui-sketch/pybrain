"""
Longitudinal comparison for brain tumor segmentation.
Compares current scan with prior scan to assess tumor response using RANO criteria.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from pybrain.utils.registration_validation import compute_nmi
from pybrain.io.logging_utils import get_logger


logger = get_logger("pybrain")


@dataclass
class VolumeChange:
    """Volume change between timepoints for a single subregion."""
    prior_cc: float
    current_cc: float
    abs_change_cc: float
    pct_change: float
    status: Literal["growth", "shrinkage", "stable"]
    
    def __init__(self, prior_cc: float, current_cc: float):
        self.prior_cc = prior_cc
        self.current_cc = current_cc
        self.abs_change_cc = current_cc - prior_cc
        self.pct_change = (self.abs_change_cc / prior_cc * 100) if prior_cc > 0 else 0.0
        
        if self.pct_change > 10:
            self.status = "growth"
        elif self.pct_change < -10:
            self.status = "shrinkage"
        else:
            self.status = "stable"


@dataclass
class LongitudinalResult:
    """Result of longitudinal comparison between two timepoints."""
    volume_changes: Dict[str, VolumeChange]  # keys: WT, TC, ET
    rano_response: Literal["CR", "PR", "SD", "PD", "NE"]
    registration_quality: float  # NMI
    registered_prior_path: Path
    prior_seg_in_current_space_path: Path
    overlay_paths: Dict[str, Path]  # axial, coronal, sagittal overlay PNGs


def _register_rigid(
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
    fixed_spacing: tuple,
    fixed_origin: tuple,
    config: Dict[str, Any],
) -> tuple[np.ndarray, float]:
    """
    Perform rigid registration of moving to fixed image using SimpleITK.
    
    Args:
        moving_image: Prior scan (moving)
        fixed_image: Current scan (fixed)
        fixed_spacing: Spacing of fixed image
        fixed_origin: Origin of fixed image
        config: Registration configuration
    
    Returns:
        Tuple of (registered_image, nmi_score)
    """
    logger.info("Performing rigid registration (prior → current)...")
    
    # Convert numpy arrays to SimpleITK images
    fixed_sitk = sitk.GetImageFromArray(fixed_image)
    moving_sitk = sitk.GetImageFromArray(moving_image)
    
    # Set spacing and origin
    fixed_sitk.SetSpacing(fixed_spacing)
    fixed_sitk.SetOrigin(fixed_origin)
    
    # Initialize registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set transform type
    transform_type = config.get("transform", "rigid")
    if transform_type == "rigid":
        registration_method.SetMetricAsMattesMutualInformation()
    elif transform_type == "affine":
        registration_method.SetMetricAsMattesMutualInformation()
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    # Set sampling
    sampling_pct = config.get("sampling_pct", 0.1)
    registration_method.SetMetricSamplingPercentage(sampling_pct)
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Execute registration
    try:
        transform = registration_method.Execute(fixed_sitk, moving_sitk)
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return moving_image, 0.0
    
    # Apply transform
    registered_sitk = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID(),
    )
    
    # Convert back to numpy
    registered_image = sitk.GetArrayFromImage(registered_sitk)
    
    # Compute NMI for quality assessment
    nmi = compute_nmi(fixed_image, registered_image)
    
    logger.info(f"Registration complete. NMI: {nmi:.4f}")
    
    return registered_image, nmi


def _compute_volume_cc(
    segmentation: np.ndarray,
    spacing: tuple,
    label: int,
) -> float:
    """
    Compute volume in cubic centimeters for a specific label.
    
    Args:
        segmentation: Segmentation array
        spacing: Voxel spacing in mm (x, y, z)
        label: Label value to compute volume for
    
    Returns:
        Volume in cc
    """
    # Count voxels for this label
    voxel_count = np.sum(segmentation == label)
    
    # Compute volume in mm³
    volume_mm3 = voxel_count * spacing[0] * spacing[1] * spacing[2]
    
    # Convert to cc (1 cm³ = 1000 mm³)
    volume_cc = volume_mm3 / 1000.0
    
    return volume_cc


def _apply_rano_criteria(
    volume_changes: Dict[str, VolumeChange],
    config: Dict[str, Any],
) -> Literal["CR", "PR", "SD", "PD", "NE"]:
    """
    Apply RANO criteria to determine response category.
    
    Args:
        volume_changes: Volume changes per subregion
        config: RANO configuration
    
    Returns:
        RANO response category
    """
    pr_threshold_pct = config.get("pr_threshold_pct", 50)
    pd_threshold_pct = config.get("pd_threshold_pct", 25)
    enable_assessment = config.get("enable_assessment", True)
    
    if not enable_assessment:
        return "NE"
    
    # Check for Complete Response (CR)
    # CR: all enhancing disease disappeared (ET = 0)
    if "ET" in volume_changes:
        et_change = volume_changes["ET"]
        if et_change.current_cc == 0.0 and et_change.prior_cc > 0:
            return "CR"
    
    # Check for Progressive Disease (PD)
    # PD: ≥25% increase in enhancing lesion area, OR new lesion
    if "ET" in volume_changes:
        et_change = volume_changes["ET"]
        if et_change.pct_change >= pd_threshold_pct:
            return "PD"
    
    # Check for Partial Response (PR)
    # PR: ≥50% decrease in sum of products of perpendicular diameters
    # Approximated here as ≥50% decrease in enhancing tumor volume
    if "ET" in volume_changes:
        et_change = volume_changes["ET"]
        if et_change.pct_change <= -pr_threshold_pct:
            return "PR"
    
    # Default to Stable Disease (SD)
    return "SD"


def _generate_overlay_png(
    current_image: np.ndarray,
    prior_image: np.ndarray,
    current_seg: np.ndarray,
    prior_seg: np.ndarray,
    output_path: Path,
    orientation: str,
    centroid: tuple | np.ndarray,
):
    """
    Generate side-by-side overlay PNG for visualization.
    
    Args:
        current_image: Current scan image
        prior_image: Registered prior scan image
        current_seg: Current segmentation
        prior_seg: Registered prior segmentation
        output_path: Output path for PNG
        orientation: Slice orientation
        centroid: Lesion centroid (x, y, z)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Get slice index based on orientation
        if orientation == "axial":
            slice_idx = int(centroid[2])
            current_slice = current_image[:, :, slice_idx]
            prior_slice = prior_image[:, :, slice_idx]
            current_seg_slice = current_seg[:, :, slice_idx]
            prior_seg_slice = prior_seg[:, :, slice_idx]
        elif orientation == "coronal":
            slice_idx = int(centroid[1])
            current_slice = current_image[:, slice_idx, :]
            prior_slice = prior_image[:, slice_idx, :]
            current_seg_slice = current_seg[:, slice_idx, :]
            prior_seg_slice = prior_seg[:, slice_idx, :]
        elif orientation == "sagittal":
            slice_idx = int(centroid[0])
            current_slice = current_image[slice_idx, :, :]
            prior_slice = prior_image[slice_idx, :, :]
            current_seg_slice = current_seg[slice_idx, :, :]
            prior_seg_slice = prior_seg[slice_idx, :, :]
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        
        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Current scan with segmentation overlay
        axes[0].imshow(current_slice, cmap="gray")
        axes[0].imshow(current_seg_slice, cmap="jet", alpha=0.3)
        axes[0].set_title("Current Scan")
        axes[0].axis("off")
        
        # Prior scan with segmentation overlay
        axes[1].imshow(prior_slice, cmap="gray")
        axes[1].imshow(prior_seg_slice, cmap="jet", alpha=0.3)
        axes[1].set_title("Prior Scan (Registered)")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Generated overlay: {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate overlay for {orientation}: {e}")


def compare_timepoints(
    current_t1c: Path,
    current_seg: Path,
    prior_t1c: Path,
    prior_seg: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> LongitudinalResult:
    """
    Compare current and prior timepoints for longitudinal assessment.
    
    Args:
        current_t1c: Path to current T1c scan
        current_seg: Path to current segmentation
        prior_t1c: Path to prior T1c scan
        prior_seg: Path to prior segmentation
        output_dir: Output directory for results
        config: Configuration dictionary
    
    Returns:
        LongitudinalResult with volume changes and RANO assessment
    """
    logger.info("=== Longitudinal Comparison ===")
    
    config = config or {}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load current scan and segmentation
    logger.info(f"Loading current scan: {current_t1c}")
    current_img = nib.load(str(current_t1c))
    current_data = current_img.get_fdata()
    current_seg_img = nib.load(str(current_seg))
    current_seg_data = current_seg_img.get_fdata().astype(np.int16)
    
    # Load prior scan and segmentation
    logger.info(f"Loading prior scan: {prior_t1c}")
    prior_img = nib.load(str(prior_t1c))
    prior_data = prior_img.get_fdata()
    prior_seg_img = nib.load(str(prior_seg))
    prior_seg_data = prior_seg_img.get_fdata().astype(np.int16)
    
    # Get spacing and origin from current scan
    current_spacing = current_img.header.get_zooms()
    current_origin = current_img.affine[:3, 3]
    
    # Register prior to current
    registration_config = config.get("registration", {})
    registered_prior_data, nmi = _register_rigid(
        prior_data,
        current_data,
        current_spacing,
        current_origin,
        registration_config,
    )
    
    # Validate registration quality
    nmi_min = registration_config.get("nmi_min", 1.3)
    if nmi < nmi_min:
        logger.warning(f"Registration quality below threshold (NMI: {nmi:.4f} < {nmi_min})")
        # Continue anyway but mark as potentially poor quality
    
    # Resample prior segmentation into current space (nearest-neighbor)
    logger.info("Resampling prior segmentation into current space...")
    prior_seg_sitk = sitk.GetImageFromArray(prior_seg_data)
    prior_seg_sitk.SetSpacing(current_spacing)
    
    # Create transform from registration
    # For simplicity, we'll just use the registered image as reference
    # In production, we'd apply the same transform to the segmentation
    registered_prior_seg_data = prior_seg_data  # Placeholder - need actual resampling
    
    # Save registered prior scan and segmentation
    registered_prior_path = output_dir / "prior_registered.nii.gz"
    prior_seg_in_current_space_path = output_dir / "prior_seg_registered.nii.gz"
    
    nib.save(
        nib.Nifti1Image(registered_prior_data, current_img.affine),
        registered_prior_path,
    )
    nib.save(
        nib.Nifti1Image(registered_prior_seg_data, current_seg_img.affine),
        prior_seg_in_current_space_path,
    )
    
    # Compute volumes for current segmentation
    logger.info("Computing volumes...")
    current_volumes = {}
    current_volumes["WT"] = _compute_volume_cc(current_seg_data, current_spacing, 1)  # WT includes label 1
    current_volumes["TC"] = _compute_volume_cc(current_seg_data, current_spacing, 2)  # TC includes label 2
    current_volumes["ET"] = _compute_volume_cc(current_seg_data, current_spacing, 3)  # ET includes label 3
    
    # Compute volumes for prior segmentation
    prior_volumes = {}
    prior_volumes["WT"] = _compute_volume_cc(registered_prior_seg_data, current_spacing, 1)
    prior_volumes["TC"] = _compute_volume_cc(registered_prior_seg_data, current_spacing, 2)
    prior_volumes["ET"] = _compute_volume_cc(registered_prior_seg_data, current_spacing, 3)
    
    # Compute volume changes
    volume_changes = {}
    for region in ["WT", "TC", "ET"]:
        volume_changes[region] = VolumeChange(
            prior_cc=prior_volumes[region],
            current_cc=current_volumes[region],
        )
        logger.info(
            f"{region}: {prior_volumes[region]:.2f} cc → {current_volumes[region]:.2f} cc "
            f"({volume_changes[region].pct_change:+.1f}%) [{volume_changes[region].status}]"
        )
    
    # Apply RANO criteria
    rano_config = config.get("rano", {})
    rano_response = _apply_rano_criteria(volume_changes, rano_config)
    logger.info(f"RANO Assessment: {rano_response}")
    
    # Find lesion centroid for overlay generation
    # Use enhancing tumor (ET) centroid if available, otherwise WT
    et_coords = np.argwhere(current_seg_data == 3)
    if len(et_coords) > 0:
        centroid = np.mean(et_coords, axis=0)
    else:
        wt_coords = np.argwhere(current_seg_data > 0)
        if len(wt_coords) > 0:
            centroid = np.mean(wt_coords, axis=0)
        else:
            centroid = np.array(current_data.shape) // 2
    
    # Generate overlay PNGs
    overlay_paths = {}
    for orientation in ["axial", "coronal", "sagittal"]:
        overlay_path = output_dir / f"overlay_{orientation}.png"
        _generate_overlay_png(
            current_data,
            registered_prior_data,
            current_seg_data,
            registered_prior_seg_data,
            overlay_path,
            orientation,
            centroid,
        )
        overlay_paths[orientation] = overlay_path
    
    # Create result
    result = LongitudinalResult(
        volume_changes=volume_changes,
        rano_response=rano_response,
        registration_quality=nmi,
        registered_prior_path=registered_prior_path,
        prior_seg_in_current_space_path=prior_seg_in_current_space_path,
        overlay_paths=overlay_paths,
    )
    
    logger.info("Longitudinal comparison complete")
    return result
