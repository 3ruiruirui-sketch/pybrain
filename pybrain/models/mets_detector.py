"""
Brain metastases detection module.
Stage A of the 2-stage mets pipeline: candidate detection on T1c.
"""

import numpy as np
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging

from pybrain.io.logging_utils import get_logger

logger = get_logger("pybrain")


@dataclass
class LesionCandidate:
    """A detected lesion candidate from the detection stage."""
    centroid: Tuple[int, int, int]  # z, y, x coordinates
    bbox: Tuple[Tuple[int, int, int], Tuple[int, int, int]]  # ((z1,y1,x1), (z2,y2,x2))
    peak_intensity: float
    volume_cc: float
    confidence: float
    spacing: Tuple[float, float, float]  # (z, y, x) spacing in mm


def detect_lesions(
    image: np.ndarray,
    brain_mask: np.ndarray,
    config: dict,
) -> List[LesionCandidate]:
    """
    Detect lesion candidates on T1c post-contrast image.

    Args:
        image: 3D T1c image array (z, y, x)
        brain_mask: 3D brain mask (z, y, x), same shape as image
        config: Detection configuration dictionary with keys:
            - method: "nndetection" | "retinanet" | "fallback_threshold"
            - min_lesion_volume_cc: Minimum lesion volume in cc
            - confidence_threshold: Confidence threshold for filtering
            - max_lesions: Maximum number of lesions to return
            - spacing: Image spacing (z, y, x) in mm

    Returns:
        List of LesionCandidate objects sorted by confidence (descending)
    """
    method = config.get("method", "fallback_threshold")
    spacing = config.get("spacing", (1.0, 1.0, 1.0))
    min_volume_cc = config.get("min_lesion_volume_cc", 0.05)
    confidence_threshold = config.get("confidence_threshold", 0.5)
    max_lesions = config.get("max_lesions", 50)

    logger.info(f"Detecting lesions using method: {method}")

    if method == "nndetection":
        candidates = _detect_nndetection(image, brain_mask, config)
    elif method == "retinanet":
        candidates = _detect_retinanet(image, brain_mask, config)
    elif method == "fallback_threshold":
        candidates = _detect_fallback_threshold(image, brain_mask, config)
    else:
        raise ValueError(f"Unknown detection method: {method}")

    # Filter by volume
    candidates = [c for c in candidates if c.volume_cc >= min_volume_cc]

    # Filter by confidence
    candidates = [c for c in candidates if c.confidence >= confidence_threshold]

    # Sort by confidence (descending)
    candidates.sort(key=lambda x: x.confidence, reverse=True)

    # Limit to max lesions
    candidates = candidates[:max_lesions]

    logger.info(f"Detected {len(candidates)} lesion candidates")
    return candidates


def _detect_nndetection(
    image: np.ndarray,
    brain_mask: np.ndarray,
    config: dict,
) -> List[LesionCandidate]:
    """
    Detect lesions using nnDetection (preferred method).

    Note: This requires nnDetection to be installed and a pretrained model
    from BraTS-Mets to be available. Falls back to threshold if unavailable.
    """
    try:
        from nnDetection import NNDetector  # type: ignore
    except ImportError:
        logger.warning("nnDetection not available, falling back to threshold method")
        return _detect_fallback_threshold(image, brain_mask, config)

    # TODO: Implement nnDetection inference
    # For now, fall back to threshold
    logger.warning("nnDetection inference not yet implemented, using fallback")
    return _detect_fallback_threshold(image, brain_mask, config)


def _detect_retinanet(
    image: np.ndarray,
    brain_mask: np.ndarray,
    config: dict,
) -> List[LesionCandidate]:
    """
    Detect lesions using 3D RetinaNet via MONAI.

    Note: This requires a pretrained RetinaNet model.
    Falls back to threshold if unavailable.
    """
    try:
        from monai.networks.nets import RetinaNet  # type: ignore
    except ImportError:
        logger.warning("MONAI RetinaNet not available, falling back to threshold method")
        return _detect_fallback_threshold(image, brain_mask, config)

    # TODO: Implement RetinaNet inference
    # For now, fall back to threshold
    logger.warning("RetinaNet inference not yet implemented, using fallback")
    return _detect_fallback_threshold(image, brain_mask, config)


def _detect_fallback_threshold(
    image: np.ndarray,
    brain_mask: np.ndarray,
    config: dict,
) -> List[LesionCandidate]:
    """
    Fallback detection using intensity threshold + connected components.
    Uses the enhancing tumor probability map from existing models.
    """
    from scipy import ndimage
    from skimage.measure import label, regionprops

    # Normalize image to [0, 1]
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Apply brain mask
    image_masked = image_norm * brain_mask

    # Use intensity threshold (enhancing regions are bright on T1c)
    # Percentile-based threshold to account for contrast variations
    threshold = np.percentile(image_masked[brain_mask > 0], 95)
    binary = (image_masked > threshold).astype(np.uint8)

    # Remove very small regions (noise)
    min_size = config.get("min_voxels", 10)
    labeled = label(binary)
    for region in regionprops(labeled):
        if region.area < min_size:
            labeled[labeled == region.label] = 0

    # Extract connected components as candidates
    candidates = []
    spacing = config.get("spacing", (1.0, 1.0, 1.0))
    voxel_volume_cc = spacing[0] * spacing[1] * spacing[2] / 1000.0

    for region in regionprops(labeled):
        if region.area < min_size:
            continue

        # Centroid (z, y, x)
        centroid = tuple(map(int, region.centroid))

        # Bounding box
        bbox = region.bbox  # (z1, y1, x1, z2, y2, x2)
        bbox_tuple = ((bbox[0], bbox[1], bbox[2]), (bbox[3], bbox[4], bbox[5]))

        # Peak intensity
        coords = region.coords
        peak_intensity = image_norm[tuple(coords.T)].max()

        # Volume in cc
        volume_cc = region.area * voxel_volume_cc

        # Confidence based on intensity above threshold
        confidence = min(1.0, (peak_intensity - threshold) / (1.0 - threshold + 1e-8))

        candidate = LesionCandidate(
            centroid=centroid,
            bbox=bbox_tuple,
            peak_intensity=peak_intensity,
            volume_cc=volume_cc,
            confidence=confidence,
            spacing=spacing,
        )
        candidates.append(candidate)

    return candidates


def _get_lesion_location(
    centroid: Tuple[int, int, int],
    image_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
) -> str:
    """
    Determine anatomical location of a lesion (lobe + structure).

    Args:
        centroid: Lesion centroid (z, y, x)
        image_shape: Image shape (z, y, x)
        spacing: Image spacing (z, y, x) in mm

    Returns:
        String describing location (e.g., "Right Frontal Lobe")
    """
    # Simplified heuristic based on hemisphere and vertical position
    z, y, x = centroid
    depth, height, width = image_shape

    # Hemisphere (left/right)
    hemisphere = "Left" if x < width / 2 else "Right"

    # Vertical position (frontal/parietal/temporal/occipital)
    # This is a simplified approximation
    y_ratio = y / height
    if y_ratio < 0.33:
        region = "Frontal"
    elif y_ratio < 0.66:
        region = "Parietal"
    else:
        region = "Occipital"

    return f"{hemisphere} {region} Lobe"


def compute_lesion_statistics(
    segmentation: np.ndarray,
    spacing: Tuple[float, float, float],
) -> dict:
    """
    Compute statistics for a lesion segmentation.

    Args:
        segmentation: Binary segmentation mask (z, y, x)
        spacing: Image spacing (z, y, x) in mm

    Returns:
        Dictionary with lesion statistics
    """
    from scipy import ndimage

    voxel_volume_cc = spacing[0] * spacing[1] * spacing[2] / 1000.0
    voxel_count = np.sum(segmentation > 0)
    volume_cc = voxel_count * voxel_volume_cc

    # Centroid
    labeled, num_features = ndimage.label(segmentation)
    if num_features == 0:
        return {"volume_cc": 0.0, "centroid": (0, 0, 0), "surface_area_mm2": 0.0}

    # Get the largest component
    sizes = ndimage.sum(segmentation, labeled, range(num_features + 1))
    max_label = sizes[1:].argmax() + 1
    largest = (labeled == max_label).astype(np.uint8)

    # Centroid of largest component
    centroid = tuple(map(int, ndimage.center_of_mass(largest)))

    # Surface area (approximate)
    # Using marching cubes or simple voxel face counting
    # For simplicity, use voxel face counting
    surface_voxels = ndimage.binary_dilation(largest) ^ largest
    surface_area_voxels = np.sum(surface_voxels)
    surface_area_mm2 = surface_area_voxels * (spacing[0] * spacing[1] + spacing[0] * spacing[2] + spacing[1] * spacing[2]) / 3

    return {
        "volume_cc": volume_cc,
        "centroid": centroid,
        "surface_area_mm2": surface_area_mm2,
    }
