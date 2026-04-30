"""
Brain metastases detection and segmentation tests.
Tests with synthetic phantoms to validate detection and volume accuracy.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple

from pybrain.models.mets_detector import (
    LesionCandidate,
    detect_lesions,
    _get_lesion_location,
    compute_lesion_statistics,
)
from pybrain.models.mets_segmenter import _extract_patch
from pybrain.analysis.mets_pipeline import run_mets_analysis, classify_analysis_mode


def create_spherical_phantom(
    shape: Tuple[int, int, int],
    sphere_centers: List[Tuple[int, int, int]],
    sphere_radii: List[int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Create a synthetic phantom with spherical lesions.

    Args:
        shape: Image shape (z, y, x)
        sphere_centers: List of (z, y, x) centers
        sphere_radii: List of sphere radii in voxels
        spacing: Image spacing in mm

    Returns:
        3D image array with spherical lesions
    """
    image = np.zeros(shape, dtype=np.float32)

    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]

    for center, radius in zip(sphere_centers, sphere_radii):
        cz, cy, cx = center
        distance = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
        image[distance <= radius] = 1.0

    return image


def create_brain_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    """Create a simple brain mask (ellipsoid)."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_z, center_y, center_x = shape[0] / 2, shape[1] / 2, shape[2] / 2
    radius_z, radius_y, radius_x = shape[0] / 2 - 5, shape[1] / 2 - 5, shape[2] / 2 - 5

    distance = np.sqrt(
        ((z - center_z) / radius_z) ** 2 +
        ((y - center_y) / radius_y) ** 2 +
        ((x - center_x) / radius_x) ** 2
    )
    mask = (distance <= 1.0).astype(np.uint8)
    return mask


def test_detect_lesions_synthetic_phantom():
    """Test detection on synthetic phantom with known number of lesions."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create phantom with 3 spherical lesions
    sphere_centers = [(20, 20, 20), (32, 32, 32), (44, 44, 44)]
    sphere_radii = [3, 4, 3]
    image = create_spherical_phantom(shape, sphere_centers, sphere_radii)
    brain_mask = create_brain_mask(shape)

    config = {
        "method": "fallback_threshold",
        "min_lesion_volume_cc": 0.01,
        "confidence_threshold": 0.3,
        "max_lesions": 50,
        "spacing": spacing,
    }

    candidates = detect_lesions(image, brain_mask, config)

    # Should detect 3 lesions
    assert len(candidates) == 3, f"Expected 3 lesions, got {len(candidates)}"

    # Check volumes are reasonable
    for candidate in candidates:
        assert candidate.volume_cc > 0
        assert candidate.confidence > 0


def test_detect_lesions_no_lesions():
    """Test detection on phantom with no lesions."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create empty phantom
    image = np.zeros(shape, dtype=np.float32)
    brain_mask = create_brain_mask(shape)

    config = {
        "method": "fallback_threshold",
        "min_lesion_volume_cc": 0.01,
        "confidence_threshold": 0.3,
        "max_lesions": 50,
        "spacing": spacing,
    }

    candidates = detect_lesions(image, brain_mask, config)

    # Should detect 0 lesions
    assert len(candidates) == 0


def test_lesion_volume_accuracy():
    """Test per-lesion volume within 10% of ground truth."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create phantom with 1 sphere of known radius
    sphere_centers = [(32, 32, 32)]
    sphere_radii = [5]  # 5 voxels radius
    image = create_spherical_phantom(shape, sphere_centers, sphere_radii)
    brain_mask = create_brain_mask(shape)

    config = {
        "method": "fallback_threshold",
        "min_lesion_volume_cc": 0.01,
        "confidence_threshold": 0.3,
        "max_lesions": 50,
        "spacing": spacing,
    }

    candidates = detect_lesions(image, brain_mask, config)

    assert len(candidates) == 1

    # Ground truth volume: (4/3) * pi * r^3 in voxels
    gt_volume_voxels = (4.0 / 3.0) * np.pi * (5.0 ** 3)
    gt_volume_cc = gt_volume_voxels * spacing[0] * spacing[1] * spacing[2] / 1000.0

    detected_volume = candidates[0].volume_cc

    # Check within 10% of ground truth
    relative_error = abs(detected_volume - gt_volume_cc) / gt_volume_cc
    assert relative_error < 0.1, f"Volume error {relative_error:.2%} exceeds 10%"


def test_extract_patch():
    """Test patch extraction around centroid."""
    shape = (64, 64, 64)
    image = np.random.rand(*shape).astype(np.float32)
    centroid = (32, 32, 32)
    patch_size = (16, 16, 16)

    patch = _extract_patch(image, centroid, patch_size)

    assert patch.shape == patch_size
    assert patch.min() >= 0
    assert patch.max() <= 1


def test_extract_patch_near_boundary():
    """Test patch extraction near image boundary (should pad)."""
    shape = (64, 64, 64)
    image = np.random.rand(*shape).astype(np.float32)
    centroid = (5, 5, 5)  # Near corner
    patch_size = (32, 32, 32)

    patch = _extract_patch(image, centroid, patch_size)

    # Should still be full patch size due to padding
    assert patch.shape == patch_size


def test_lesion_location():
    """Test anatomical location determination."""
    centroid = (32, 16, 48)  # z=32, y=16 (frontal), x=48 (right)
    image_shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    location = _get_lesion_location(centroid, image_shape, spacing)

    # Should be Right Frontal Lobe
    assert "Right" in location
    assert "Frontal" in location
    assert "Lobe" in location


def test_lesion_statistics():
    """Test lesion statistics computation."""
    shape = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)

    # Create simple spherical segmentation
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = (16, 16, 16)
    radius = 5
    distance = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    seg = (distance <= radius).astype(np.uint8)

    stats = compute_lesion_statistics(seg, spacing)

    assert stats["volume_cc"] > 0
    assert stats["centroid"] == center
    assert stats["surface_area_mm2"] > 0


def test_classify_analysis_mode_multi_lesion():
    """Test auto mode classifies multi-lesion phantom as mets."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create phantom with 4 lesions (should be mets)
    sphere_centers = [(16, 16, 16), (24, 24, 24), (32, 32, 32), (40, 40, 40)]
    sphere_radii = [3, 3, 3, 3]
    t1c = create_spherical_phantom(shape, sphere_centers, sphere_radii)
    brain_mask = create_brain_mask(shape)

    config = {
        "mets": {
            "detection": {
                "min_lesion_volume_cc": 0.01,
            }
        },
        "spacing": spacing,
    }

    mode = classify_analysis_mode(t1c, brain_mask, config)

    # Should classify as mets
    assert mode == "mets", f"Expected 'mets', got '{mode}'"


def test_classify_analysis_mode_single_lesion():
    """Test auto mode classifies single large lesion as glioma."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create phantom with 1 large lesion (should be glioma)
    sphere_centers = [(32, 32, 32)]
    sphere_radii = [20]  # Large lesion
    t1c = create_spherical_phantom(shape, sphere_centers, sphere_radii)
    brain_mask = create_brain_mask(shape)

    config = {
        "mets": {
            "detection": {
                "min_lesion_volume_cc": 0.01,
            }
        },
        "spacing": spacing,
    }

    mode = classify_analysis_mode(t1c, brain_mask, config)

    # Should classify as glioma
    assert mode == "glioma", f"Expected 'glioma', got '{mode}'"


def test_run_mets_analysis_synthetic():
    """Test end-to-end mets analysis on synthetic data."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create synthetic data
    sphere_centers = [(20, 20, 20), (32, 32, 32), (44, 44, 44)]
    sphere_radii = [3, 4, 3]
    t1c = create_spherical_phantom(shape, sphere_centers, sphere_radii)
    t1 = np.random.rand(*shape).astype(np.float32)
    t2 = np.random.rand(*shape).astype(np.float32)
    flair = np.random.rand(*shape).astype(np.float32)
    brain_mask = create_brain_mask(shape)

    config = {
        "mets": {
            "enabled": True,
            "detection": {
                "method": "fallback_threshold",
                "min_lesion_volume_cc": 0.01,
                "confidence_threshold": 0.3,
                "max_lesions": 50,
            },
            "segmentation": {
                "patch_size": [32, 32, 32],
                "model_path": "models/mets_bundle/segmenter.pth",
            },
        },
        "spacing": spacing,
        "hardware": {"device": "cpu"},
    }

    result = run_mets_analysis(
        t1c=t1c,
        t1=t1,
        t2=t2,
        flair=flair,
        brain_mask=brain_mask,
        config=config,
        device="cpu",
    )

    # Check result structure
    assert result.total_lesion_count >= 0
    assert result.total_lesion_volume_cc >= 0
    assert result.detection_method == "fallback_threshold"
    assert result.longitudinal_ready is True


def test_lesion_candidate_dataclass():
    """Test LesionCandidate dataclass."""
    candidate = LesionCandidate(
        centroid=(32, 32, 32),
        bbox=((20, 20, 20), (44, 44, 44)),
        peak_intensity=0.8,
        volume_cc=1.5,
        confidence=0.9,
        spacing=(1.0, 1.0, 1.0),
    )

    assert candidate.centroid == (32, 32, 32)
    assert candidate.volume_cc == 1.5
    assert candidate.confidence == 0.9


def test_detect_lesions_max_lesions_limit():
    """Test that max_lesions parameter limits returned candidates."""
    shape = (64, 64, 64)
    spacing = (1.0, 1.0, 1.0)

    # Create phantom with 10 lesions
    sphere_centers = [(i * 6, i * 6, i * 6) for i in range(10)]
    sphere_radii = [2] * 10
    image = create_spherical_phantom(shape, sphere_centers, sphere_radii)
    brain_mask = create_brain_mask(shape)

    config = {
        "method": "fallback_threshold",
        "min_lesion_volume_cc": 0.01,
        "confidence_threshold": 0.3,
        "max_lesions": 5,  # Limit to 5
        "spacing": spacing,
    }

    candidates = detect_lesions(image, brain_mask, config)

    # Should return at most 5 lesions
    assert len(candidates) <= 5
