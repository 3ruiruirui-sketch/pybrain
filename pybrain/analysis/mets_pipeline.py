"""
Brain metastases analysis pipeline.
2-stage workflow: detection → per-lesion segmentation.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from pybrain.models.mets_detector import (
    LesionCandidate,
    detect_lesions,
    _get_lesion_location,
    compute_lesion_statistics,
)
from pybrain.models.mets_segmenter import (
    segment_all_lesions,
    load_mets_segmenter,
    combine_lesion_segmentations,
)
from pybrain.io.logging_utils import get_logger

logger = get_logger("pybrain")


@dataclass
class Lesion:
    """A segmented lesion with metadata."""
    id: int
    centroid: tuple[int, int, int]  # z, y, x
    location: str  # Anatomical location (e.g., "Right Frontal Lobe")
    volume_cc: float
    surface_area_mm2: float
    confidence: float  # Detection confidence
    segmentation: np.ndarray = field(repr=False)  # Binary mask


@dataclass
class MetsResult:
    """Result of brain metastases analysis."""
    lesions: List[Lesion]
    total_lesion_count: int
    total_lesion_volume_cc: float
    combined_segmentation: np.ndarray = field(repr=False)
    detection_method: str
    segmentation_method: str
    longitudinal_ready: bool = True  # Can be used for longitudinal comparison


def run_mets_analysis(
    t1c: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    brain_mask: np.ndarray,
    config: dict,
    device: str = "cuda",
) -> MetsResult:
    """
    Run the 2-stage brain metastases analysis pipeline.

    Args:
        t1c: T1 post-contrast image (z, y, x)
        t1: T1 pre-contrast image (z, y, x)
        t2: T2 image (z, y, x)
        flair: FLAIR image (z, y, x)
        brain_mask: Brain mask (z, y, x)
        config: Configuration dictionary with mets settings
        device: Device to run inference on

    Returns:
        MetsResult with detected and segmented lesions
    """
    logger.info("Starting brain metastases analysis")

    # Extract mets config
    mets_config = config.get("mets", {})
    detection_config = mets_config.get("detection", {})
    seg_config = mets_config.get("segmentation", {})

    # Add spacing to detection config
    spacing = config.get("spacing", (1.0, 1.0, 1.0))
    detection_config["spacing"] = spacing

    # Stage A: Detection
    logger.info("Stage A: Lesion detection")
    candidates = detect_lesions(t1c, brain_mask, detection_config)

    if len(candidates) == 0:
        logger.warning("No lesions detected")
        return MetsResult(
            lesions=[],
            total_lesion_count=0,
            total_lesion_volume_cc=0.0,
            combined_segmentation=np.zeros_like(t1c, dtype=np.uint8),
            detection_method=detection_config.get("method", "fallback_threshold"),
            segmentation_method="none",
        )

    # Stage B: Per-lesion segmentation
    logger.info(f"Stage B: Segmenting {len(candidates)} lesions")

    # Load segmentation model
    model_path = Path(seg_config.get("model_path", "models/mets_bundle/segmenter.pth"))
    try:
        model = load_mets_segmenter(model_path, device=device)
    except Exception as exc:
        logger.error(f"Failed to load segmentation model: {exc}")
        # Return detection-only result
        return MetsResult(
            lesions=[],
            total_lesion_count=len(candidates),
            total_lesion_volume_cc=sum(c.volume_cc for c in candidates),
            combined_segmentation=np.zeros_like(t1c, dtype=np.uint8),
            detection_method=detection_config.get("method", "fallback_threshold"),
            segmentation_method="failed",
        )

    # Segment all lesions
    patch_size = tuple(seg_config.get("patch_size", [64, 64, 64]))
    segmentations = segment_all_lesions(
        candidates=candidates,
        t1c=t1c,
        t1=t1,
        t2=t2,
        flair=flair,
        model=model,
        patch_size=patch_size,
        device=device,
    )

    # Combine segmentations into full volume
    combined_seg = combine_lesion_segmentations(
        segmentations=segmentations,
        candidates=candidates,
        image_shape=t1c.shape,
    )

    # Compute lesion statistics and create Lesion objects
    lesions = []
    total_volume = 0.0

    for i, (candidate, seg) in enumerate(zip(candidates, segmentations)):
        # Compute statistics
        stats = compute_lesion_statistics(seg, spacing)

        # Get anatomical location
        location = _get_lesion_location(candidate.centroid, t1c.shape, spacing)

        lesion = Lesion(
            id=i + 1,
            centroid=candidate.centroid,
            location=location,
            volume_cc=stats["volume_cc"],
            surface_area_mm2=stats["surface_area_mm2"],
            confidence=candidate.confidence,
            segmentation=seg,
        )
        lesions.append(lesion)
        total_volume += stats["volume_cc"]

    logger.info(f"Mets analysis complete: {len(lesions)} lesions, total volume: {total_volume:.2f} cc")

    return MetsResult(
        lesions=lesions,
        total_lesion_count=len(lesions),
        total_lesion_volume_cc=total_volume,
        combined_segmentation=combined_seg,
        detection_method=detection_config.get("method", "fallback_threshold"),
        segmentation_method="segresnet",
    )


def classify_analysis_mode(
    t1c: np.ndarray,
    brain_mask: np.ndarray,
    config: dict,
) -> str:
    """
    Classify whether to use glioma or mets analysis mode.

    Heuristic: many small lesions → mets; single large lesion → glioma

    Args:
        t1c: T1 post-contrast image (z, y, x)
        brain_mask: Brain mask (z, y, x)
        config: Configuration dictionary

    Returns:
        "glioma", "mets", or "auto" (if classification uncertain)
    """
    from pybrain.models.mets_detector import detect_lesions

    # Use fallback detection to count candidate lesions
    detection_config = config.get("mets", {}).get("detection", {})
    detection_config["spacing"] = config.get("spacing", (1.0, 1.0, 1.0))
    detection_config["method"] = "fallback_threshold"

    candidates = detect_lesions(t1c, brain_mask, detection_config)

    if len(candidates) == 0:
        return "glioma"  # Default to glioma if no lesions detected

    # Count lesions with significant volume
    min_volume_cc = config.get("mets", {}).get("detection", {}).get("min_lesion_volume_cc", 0.05)
    significant_lesions = [c for c in candidates if c.volume_cc >= min_volume_cc]

    # Heuristics
    if len(significant_lesions) >= 3:
        # Multiple lesions → likely mets
        return "mets"
    elif len(significant_lesions) == 1:
        # Single lesion → likely glioma
        # But check if it's very large (could be glioma)
        max_volume = max(c.volume_cc for c in significant_lesions)
        if max_volume > 50.0:  # > 50cc is very large, likely glioma
            return "glioma"
        else:
            return "auto"  # Uncertain, let user decide
    else:
        # 2 lesions - uncertain
        return "auto"


def generate_mets_report(result: MetsResult) -> Dict[str, Any]:
    """
    Generate a summary report for mets analysis.

    Args:
        result: MetsResult from run_mets_analysis

    Returns:
        Dictionary with summary statistics
    """
    return {
        "total_lesion_count": result.total_lesion_count,
        "total_lesion_volume_cc": result.total_lesion_volume_cc,
        "detection_method": result.detection_method,
        "segmentation_method": result.segmentation_method,
        "lesions": [
            {
                "id": lesion.id,
                "centroid": lesion.centroid,
                "location": lesion.location,
                "volume_cc": lesion.volume_cc,
                "surface_area_mm2": lesion.surface_area_mm2,
                "confidence": lesion.confidence,
            }
            for lesion in result.lesions
        ],
    }
