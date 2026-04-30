# pybrain/io/dicom_seg_writer.py
"""
DICOM Segmentation (DICOM-SEG) writer for exporting segmentation masks
in standards-compliant format for PACS and clinical workstations.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import highdicom as hd
from highdicom.seg import SegmentationDataset

from pybrain.io.logging_utils import get_logger


logger = get_logger("pybrain")


@dataclass
class SegmentDef:
    """Definition for a single segment in DICOM-SEG."""
    name: str
    label_value: int
    algorithm_name: str = "PY-BRAIN v2"
    algorithm_type: str = "AUTOMATIC"
    segmented_property_category: str = "Anatomical Structure"
    segmented_property_type: str = "Mass"
    tracking_uid: str | None = None
    tracking_id: str | None = None
    color: tuple[int, int, int] | None = None


# Default segment definitions for BraTS classes using SNOMED-CT codes
DEFAULT_SEGMENT_DEFINITIONS = [
    SegmentDef(
        name="Whole Tumor",
        label_value=1,
        segmented_property_type="Mass",
        tracking_uid=str(uuid.uuid4()),
        tracking_id="WT",
        color=(255, 0, 0),  # Red
    ),
    SegmentDef(
        name="Tumor Core",
        label_value=2,
        segmented_property_type="Neoplasm, Primary",
        tracking_uid=str(uuid.uuid4()),
        tracking_id="TC",
        color=(0, 255, 0),  # Green
    ),
    SegmentDef(
        name="Enhancing Tumor",
        label_value=3,
        segmented_property_type="Tumor Core",
        tracking_uid=str(uuid.uuid4()),
        tracking_id="ET",
        color=(0, 0, 255),  # Blue
    ),
]


def write_dicom_seg(
    segmentation: np.ndarray,
    source_dicom_dir: Path,
    output_path: Path,
    segment_definitions: List[SegmentDef] | None = None,
    patient_metadata: Dict[str, Any] | None = None,
    series_description: str = "PY-BRAIN BraTS Segmentation (Research Only)",
    algorithm_name: str = "PY-BRAIN v2",
    manufacturer: str = "PY-BRAIN",
    manufacturer_model_name: str = "py-brain v2",
    software_versions: str = "2.0.0",
    include_disclaimer: bool = True,
) -> Path:
    """
    Write a segmentation mask as a DICOM-SEG file.
    
    Args:
        segmentation: 3D segmentation array with label values (1=WT, 2=TC, 3=ET)
        source_dicom_dir: Directory containing source DICOM series
        output_path: Output path for DICOM-SEG file
        segment_definitions: List of SegmentDef objects (defaults to BraTS WT/TC/ET)
        patient_metadata: Optional patient metadata dict
        series_description: Series description for the DICOM-SEG
        algorithm_name: Algorithm name for metadata
        manufacturer: Manufacturer name
        manufacturer_model_name: Manufacturer model name
        software_versions: Software version
        include_disclaimer: Whether to include research-only disclaimer
    
    Returns:
        Path to the written DICOM-SEG file
    """
    logger.info(f"Writing DICOM-SEG to {output_path}")
    
    if segment_definitions is None:
        segment_definitions = DEFAULT_SEGMENT_DEFINITIONS
        logger.info("Using default BraTS segment definitions (WT/TC/ET)")
    
    # Load source DICOM series to get metadata
    source_files = sorted(source_dicom_dir.glob("*.dcm"))
    if not source_files:
        raise ValueError(f"No DICOM files found in {source_dicom_dir}")
    
    logger.info(f"Found {len(source_files)} DICOM files in source directory")
    
    # Read first DICOM to get basic metadata
    first_ds = pydicom.dcmread(str(source_files[0]))
    
    # Create segmentation segments using highdicom's CodeSequence
    from highdicom.sr import CodeSequence
    
    segments = []
    for seg_def in segment_definitions:
        # Create binary mask for this segment
        seg_mask = (segmentation == seg_def.label_value).astype(np.uint8)
        
        # Create segment attributes as a dict
        segment_attributes = {
            "SegmentNumber": len(segments) + 1,
            "SegmentLabel": seg_def.name,
            "SegmentDescription": f"{seg_def.name} segmentation",
            "SegmentAlgorithmType": seg_def.algorithm_type,
            "SegmentAlgorithmName": seg_def.algorithm_name,
            "SegmentedPropertyCategoryCodeSequence": CodeSequence(
                value=seg_def.segmented_property_category,
                scheme_designator="SCT",
                meaning=seg_def.segmented_property_category,
            ),
            "SegmentedPropertyTypeCodeSequence": CodeSequence(
                value=seg_def.segmented_property_type,
                scheme_designator="SCT",
                meaning=seg_def.segmented_property_type,
            ),
        }
        
        # Add tracking IDs if provided
        if seg_def.tracking_id:
            segment_attributes["TrackingID"] = seg_def.tracking_id
            segment_attributes["TrackingUID"] = seg_def.tracking_uid or str(uuid.uuid4())
        
        # Add recommended display color if provided
        if seg_def.color:
            segment_attributes["RecommendedDisplayCIELabValue"] = seg_def.color
        
        segments.append(segment_attributes)
    
    # Create DICOM-SEG dataset using highdicom
    seg_ds = SegmentationDataset(
        source_images=source_files,
        segmentation=segmentation,
        segment_attributes=segments,
        series_description=series_description,
        series_number=1,
        instance_number=1,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
        software_versions=software_versions,
    )
    
    # Add research-only disclaimer if requested
    if include_disclaimer:
        if not "(Research Only)" in seg_ds.SeriesDescription:
            seg_ds.SeriesDescription += " (Research Only)"
    
    # Save DICOM-SEG file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_ds.save_as(str(output_path))
    
    logger.info(f"DICOM-SEG written successfully to {output_path}")
    logger.info(f"  Series Description: {seg_ds.SeriesDescription}")
    logger.info(f"  Number of segments: {len(segments)}")
    
    return output_path
