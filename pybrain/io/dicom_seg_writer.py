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
import highdicom as hd
from highdicom.seg import Segmentation, SegmentDescription
from highdicom.seg import SegmentAlgorithmTypeValues, SegmentationTypeValues
from highdicom.sr.coding import CodedConcept

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
    
    # Read source images as pydicom Datasets
    source_images = [pydicom.dcmread(str(p)) for p in source_files]
    
    # Build SegmentDescription objects using real highdicom API
    seg_descriptions = []
    for i, seg_def in enumerate(segment_definitions):
        # Create CodedConcept for property category
        if seg_def.segmented_property_category == "Mass":
            category_code = CodedConcept(
                value="49755003", scheme_designator="SCT", meaning="Morphologically Abnormal Structure"
            )
        elif seg_def.segmented_property_type == "Neoplasm, Primary":
            category_code = CodedConcept(
                value="86049000", scheme_designator="SCT", meaning="Neoplasm, Primary"
            )
        else:
            category_code = CodedConcept(
                value="49755003", scheme_designator="SCT", meaning="Morphologically Abnormal Structure"
            )
        
        # Create CodedConcept for property type
        if seg_def.segmented_property_type == "Mass":
            type_code = CodedConcept(
                value="86049000", scheme_designator="SCT", meaning="Neoplasm, Primary"
            )
        elif seg_def.segmented_property_type == "Tumor Core":
            type_code = CodedConcept(
                value="43134007", scheme_designator="SCT", meaning="Tumor Core"
            )
        else:
            type_code = CodedConcept(
                value="86049000", scheme_designator="SCT", meaning="Neoplasm, Primary"
            )
        
        # Create SegmentDescription
        seg_desc = SegmentDescription(
            segment_number=i + 1,
            segment_label=seg_def.name,
            segmented_property_category=category_code,
            segmented_property_type=type_code,
            algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        )
        
        # Add recommended display color if provided
        if seg_def.color:
            seg_desc.recommended_display_cielab_value = seg_def.color
        
        seg_descriptions.append(seg_desc)
    
    # Ensure segmentation is in correct shape (slices, rows, columns)
    # highdicom expects (slices, rows, columns)
    if segmentation.ndim == 3:
        # If segmentation is (rows, cols, slices), transpose to (slices, rows, cols)
        if segmentation.shape[2] < segmentation.shape[0]:
            segmentation = np.transpose(segmentation, (2, 0, 1))
    
    # Add research-only disclaimer if requested
    if include_disclaimer and "(Research Only)" not in series_description:
        series_description += " (Research Only)"
    
    # Create the Segmentation using real highdicom API
    seg = Segmentation(
        source_images=source_images,
        pixel_array=segmentation,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=seg_descriptions,
        series_instance_uid=hd.UID(),
        series_number=1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
        software_versions=software_versions,
        device_serial_number="PYBRAIN-DEV",
    )
    
    # Save DICOM-SEG file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg.save_as(str(output_path))
    
    logger.info(f"DICOM-SEG written successfully to {output_path}")
    logger.info(f"  Series Description: {series_description}")
    logger.info(f"  Number of segments: {len(seg_descriptions)}")
    
    return output_path
