# pybrain/io/rtstruct_writer.py
"""
DICOM RTSTRUCT (Radiotherapy Structure Set) writer for exporting segmentation masks
in RTSS format compatible with radiotherapy treatment planning systems.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom

from pybrain.io.logging_utils import get_logger


logger = get_logger("pybrain")


def write_rtstruct(
    segmentation: np.ndarray,
    reference_dicom_dir: Path,
    output_path: Path,
    structure_definitions: List[Dict[str, Any]] | None = None,
) -> Path:
    """
    Write a segmentation mask as a DICOM RTSTRUCT file using rt-utils.
    
    Args:
        segmentation: 3D segmentation array with label values
        reference_dicom_dir: Directory containing reference DICOM series
        output_path: Path for output RTSTRUCT file
        structure_definitions: List of structure definitions (name, label_value, color)
    
    Returns:
        Path to written RTSTRUCT file
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError as e:
        raise ImportError(f"Required dependency missing: {e}. Install rt-utils>=1.2.7")
    
    logger.info(f"Writing RTSTRUCT to {output_path}")
    
    if structure_definitions is None:
        structure_definitions = [
            {"name": "Whole Tumor", "label_value": 1, "color": (255, 0, 0)},
            {"name": "Tumor Core", "label_value": 2, "color": (0, 255, 0)},
            {"name": "Enhancing Tumor", "label_value": 3, "color": (0, 0, 255)},
        ]
    
    # Create RTStruct using rt-utils
    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(reference_dicom_dir))
    
    # Add structures for each label value
    for struct_def in structure_definitions:
        # Create binary mask for this structure
        binary_mask = (segmentation == struct_def["label_value"]).astype(np.uint8)
        
        # rt-utils expects (H, W, slices) ordering
        # If segmentation is (slices, H, W), transpose to (H, W, slices)
        if segmentation.shape[0] < segmentation.shape[2]:
            binary_mask = np.transpose(binary_mask, (1, 2, 0))
        
        # Add ROI to RTSTRUCT
        rtstruct.add_roi(
            mask=binary_mask,
            color=list(struct_def.get("color", (255, 0, 0))),
            name=struct_def["name"],
        )
    
    # Save RTSTRUCT (rt-utils appends .dcm automatically)
    output_path = Path(str(output_path).replace(".dcm", ""))
    rtstruct.save(str(output_path))
    
    logger.info(f"RTSTRUCT written successfully to {output_path}.dcm")
    logger.info(f"  Number of structures: {len(structure_definitions)}")
    
    return Path(str(output_path) + ".dcm")


class RTStructWriter:
    """
    High-level interface for writing RTSTRUCT files using rt-utils.
    """
    
    def __init__(self):
        """Initialize RTSTRUCT writer."""
        pass
    
    def write_from_segmentation(
        self,
        segmentation: np.ndarray,
        reference_dicom_dir: Path,
        output_path: Path,
        structure_definitions: List[Dict[str, Any]] | None = None,
    ) -> Path:
        """
        Write RTSTRUCT from segmentation array.
        
        Args:
            segmentation: 3D segmentation array with label values
            reference_dicom_dir: Directory containing reference DICOM series
            output_path: Path for output RTSTRUCT file
            structure_definitions: List of structure definitions
        
        Returns:
            Path to written RTSTRUCT file
        """
        return write_rtstruct(
            segmentation=segmentation,
            reference_dicom_dir=reference_dicom_dir,
            output_path=output_path,
            structure_definitions=structure_definitions,
        )
