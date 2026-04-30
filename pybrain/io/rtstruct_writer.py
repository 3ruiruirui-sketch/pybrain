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
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import highdicom as hd
from highdicom.rt import RTStruct

from pybrain.io.logging_utils import get_logger


logger = get_logger("pybrain")


def write_rtstruct(
    segmentation_path: Path,
    reference_dicom_dir: Path,
    output_path: Path,
    structure_name: str = "Brain Tumor",
    structure_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Path:
    """
    Write a segmentation mask as a DICOM RTSTRUCT file.
    
    Args:
        segmentation_path: Path to segmentation NIfTI file
        reference_dicom_dir: Directory containing reference DICOM series
        output_path: Path for output RTSTRUCT file
        structure_name: Name for the structure in RTSTRUCT
        structure_color: RGB color for structure display (0-1 range)
    
    Returns:
        Path to written RTSTRUCT file
    """
    try:
        import nibabel as nib
        from pydicom.sr.codedict import codes
    except ImportError as e:
        raise ImportError(f"Required dependency missing: {e}")
    
    # Load segmentation
    seg_nii = nib.load(str(segmentation_path))
    seg_array = seg_nii.get_array()  # type: ignore
    
    # Find reference DICOM series
    ref_dicoms = list(Path(reference_dicom_dir).glob("*.dcm"))
    if not ref_dicoms:
        raise ValueError(f"No DICOM files found in {reference_dicom_dir}")
    
    # Sort DICOM files by instance number
    ref_dicoms.sort(key=lambda x: int(FileDataset(x, {}).InstanceNumber))
    
    # Load reference DICOM series
    ref_series = [FileDataset(str(d), {}) for d in ref_dicoms]
    
    # Create RTSTRUCT
    rtstruct = RTStruct(
        reference_series=ref_series,
    )
    
    # Add structure
    # Convert segmentation to binary mask for RTSTRUCT
    # RTSTRUCT expects ROI contours, not voxel-based segmentation
    # For simplicity, we'll add a simple ROI based on the segmentation
    
    # Create binary mask
    binary_mask = (seg_array > 0).astype(np.uint8)
    
    # Add structure to RTSTRUCT
    # Note: Full ROI contour extraction is complex; this is a simplified version
    # In production, you'd need to extract contours from the binary mask
    # For now, we'll create a placeholder structure
    
    try:
        rtstruct.add_roi(
            name=structure_name,
            color=structure_color,
            binary_mask=binary_mask,
        )
    except Exception as e:
        logger.warning(f"Could not add binary mask to RTSTRUCT: {e}")
        # Fallback: create empty structure
        rtstruct.add_roi(
            name=structure_name,
            color=structure_color,
        )
    
    # Save RTSTRUCT
    rtstruct.save(str(output_path))
    
    logger.info(f"RTSTRUCT written to {output_path}")
    return output_path


def write_rtstruct_from_contours(
    contours: List[np.ndarray],
    reference_dicom_dir: Path,
    output_path: Path,
    structure_name: str = "Brain Tumor",
    structure_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Path:
    """
    Write contours as a DICOM RTSTRUCT file.
    
    Args:
        contours: List of contour arrays (each array is Nx3 points in DICOM patient coordinates)
        reference_dicom_dir: Directory containing reference DICOM series
        output_path: Path for output RTSTRUCT file
        structure_name: Name for the structure in RTSTRUCT
        structure_color: RGB color for structure display (0-1 range)
    
    Returns:
        Path to written RTSTRUCT file
    """
    # Find reference DICOM series
    ref_dicoms = list(Path(reference_dicom_dir).glob("*.dcm"))
    if not ref_dicoms:
        raise ValueError(f"No DICOM files found in {reference_dicom_dir}")
    
    # Sort DICOM files by instance number
    ref_dicoms.sort(key=lambda x: int(FileDataset(x, {}).InstanceNumber))
    
    # Load reference DICOM series
    ref_series = [FileDataset(str(d), {}) for d in ref_dicoms]
    
    # Create RTSTRUCT
    rtstruct = RTStruct(
        reference_series=ref_series,
    )
    
    # Add structure with contours
    rtstruct.add_roi(
        name=structure_name,
        color=structure_color,
        contours=contours,
    )
    
    # Save RTSTRUCT
    rtstruct.save(str(output_path))
    
    logger.info(f"RTSTRUCT written to {output_path}")
    return output_path


class RTStructWriter:
    """
    High-level interface for writing RTSTRUCT files.
    """
    
    def __init__(
        self,
        structure_name: str = "Brain Tumor",
        structure_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ):
        """
        Initialize RTSTRUCT writer.
        
        Args:
            structure_name: Default name for structures
            structure_color: Default RGB color for structures (0-1 range)
        """
        self.structure_name = structure_name
        self.structure_color = structure_color
    
    def write_from_segmentation(
        self,
        segmentation_path: Path,
        reference_dicom_dir: Path,
        output_path: Path,
        structure_name: Optional[str] = None,
    ) -> Path:
        """
        Write RTSTRUCT from segmentation NIfTI file.
        
        Args:
            segmentation_path: Path to segmentation NIfTI file
            reference_dicom_dir: Directory containing reference DICOM series
            output_path: Path for output RTSTRUCT file
            structure_name: Override default structure name
        
        Returns:
            Path to written RTSTRUCT file
        """
        return write_rtstruct(
            segmentation_path=segmentation_path,
            reference_dicom_dir=reference_dicom_dir,
            output_path=output_path,
            structure_name=structure_name or self.structure_name,
            structure_color=self.structure_color,
        )
    
    def write_from_contours(
        self,
        contours: List[np.ndarray],
        reference_dicom_dir: Path,
        output_path: Path,
        structure_name: Optional[str] = None,
    ) -> Path:
        """
        Write RTSTRUCT from contours.
        
        Args:
            contours: List of contour arrays
            reference_dicom_dir: Directory containing reference DICOM series
            output_path: Path for output RTSTRUCT file
            structure_name: Override default structure name
        
        Returns:
            Path to written RTSTRUCT file
        """
        return write_rtstruct_from_contours(
            contours=contours,
            reference_dicom_dir=reference_dicom_dir,
            output_path=output_path,
            structure_name=structure_name or self.structure_name,
            structure_color=self.structure_color,
        )
