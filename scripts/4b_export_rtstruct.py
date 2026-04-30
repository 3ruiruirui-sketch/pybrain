#!/usr/bin/env python3
"""
Stage 4b — RT-Struct Export
============================
Exports segmentation masks as DICOM RT-Struct files
for PACS integration and radiation therapy planning.

Usage:
  python scripts/4b_export_rtstruct.py \\
    --segmentation results/<case>/seg.nii.gz \\
    --source-dicom data/<case>/T1c/ \\
    --output results/<case>/rtstruct.dcm
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.io.rtstruct_writer import write_rtstruct
from pybrain.io.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export segmentation as DICOM RT-Struct file"
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        required=True,
        help="Path to segmentation NIfTI file (label values: 1=WT, 2=TC, 3=ET)",
    )
    parser.add_argument(
        "--source-dicom",
        type=Path,
        required=True,
        help="Directory containing source DICOM series",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for RT-Struct file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = get_logger("pybrain")
    
    logger.info("=== RT-Struct Export ===")
    logger.info(f"Segmentation: {args.segmentation}")
    logger.info(f"Source DICOM: {args.source_dicom}")
    logger.info(f"Output: {args.output}")
    
    # Load segmentation
    logger.info("Loading segmentation NIfTI...")
    seg_img = nib.load(str(args.segmentation))
    seg_data = seg_img.get_fdata().astype(np.int16)
    logger.info(f"Segmentation shape: {seg_data.shape}")
    logger.info(f"Unique labels: {np.unique(seg_data)}")
    
    # Write RT-Struct
    output_path = write_rtstruct(
        segmentation=seg_data,
        reference_dicom_dir=args.source_dicom,
        output_path=args.output,
    )
    
    logger.info(f"RT-Struct export completed: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
