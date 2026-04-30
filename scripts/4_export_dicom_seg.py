#!/usr/bin/env python3
"""
Stage 4 — DICOM-SEG Export
============================
Exports segmentation masks as standards-compliant DICOM-SEG files
for PACS integration and clinical workstation viewing.

Usage:
  python scripts/4_export_dicom_seg.py \\
    --segmentation results/<case>/seg.nii.gz \\
    --source-dicom data/<case>/T1c/ \\
    --output results/<case>/seg.dcm
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.io.dicom_seg_writer import write_dicom_seg, DEFAULT_SEGMENT_DEFINITIONS
from pybrain.io.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export segmentation as DICOM-SEG file"
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
        help="Output path for DICOM-SEG file",
    )
    parser.add_argument(
        "--series-description",
        type=str,
        default="PY-BRAIN BraTS Segmentation (Research Only)",
        help="Series description for DICOM-SEG metadata",
    )
    parser.add_argument(
        "--algorithm-name",
        type=str,
        default="PY-BRAIN v2",
        help="Algorithm name for DICOM-SEG metadata",
    )
    parser.add_argument(
        "--no-disclaimer",
        action="store_true",
        help="Omit research-only disclaimer from series description",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = get_logger("pybrain")
    
    logger.info("=== DICOM-SEG Export ===")
    logger.info(f"Segmentation: {args.segmentation}")
    logger.info(f"Source DICOM: {args.source_dicom}")
    logger.info(f"Output: {args.output}")
    
    # Load segmentation
    logger.info("Loading segmentation NIfTI...")
    seg_img = nib.load(str(args.segmentation))
    seg_data = seg_img.get_fdata().astype(np.int16)
    logger.info(f"Segmentation shape: {seg_data.shape}")
    logger.info(f"Unique labels: {np.unique(seg_data)}")
    
    # Write DICOM-SEG
    output_path = write_dicom_seg(
        segmentation=seg_data,
        source_dicom_dir=args.source_dicom,
        output_path=args.output,
        segment_definitions=DEFAULT_SEGMENT_DEFINITIONS,
        series_description=args.series_description,
        algorithm_name=args.algorithm_name,
        include_disclaimer=not args.no_disclaimer,
    )
    
    logger.info(f"DICOM-SEG export completed: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
