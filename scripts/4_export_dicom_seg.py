#!/usr/bin/env python
"""Export a NIfTI segmentation as DICOM-SEG referenced to a source DICOM series.

Usage:
  python scripts/4_export_dicom_seg.py \\
    --segmentation results/<case>/segmentation.nii.gz \\
    --source-dicom data/<case>/T1c/ \\
    --output results/<case>/segmentation.dcm
"""
import argparse
import sys
from pathlib import Path

# Project root on path so the script works without `pip install -e .` 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np

from pybrain.io.dicom_seg_writer import write_dicom_seg, DEFAULT_SEGMENT_DEFINITIONS
from pybrain.io.logging_utils import get_logger


logger = get_logger("scripts.export_dicom_seg")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--segmentation",
        type=Path,
        required=True,
        help="NIfTI segmentation (labels: 1=WT, 2=TC, 3=ET)",
    )
    p.add_argument(
        "--source-dicom",
        type=Path,
        required=True,
        help="Directory containing the source DICOM series (e.g. T1c)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the DICOM-SEG file (.dcm)",
    )
    p.add_argument(
        "--series-description",
        type=str,
        default="PY-BRAIN BraTS Segmentation (Research Only)",
        help="DICOM SeriesDescription for the output",
    )
    p.add_argument(
        "--no-disclaimer",
        action="store_true",
        help="Suppress the research-only disclaimer in metadata (NOT recommended)",
    )
    args = p.parse_args()

    if not args.segmentation.exists():
        logger.error(f"Segmentation not found: {args.segmentation}")
        return 1
    if not args.source_dicom.exists() or not args.source_dicom.is_dir():
        logger.error(f"Source DICOM directory not found: {args.source_dicom}")
        return 1

    logger.info(f"Loading segmentation from {args.segmentation}")
    seg = nib.load(str(args.segmentation)).get_fdata().astype(np.uint8)
    logger.info(f"Segmentation shape: {seg.shape}, unique labels: {sorted(np.unique(seg).tolist())}")

    out = write_dicom_seg(
        segmentation=seg,
        source_dicom_dir=args.source_dicom,
        output_path=args.output,
        segment_definitions=DEFAULT_SEGMENT_DEFINITIONS,
        series_description=args.series_description,
        include_disclaimer=not args.no_disclaimer,
    )
    logger.info(f"DICOM-SEG written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
