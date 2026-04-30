#!/usr/bin/env python3
"""
Brain Metastases Analysis Script
================================
2-stage pipeline for brain metastases detection and segmentation.
Parallel to scripts/3_brain_tumor_analysis.py but for mets workflow.

Usage:
  python scripts/3b_mets_analysis.py \
    --t1c path/to/T1c.nii.gz \
    --t1 path/to/T1.nii.gz \
    --t2 path/to/T2.nii.gz \
    --flair path/to/FLAIR.nii.gz \
    --output-dir results/mets_case \
    --config pybrain/config/defaults.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

from pybrain.config.config import load_config
from pybrain.io.logging_utils import setup_logging, get_logger
from pybrain.io.dicom import load_nifti
from pybrain.utils.brain_extraction import extract_brain_mask
from pybrain.analysis.mets_pipeline import run_mets_analysis, generate_mets_report
from pybrain.utils.save_nifti import save_nifti


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain metastases detection and segmentation"
    )
    parser.add_argument(
        "--t1c",
        type=Path,
        required=True,
        help="T1 post-contrast NIfTI file",
    )
    parser.add_argument(
        "--t1",
        type=Path,
        required=True,
        help="T1 pre-contrast NIfTI file",
    )
    parser.add_argument(
        "--t2",
        type=Path,
        required=True,
        help="T2 NIfTI file",
    )
    parser.add_argument(
        "--flair",
        type=Path,
        required=True,
        help="FLAIR NIfTI file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: pybrain/config/defaults.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--brain-mask",
        type=Path,
        default=None,
        help="Pre-computed brain mask (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = get_logger("pybrain")

    logger.info("=== Brain Metastases Analysis ===")

    # Load config
    config = load_config(args.config)

    # Check if mets is enabled
    mets_config = config.get("mets", {})
    if not mets_config.get("enabled", False):
        logger.error("Mets analysis is not enabled in config. Set mets.enabled: true")
        return 1

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load images
    logger.info("Loading images...")
    t1c_img = load_nifti(args.t1c)
    t1_img = load_nifti(args.t1)
    t2_img = load_nifti(args.t2)
    flair_img = load_nifti(args.flair)

    t1c = t1c_img.get_fdata()
    t1 = t1_img.get_fdata()
    t2 = t2_img.get_fdata()
    flair = flair_img.get_fdata()

    # Get spacing
    spacing = t1c_img.header.get_zooms()[:3]
    config["spacing"] = spacing
    logger.info(f"Image spacing: {spacing}")

    # Extract brain mask if not provided
    if args.brain_mask:
        logger.info(f"Loading pre-computed brain mask from {args.brain_mask}")
        brain_mask_img = load_nifti(args.brain_mask)
        brain_mask = brain_mask_img.get_fdata() > 0
    else:
        logger.info("Extracting brain mask...")
        brain_mask = extract_brain_mask(t1, t1c, t2, flair)

    # Run mets analysis
    logger.info("Running mets analysis pipeline...")
    result = run_mets_analysis(
        t1c=t1c,
        t1=t1,
        t2=t2,
        flair=flair,
        brain_mask=brain_mask,
        config=config,
        device=args.device,
    )

    # Save combined segmentation
    seg_path = output_dir / "mets_segmentation.nii.gz"
    save_nifti(result.combined_segmentation, t1c_img, seg_path)
    logger.info(f"Saved combined segmentation to {seg_path}")

    # Save per-lesion segmentations
    for lesion in result.lesions:
        lesion_path = output_dir / f"lesion_{lesion.id:03d}.nii.gz"
        # Create a full-size volume with just this lesion
        lesion_full = np.zeros_like(result.combined_segmentation)
        lesion_full[lesion.segmentation > 0] = lesion.segmentation[lesion.segmentation > 0]
        save_nifti(lesion_full, t1c_img, lesion_path)

    # Generate report
    report = generate_mets_report(result)
    report_path = output_dir / "mets_report.json"
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to {report_path}")

    # Print summary
    logger.info("=" * 50)
    logger.info("Mets Analysis Summary")
    logger.info("=" * 50)
    logger.info(f"Total lesions: {result.total_lesion_count}")
    logger.info(f"Total volume: {result.total_lesion_volume_cc:.2f} cc")
    logger.info(f"Detection method: {result.detection_method}")
    logger.info(f"Segmentation method: {result.segmentation_method}")
    logger.info("")
    logger.info("Lesions:")
    for lesion in result.lesions:
        logger.info(
            f"  {lesion.id}: {lesion.location} - {lesion.volume_cc:.2f} cc (conf: {lesion.confidence:.2f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
