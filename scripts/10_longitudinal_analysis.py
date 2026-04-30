#!/usr/bin/env python3
"""
Longitudinal Analysis
======================
Compares current scan with prior scan to assess tumor response using RANO criteria.

Usage:
  python scripts/10_longitudinal_analysis.py \\
    --current results/BraTS2021_00000/ \\
    --prior results/BraTS2021_00000_prior/ \\
    --output results/BraTS2021_00000/longitudinal/
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.analysis.longitudinal import compare_timepoints
from pybrain.io.config import get_config
from pybrain.io.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Longitudinal analysis for brain tumor response assessment"
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current session directory (containing T1c and segmentation)",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        required=True,
        help="Path to prior session directory (containing T1c and segmentation)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for longitudinal analysis results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: pybrain/config/defaults.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = get_logger("pybrain")
    
    logger.info("=== Longitudinal Analysis ===")
    logger.info(f"Current session: {args.current}")
    logger.info(f"Prior session: {args.prior}")
    logger.info(f"Output: {args.output}")
    
    # Load config
    config = get_config()
    
    # Find required files
    current_t1c = args.current / "T1c.nii.gz"
    current_seg = args.current / "segmentation_full.nii.gz"
    prior_t1c = args.prior / "T1c.nii.gz"
    prior_seg = args.prior / "segmentation_full.nii.gz"
    
    # Validate files exist
    if not current_t1c.exists():
        logger.error(f"Current T1c not found: {current_t1c}")
        return 1
    
    if not current_seg.exists():
        logger.error(f"Current segmentation not found: {current_seg}")
        return 1
    
    if not prior_t1c.exists():
        logger.error(f"Prior T1c not found: {prior_t1c}")
        return 1
    
    if not prior_seg.exists():
        logger.error(f"Prior segmentation not found: {prior_seg}")
        return 1
    
    # Run longitudinal comparison
    result = compare_timepoints(
        current_t1c=current_t1c,
        current_seg=current_seg,
        prior_t1c=prior_t1c,
        prior_seg=prior_seg,
        output_dir=args.output,
        config=config.get("longitudinal", {}),
    )
    
    # Print results
    logger.info("\n=== Longitudinal Results ===")
    logger.info(f"Registration Quality (NMI): {result.registration_quality:.4f}")
    logger.info(f"RANO Assessment: {result.rano_response}")
    logger.info("\nVolume Changes:")
    for region, change in result.volume_changes.items():
        logger.info(
            f"  {region}: {change.prior_cc:.2f} cc → {change.current_cc:.2f} cc "
            f"({change.pct_change:+.1f}%) [{change.status}]"
        )
    
    # Save results to JSON
    results_json = {
        "registration_quality": result.registration_quality,
        "rano_response": result.rano_response,
        "volume_changes": {
            region: {
                "prior_cc": change.prior_cc,
                "current_cc": change.current_cc,
                "abs_change_cc": change.abs_change_cc,
                "pct_change": change.pct_change,
                "status": change.status,
            }
            for region, change in result.volume_changes.items()
        },
        "registered_prior_path": str(result.registered_prior_path),
        "prior_seg_in_current_space_path": str(result.prior_seg_in_current_space_path),
        "overlay_paths": {
            orientation: str(path) for orientation, path in result.overlay_paths.items()
        },
    }
    
    results_path = args.output / "longitudinal_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Registered prior scan: {result.registered_prior_path}")
    logger.info(f"Registered prior segmentation: {result.prior_seg_in_current_space_path}")
    logger.info(f"Overlay images: {list(result.overlay_paths.values())}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
