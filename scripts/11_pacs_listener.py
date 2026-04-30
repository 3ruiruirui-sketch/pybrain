#!/usr/bin/env python3
"""
PACS Listener Daemon
======================
Long-running daemon that listens for incoming DICOM studies from PACS,
triages them based on configured criteria, and triggers the PY-BRAIN pipeline.

Usage:
  python scripts/11_pacs_listener.py --config pybrain/config/defaults.yaml
"""

import argparse
import signal
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.io.config import get_config
from pybrain.io.pacs_listener import PACSListener
from pybrain.io.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="PACS listener daemon for automatic study triage and pipeline triggering"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: pybrain/config/defaults.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for received DICOM files (default: results/pacs)",
    )
    return parser.parse_args()


def pipeline_handler(dicom_dir: Path, study_metadata: dict):
    """
    Callback to trigger the PY-BRAIN pipeline for an accepted study.

    Args:
        dicom_dir: Directory containing DICOM files for the study
        study_metadata: Study metadata (patient ID, study date, etc.)
    """
    logger = get_logger("pybrain")
    
    try:
        from pybrain.pipeline import run
        
        # Build assignments from DICOM directory
        # This assumes the DICOM files are organized by series description
        assignments = {}
        
        # Find DICOM files by series description
        import pydicom
        from collections import defaultdict
        
        series_files = defaultdict(list)
        for dcm_file in dicom_dir.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                series_desc = str(ds.get("SeriesDescription", "")).lower()
                series_files[series_desc].append(dcm_file)
            except Exception:
                continue
        
        # Map series descriptions to PY-BRAIN roles
        desc_to_role = {
            "t1": "T1",
            "t1c": "T1c",
            "t1c+": "T1c",
            "t1c+gd": "T1c",
            "t2": "T2",
            "flair": "FLAIR",
        }
        
        for desc, files in series_files.items():
            for role in desc_to_role:
                if role in desc:
                    assignments[desc_to_role[role]] = str(files[0].parent)
                    break
        
        if not assignments:
            logger.warning(f"No matching series found in {dicom_dir}")
            return
        
        # Create output directory
        study_uid = study_metadata.get("study_uid", "unknown")
        patient_id = study_metadata.get("patient_id", "unknown")
        output_dir = Path("results") / f"{patient_id}_{study_uid}"
        
        logger.info(f"Triggering pipeline for study {study_uid}")
        logger.info(f"Assignments: {assignments}")
        
        # Run pipeline
        result = run(
            assignments=assignments,
            output_dir=output_dir,
            export_dicom_seg=True,
            source_dicom_dir=dicom_dir,
            export_dicom_sr=True,
        )
        
        logger.info(f"Pipeline completed for study {study_uid}")
        
    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()
    setup_logging()
    logger = get_logger("pybrain")
    
    logger.info("=== PACS Listener Daemon ===")
    
    # Load config
    config = get_config()
    
    # Check if PACS is enabled
    pacs_config = config.get("pacs", {})
    if not pacs_config.get("enabled", False):
        logger.error("PACS is not enabled in config. Set pacs.enabled: true")
        return 1
    
    # Determine output directory
    output_dir = args.output_dir or Path("results/pacs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Local AE Title: {pacs_config.get('local_ae_title', 'PYBRAIN')}")
    logger.info(f"Peer: {pacs_config.get('peer', {}).get('ae_title', 'ORTHANC')}@{pacs_config.get('peer', {}).get('host', 'localhost')}:{pacs_config.get('peer', {}).get('port', 4242)}")
    logger.info(f"Triage rules: {pacs_config.get('triage', {})}")
    
    # Create listener
    listener = PACSListener(
        config=config,
        output_dir=output_dir,
        pipeline_handler=pipeline_handler,
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        listener.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start listener
    listener.start()
    
    # Run forever
    try:
        listener.run_forever()
    except Exception as exc:
        logger.error(f"Listener failed: {exc}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
