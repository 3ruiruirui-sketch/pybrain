"""
PACS listener daemon for automatic study triage and pipeline triggering.
Listens for C-STORE requests, filters incoming studies, and triggers analysis.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Set, Optional, Callable
from collections import defaultdict

from pybrain.io.pacs_client import PACSClient
from pybrain.io.logging_utils import get_logger

logger = get_logger("pybrain")


class StudyAccumulator:
    """Accumulates DICOM instances by study until complete."""

    def __init__(self):
        self.studies: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "instances": [],
                "modalities": set(),
                "series_descriptions": set(),
                "metadata": {},
            }
        )
        self.lock = threading.Lock()

    def add_instance(self, study_uid: str, dicom_path: Path, metadata: Dict[str, Any]) -> None:
        """Add a DICOM instance to a study."""
        with self.lock:
            study = self.studies[study_uid]
            study["instances"].append(dicom_path)
            study["modalities"].add(metadata.get("modality", ""))
            study["series_descriptions"].add(metadata.get("series_description", ""))
            if not study["metadata"]:
                study["metadata"] = metadata

    def get_study(self, study_uid: str) -> Optional[Dict[str, Any]]:
        """Get study data if exists."""
        with self.lock:
            return self.studies.get(study_uid)

    def remove_study(self, study_uid: str) -> None:
        """Remove study from accumulator."""
        with self.lock:
            self.studies.pop(study_uid, None)


class PACSListener:
    """
    Long-running daemon that listens for C-STORE, triages incoming studies,
    and triggers the pipeline when a study matches configured criteria.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        pipeline_handler: Optional[Callable[[Path, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize PACS listener.

        Args:
            config: PACS configuration from defaults.yaml
            output_dir: Directory for received DICOM files
            pipeline_handler: Callback to trigger pipeline for accepted studies
        """
        self.config = config
        self.output_dir = output_dir
        self.pipeline_handler = pipeline_handler

        self.accumulator = StudyAccumulator()
        self.running = False
        self.shutdown_event = threading.Event()

        # Initialize PACS client for C-STORE SCP
        pacs_config = config.get("pacs", {})
        self.client = PACSClient(
            ae_title=pacs_config.get("local_ae_title", "PYBRAIN"),
            peer_ae_title=pacs_config.get("peer", {}).get("ae_title", "ORTHANC"),
            peer_host=pacs_config.get("peer", {}).get("host", "localhost"),
            peer_port=pacs_config.get("peer", {}).get("port", 4242),
            local_port=pacs_config.get("local_port", 11112),
        )

        # Triage rules
        self.triage_config = pacs_config.get("triage", {})
        self.required_modalities = set(
            self.triage_config.get("require_modalities", ["MR"])
        )
        self.required_series_descriptions = set(
            self.triage_config.get("require_series_descriptions", ["T1", "T1c", "T2", "FLAIR"])
        )
        self.auto_route_back = self.triage_config.get("auto_route_back", True)
        self.output_destination_ae = self.triage_config.get(
            "output_destination_ae", "ORTHANC"
        )

        logger.info(f"PACSListener initialized with triage rules: {self.triage_config}")

    def _matches_triage_rules(self, study_uid: str) -> bool:
        """
        Check if a study matches triage criteria.

        Args:
            study_uid: Study Instance UID

        Returns:
            True if study matches criteria, False otherwise
        """
        study = self.accumulator.get_study(study_uid)
        if not study:
            return False

        study_modalities = study["modalities"]
        study_series = study["series_descriptions"]

        # Check modality requirement
        if not self.required_modalities.issubset(study_modalities):
            logger.info(
                f"Study {study_uid} rejected: missing modalities "
                f"(required: {self.required_modalities}, found: {study_modalities})"
            )
            return False

        # Check series description requirement
        if not self.required_series_descriptions.issubset(study_series):
            logger.info(
                f"Study {study_uid} rejected: missing series descriptions "
                f"(required: {self.required_series_descriptions}, found: {study_series})"
            )
            return False

        logger.info(f"Study {study_uid} matches triage criteria")
        return True

    def _handle_received_dicom(self, dicom_path: Path, metadata: Dict[str, Any]) -> None:
        """
        Handle a received DICOM file from C-STORE SCP.

        Args:
            dicom_path: Path to received DICOM file
            metadata: DICOM metadata
        """
        study_uid = metadata.get("study_uid", "")
        if not study_uid:
            logger.warning(f"Received DICOM without StudyInstanceUID: {dicom_path}")
            return

        logger.info(f"Received DICOM for study {study_uid}: {dicom_path.name}")

        # Add to accumulator
        self.accumulator.add_instance(study_uid, dicom_path, metadata)

        # Check if study is complete (heuristic: all required series received)
        study = self.accumulator.get_study(study_uid)
        if self._matches_triage_rules(study_uid):
            # Trigger pipeline
            self._trigger_pipeline(study_uid)

    def _trigger_pipeline(self, study_uid: str) -> None:
        """
        Trigger pipeline for an accepted study.

        Args:
            study_uid: Study Instance UID
        """
        study = self.accumulator.get_study(study_uid)
        if not study:
            logger.error(f"Study {study_uid} not found in accumulator")
            return

        logger.info(f"Triggering pipeline for study {study_uid}")

        # Create study-specific output directory
        study_output_dir = self.output_dir / study_uid
        study_output_dir.mkdir(parents=True, exist_ok=True)

        # Move DICOM files to study directory
        for dicom_path in study["instances"]:
            try:
                import shutil
                shutil.move(str(dicom_path), str(study_output_dir / dicom_path.name))
            except Exception as exc:
                logger.error(f"Failed to move {dicom_path}: {exc}")

        # Call pipeline handler if provided
        if self.pipeline_handler:
            try:
                self.pipeline_handler(study_output_dir, study["metadata"])
            except Exception as exc:
                logger.error(f"Pipeline handler failed: {exc}")

        # Auto-route results back to PACS if enabled
        if self.auto_route_back:
            self._route_results_back(study_output_dir)

        # Remove from accumulator
        self.accumulator.remove_study(study_uid)

    def _route_results_back(self, study_output_dir: Path) -> None:
        """
        Route analysis results back to PACS via C-STORE.

        Args:
            study_output_dir: Study output directory containing results
        """
        try:
            # Find result DICOM files (SEG, SR, etc.)
            result_files = list(study_output_dir.glob("*.dcm"))
            if not result_files:
                logger.info(f"No DICOM results to route back for {study_output_dir}")
                return

            # Store each result file
            success_count = 0
            for result_file in result_files:
                if self.client.c_store(result_file):
                    success_count += 1

            logger.info(
                f"Routed {success_count}/{len(result_files)} results back to PACS"
            )
        except Exception as exc:
            logger.error(f"Failed to route results back: {exc}")

    def start(self) -> None:
        """Start the PACS listener daemon."""
        if self.running:
            logger.warning("PACSListener already running")
            return

        self.running = True
        self.shutdown_event.clear()

        # Start storage SCP in background
        self.client.start_storage_scp(
            handler=self._handle_received_dicom,
            output_dir=self.output_dir / "incoming",
        )

        logger.info("PACSListener started")

    def stop(self) -> None:
        """Stop the PACS listener daemon gracefully."""
        if not self.running:
            return

        logger.info("Stopping PACSListener...")
        self.running = False
        self.shutdown_event.set()

        # Note: The storage SCP thread is daemon and will exit when main thread exits
        logger.info("PACSListener stopped")

    def run_forever(self) -> None:
        """Run the listener until interrupted."""
        self.start()
        try:
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.stop()
