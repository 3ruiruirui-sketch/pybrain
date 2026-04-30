"""
PACS client for DICOM communication using pynetdicom.
Provides C-ECHO, C-FIND, C-MOVE, and C-STORE operations.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
import threading
import time

try:
    from pynetdicom import AE, evt, build_context
    from pynetdicom.sop_class import (
        Verification,
        StudyRootQueryRetrieveInformationModelFind,
        StudyRootQueryRetrieveInformationModelMove,
        CTImageStorage,
        MRImageStorage,
        SecondaryCaptureImageStorage,
    )
    from pynetdicom import Association
    PYNETDICOM_AVAILABLE = True
except ImportError:
    PYNETDICOM_AVAILABLE = False

from pybrain.io.logging_utils import get_logger

logger = get_logger("pybrain")


@dataclass
class StudyInfo:
    """Information about a DICOM study from C-FIND."""
    study_uid: str
    patient_id: str
    patient_name: str
    study_date: str
    study_description: str
    modalities: List[str]
    series_count: int


class PACSClient:
    """DICOM SCU for communication with PACS."""

    def __init__(
        self,
        ae_title: str,
        peer_ae_title: str,
        peer_host: str,
        peer_port: int,
        local_port: int = 11112,
    ):
        """
        Initialize PACS client.

        Args:
            ae_title: Local Application Entity title (e.g., "PYBRAIN")
            peer_ae_title: Remote PACS AE title (e.g., "ORTHANC")
            peer_host: Remote PACS hostname or IP
            peer_port: Remote PACS DICOM port
            local_port: Local port for DICOM communication
        """
        if not PYNETDICOM_AVAILABLE:
            raise ImportError(
                "pynetdicom is required for PACS connectivity. "
                "Install with: pip install pynetdicom>=2.0"
            )

        self.ae_title = ae_title
        self.peer_ae_title = peer_ae_title
        self.peer_host = peer_host
        self.peer_port = peer_port
        self.local_port = local_port

        # Initialize Application Entity
        self.ae = AE(ae_title=ae_title)
        self.ae.add_requested_context(Verification)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
        self.ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)
        self.ae.add_requested_context(CTImageStorage)
        self.ae.add_requested_context(MRImageStorage)
        self.ae.add_requested_context(SecondaryCaptureImageStorage)

        logger.info(
            f"PACSClient initialized: {ae_title} -> {peer_ae_title}@{peer_host}:{peer_port}"
        )

    def c_echo(self) -> bool:
        """
        Verify connectivity to PACS using C-ECHO.

        Returns:
            True if C-ECHO successful, False otherwise
        """
        try:
            assoc = self.ae.associate(
                self.peer_host, self.peer_port, ae_title=self.peer_ae_title
            )
            if assoc.is_established:
                status = assoc.send_c_echo()
                assoc.release()
                logger.info(f"C-ECHO successful: {status}")
                return True
            else:
                logger.warning("C-ECHO: Association not established")
                return False
        except Exception as exc:
            logger.error(f"C-ECHO failed: {exc}")
            return False

    def c_find_studies(self, query: Dict[str, Any]) -> List[StudyInfo]:
        """
        Query PACS for studies using C-FIND.

        Args:
            query: Query dictionary with DICOM tags (e.g., {"PatientName": "Doe^John"})

        Returns:
            List of StudyInfo objects matching the query
        """
        studies = []

        def handle_find(event):
            """Handle C-FIND responses."""
            dataset = event.dataset
            if dataset:
                # Extract study information
                study_info = StudyInfo(
                    study_uid=str(dataset.get("StudyInstanceUID", "")),
                    patient_id=str(dataset.get("PatientID", "")),
                    patient_name=str(dataset.get("PatientName", "")),
                    study_date=str(dataset.get("StudyDate", "")),
                    study_description=str(dataset.get("StudyDescription", "")),
                    modalities=str(dataset.get("ModalitiesInStudy", "")).split("\\")
                    if "ModalitiesInStudy" in dataset
                    else [],
                    series_count=int(dataset.get("NumberOfSeriesInStudy", 0)),
                )
                studies.append(study_info)
                return 0x0000  # Success
            return 0xFE00  # Warning

        try:
            # Build query dataset
            from pydicom.dataset import Dataset
            ds = Dataset()
            ds.QueryRetrieveLevel = "STUDY"
            for key, value in query.items():
                setattr(ds, key, value)

            # Add required return keys
            ds.StudyInstanceUID = ""
            ds.PatientID = ""
            ds.PatientName = ""
            ds.StudyDate = ""
            ds.StudyDescription = ""
            ds.ModalitiesInStudy = ""
            ds.NumberOfSeriesInStudy = ""

            assoc = self.ae.associate(
                self.peer_host,
                self.peer_port,
                ae_title=self.peer_ae_title,
            )
            if assoc.is_established:
                assoc.send_c_find(
                    ds, StudyRootQueryRetrieveInformationModelFind, handle_find
                )
                assoc.release()
                logger.info(f"C-FIND returned {len(studies)} studies")
            else:
                logger.warning("C-FIND: Association not established")
        except Exception as exc:
            logger.error(f"C-FIND failed: {exc}")

        return studies

    def c_move_series(
        self, study_uid: str, series_uid: str, dest_ae: str
    ) -> bool:
        """
        Request C-MOVE to retrieve a series from PACS.

        Args:
            study_uid: Study Instance UID
            series_uid: Series Instance UID
            dest_ae: Destination AE title for C-MOVE

        Returns:
            True if C-MOVE successful, False otherwise
        """
        success = False

        def handle_move(event):
            """Handle C-MOVE responses."""
            nonlocal success
            status = event.status
            if status in (0x0000, 0x0001):
                success = True
            return 0x0000

        try:
            from pydicom.dataset import Dataset
            ds = Dataset()
            ds.QueryRetrieveLevel = "SERIES"
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid

            assoc = self.ae.associate(
                self.peer_host,
                self.peer_port,
                ae_title=self.peer_ae_title,
            )
            if assoc.is_established:
                assoc.send_c_move(
                    ds, StudyRootQueryRetrieveInformationModelMove, dest_ae, handle_move
                )
                assoc.release()
                logger.info(f"C-MOVE {'successful' if success else 'failed'}")
            else:
                logger.warning("C-MOVE: Association not established")
        except Exception as exc:
            logger.error(f"C-MOVE failed: {exc}")

        return success

    def c_store(self, dicom_path: Path) -> bool:
        """
        Push a single DICOM file to PACS using C-STORE.

        Args:
            dicom_path: Path to DICOM file to store

        Returns:
            True if C-STORE successful, False otherwise
        """
        try:
            from pydicom import dcmread

            dataset = dcmread(str(dicom_path))
            success = False

            def handle_store(event):
                """Handle C-STORE responses."""
                nonlocal success
                status = event.status
                if status in (0x0000, 0x0001):
                    success = True
                return 0x0000

            # Determine presentation context based on SOP Class UID
            sop_class = dataset.SOPClassUID
            self.ae.add_requested_context(sop_class)

            assoc = self.ae.associate(
                self.peer_host,
                self.peer_port,
                ae_title=self.peer_ae_title,
            )
            if assoc.is_established:
                assoc.send_c_store(dataset, handle_store)
                assoc.release()
                logger.info(f"C-STORE {dicom_path.name}: {'successful' if success else 'failed'}")
                return success
            else:
                logger.warning("C-STORE: Association not established")
                return False
        except Exception as exc:
            logger.error(f"C-STORE failed for {dicom_path}: {exc}")
            return False

    def c_store_directory(self, dicom_dir: Path) -> int:
        """
        Push all DICOM files in a directory to PACS.

        Args:
            dicom_dir: Directory containing .dcm files

        Returns:
            Number of successfully stored files
        """
        dicom_files = list(dicom_dir.glob("*.dcm"))
        if not dicom_files:
            dicom_files = list(dicom_dir.glob("*"))  # Try all files
            dicom_files = [f for f in dicom_files if f.is_file()]

        success_count = 0
        for dicom_path in dicom_files:
            if self.c_store(dicom_path):
                success_count += 1

        logger.info(f"C-STORE directory: {success_count}/{len(dicom_files)} files successful")
        return success_count

    def start_storage_scp(
        self, handler: Callable[[Path, Dict[str, Any]], None], output_dir: Path
    ) -> None:
        """
        Start a DICOM Storage SCP to listen for incoming C-STORE requests.

        This runs in a background thread.

        Args:
            handler: Callback function to handle received DICOM files.
                     Signature: handler(dicom_path: Path, metadata: Dict) -> None
            output_dir: Directory to store received DICOM files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create AE for SCP
        scp_ae = AE(ae_title=self.ae_title)
        scp_ae.add_supported_context(Verification)
        scp_ae.add_supported_context(CTImageStorage)
        scp_ae.add_supported_context(MRImageStorage)
        scp_ae.add_supported_context(SecondaryCaptureImageStorage)

        def handle_store_scp(event):
            """Handle incoming C-STORE requests."""
            try:
                dataset = event.dataset
                dataset.file_meta = event.file_meta

                # Generate filename
                sop_instance_uid = str(dataset.get("SOPInstanceUID", "unknown"))
                filename = f"{sop_instance_uid}.dcm"
                output_path = output_dir / filename

                # Save file
                dataset.save_as(str(output_path), write_like_original=False)

                # Extract metadata
                metadata = {
                    "study_uid": str(dataset.get("StudyInstanceUID", "")),
                    "series_uid": str(dataset.get("SeriesInstanceUID", "")),
                    "sop_uid": sop_instance_uid,
                    "patient_id": str(dataset.get("PatientID", "")),
                    "patient_name": str(dataset.get("PatientName", "")),
                    "modality": str(dataset.get("Modality", "")),
                    "series_description": str(dataset.get("SeriesDescription", "")),
                    "study_date": str(dataset.get("StudyDate", "")),
                }

                # Call handler
                handler(output_path, metadata)

                logger.info(f"C-STORE SCP received: {filename}")
                return 0x0000
            except Exception as exc:
                logger.error(f"C-STORE SCP handler failed: {exc}")
                return 0x0110  # Processing failure

        def run_scp():
            """Run SCP in background."""
            try:
                scp = scp_ae.start_server(
                    ("", self.local_port),
                    block=False,
                    evt_handlers=[(evt.EVT_C_STORE, handle_store_scp)],
                )
                logger.info(f"Storage SCP listening on port {self.local_port}")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Storage SCP shutting down")
                scp.shutdown()

        # Start in background thread
        scp_thread = threading.Thread(target=run_scp, daemon=True)
        scp_thread.start()
        logger.info("Storage SCP started in background")
