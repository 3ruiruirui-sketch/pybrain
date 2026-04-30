"""DICOM C-STORE SCP listener that triages incoming studies and triggers analysis."""
from pathlib import Path
from typing import Callable, Optional

from pynetdicom import AE, evt, AllStoragePresentationContexts

from pybrain.io.logging_utils import get_logger


logger = get_logger("pacs.listener")


class PACSListener:
    """
    A minimal DICOM C-STORE Service Class Provider.

    Listens on a TCP port for incoming DICOM studies, writes them to disk
    organized by Study/Series UID, and invokes a callback when a study
    appears complete (>=4 series buffered, configurable).

    Research-only: this listener is not validated for clinical use and
    does not enforce HIPAA / SOC 2 / FDA-required audit and access controls.
    """

    def __init__(
        self,
        output_dir: Path,
        on_study_received: Optional[Callable[[str, Path], None]] = None,
        min_series_for_complete: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_study_received = on_study_received
        self.min_series_for_complete = min_series_for_complete
        self._scp = None
        self._study_buffers: dict[str, list[Path]] = {}

    def _handle_store(self, event):
        """C-STORE handler: write the incoming dataset to disk."""
        ds = event.dataset
        ds.file_meta = event.file_meta
        study_uid = str(ds.StudyInstanceUID)
        series_uid = str(ds.SeriesInstanceUID)
        sop_uid = str(ds.SOPInstanceUID)

        study_dir = self.output_dir / study_uid / series_uid
        study_dir.mkdir(parents=True, exist_ok=True)
        out_path = study_dir / f"{sop_uid}.dcm"
        ds.save_as(str(out_path), write_like_original=False)

        self._study_buffers.setdefault(study_uid, []).append(out_path)
        logger.info(f"Stored {sop_uid} in {study_dir}")
        return 0x0000  # Status: Success

    def _handle_release(self, event):
        """A-RELEASE handler: triage any study that now looks complete."""
        for study_uid, files in list(self._study_buffers.items()):
            if self._is_study_complete(study_uid):
                study_dir = self.output_dir / study_uid
                logger.info(
                    f"Triaging study {study_uid} ({len(files)} files, "
                    f"{len(list(study_dir.iterdir()))} series)"
                )
                if self.on_study_received:
                    try:
                        self.on_study_received(study_uid, study_dir)
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            f"on_study_received callback failed for {study_uid}: {e}"
                        )
                self._study_buffers.pop(study_uid, None)

    def _is_study_complete(self, study_uid: str) -> bool:
        """Heuristic: a study is 'complete' when N series directories exist.

        For brain tumor analysis we need at least T1, T1c, T2, FLAIR (4 series).
        Subclass and override for stricter modality / SeriesDescription matching.
        """
        study_dir = self.output_dir / study_uid
        if not study_dir.exists():
            return False
        return len(list(study_dir.iterdir())) >= self.min_series_for_complete

    def start(
        self,
        ae_title: str = "PYBRAIN",
        host: str = "0.0.0.0",
        port: int = 11112,
        block: bool = True,
    ):
        """Start the SCP. With block=True this call will not return until shutdown."""
        ae = AE(ae_title=ae_title)
        ae.supported_contexts = AllStoragePresentationContexts
        handlers = [
            (evt.EVT_C_STORE, self._handle_store),
            (evt.EVT_RELEASED, self._handle_release),
        ]
        logger.info(
            f"PACS listener starting on {host}:{port} "
            f"(AE={ae_title}, output_dir={self.output_dir})"
        )
        self._scp = ae.start_server(
            (host, port), evt_handlers=handlers, block=block
        )

    def stop(self):
        """Shut down the SCP if running."""
        if self._scp is not None:
            self._scp.shutdown()
            self._scp = None
            logger.info("PACS listener stopped")
