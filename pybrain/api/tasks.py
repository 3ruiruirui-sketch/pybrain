"""
Celery tasks for async job processing.
"""

import shutil
from datetime import datetime
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession

from pybrain.api.config import settings
from pybrain.api.db.base import AsyncSessionLocal
from pybrain.api.db.models import Job, Case
from pybrain.api.storage import storage
import logging

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "pybrain",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["pybrain.api.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.job_timeout_minutes * 60,
)


@celery_app.task(bind=True)
def segment_case(self, case_id: str) -> Dict[str, Any]:
    """
    Run segmentation pipeline on a case.

    Args:
        case_id: Case ID to segment

    Returns:
        Result dictionary with volumes and paths
    """
    import asyncio
    from sqlalchemy import select, update

    async def _run_segmentation():
        async with AsyncSessionLocal() as db:
            # Update job status to started
            job = await db.get(Job, self.request.id)
            if job:
                job.status = "started"
                job.started_at = datetime.utcnow()
                await db.commit()

            # Get case
            case = await db.get(Case, case_id)
            if not case:
                raise ValueError(f"Case not found: {case_id}")

            # Update case status
            case.status = "processing"
            await db.commit()

            try:
                # Run actual PY-BRAIN pipeline
                logger.info(f"Segmenting case {case_id}")

                # Get input files from storage
                input_files = {}
                storage_path = case.storage_path
                
                # Discover input files in storage
                # Expected structure: cases/{case_id}/{modality}.nii.gz
                for modality in ["T1", "T1c", "T2", "FLAIR"]:
                    file_path = f"{storage_path}/{modality.lower()}.nii.gz"
                    try:
                        if await storage.file_exists(file_path):
                            # Download file to temp location
                            temp_dir = Path(f"/tmp/pybrain_{case_id}")
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            temp_file = temp_dir / f"{modality.lower()}.nii.gz"
                            
                            file_content = await storage.get_file(file_path)
                            with open(temp_file, "wb") as f:
                                shutil.copyfileobj(file_content, f)
                            
                            input_files[modality] = str(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not find {modality} file: {e}")
                
                if not input_files:
                    raise ValueError(f"No input files found for case {case_id}")

                # Create output directory
                output_dir = Path(f"/tmp/pybrain_output_{case_id}")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Update progress
                if job:
                    job.progress = 10
                    await db.commit()

                # Run PY-BRAIN pipeline
                from pybrain.pipeline import run as run_pipeline
                
                analysis_mode: Literal["glioma", "mets", "auto"] = (
                    case.analysis_mode or "auto"
                )  # type: ignore
                
                result = run_pipeline(
                    assignments=input_files,
                    output_dir=output_dir,
                    analysis_mode=analysis_mode,
                    run_location=True,
                    run_morphology=True,
                    run_radiomics=False,  # Skip radiomics for API
                    run_report=True,
                    patient={
                        "name": case.patient_name,
                        "age": case.patient_age,
                        "sex": case.patient_sex,
                    } if case.patient_name else None,
                )

                # Update progress
                if job:
                    job.progress = 80
                    await db.commit()

                # Save results to storage
                segmentation_path = output_dir / "segmentation_full.nii.gz"
                report_path = output_dir / f"report_{case_id}.pdf"
                
                if segmentation_path.exists():
                    seg_storage_path = f"{storage_path}/segmentation.nii.gz"
                    with open(segmentation_path, "rb") as f:
                        await storage.save_file(seg_storage_path, f)
                    case.segmentation_path = seg_storage_path
                
                if report_path.exists():
                    report_storage_path = f"{storage_path}/report.pdf"
                    with open(report_path, "rb") as f:
                        await storage.save_file(report_storage_path, f)
                    case.report_path = report_storage_path

                # Extract volumes from result
                volumes = result.get("volumes", {})
                
                # Update case with results
                case.volumes = volumes
                case.status = "completed"
                case.updated_at = datetime.utcnow()

                # Update job
                if job:
                    job.status = "success"
                    job.progress = 100
                    job.completed_at = datetime.utcnow()
                    job.result = {
                        "volumes": volumes,
                        "segmentation_path": case.segmentation_path,
                        "report_path": case.report_path,
                    }

                await db.commit()

                return {
                    "volumes": volumes,
                    "segmentation_path": case.segmentation_path,
                    "report_path": case.report_path,
                }

            except Exception as e:
                logger.error(f"Segmentation failed for case {case_id}: {e}")
                case.status = "failed"
                case.error_message = str(e)
                case.updated_at = datetime.utcnow()

                if job:
                    job.status = "failure"
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()

                await db.commit()
                raise

    return asyncio.run(_run_segmentation())


@celery_app.task(bind=True)
def longitudinal_compare(self, case_id: str, prior_case_id: str) -> Dict[str, Any]:
    """Compare two timepoints for a patient — RANO assessment."""
    import asyncio
    from pathlib import Path
    from pybrain.analysis.longitudinal import compare_timepoints
    from pybrain.io.config import get_config
    from pybrain.api.storage import storage
    from datetime import datetime
    import shutil

    async def _run():
        async with AsyncSessionLocal() as db:
            # Fetch both cases
            current = await db.get(Case, case_id)
            prior = await db.get(Case, prior_case_id)
            if not current or not prior:
                raise ValueError("Both current and prior cases must exist")
            if not current.segmentation_path:
                raise ValueError(f"Current case {case_id} not yet segmented")
            if not prior.segmentation_path:
                raise ValueError(f"Prior case {prior_case_id} not yet segmented")

            # Update job status
            job = await db.get(Job, self.request.id)
            if job:
                job.status = "started"
                job.started_at = datetime.utcnow()
                await db.commit()

        # Pull files to a working directory
        work_dir = Path(f"/tmp/pybrain_long_{case_id}_{prior_case_id}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # Resolve T1c paths (current, prior) and segmentation paths
        # File naming convention from segment_case
        async def _fetch(storage_path: str, local_name: str) -> Path:
            local = work_dir / local_name
            file_content = await storage.get_file(storage_path)
            with open(local, "wb") as f:
                shutil.copyfileobj(file_content, f)
            return local

        current_t1c = await _fetch(f"{current.storage_path}/t1c.nii.gz", "current_t1c.nii.gz")
        prior_t1c = await _fetch(f"{prior.storage_path}/t1c.nii.gz", "prior_t1c.nii.gz")
        current_seg = await _fetch(current.segmentation_path, "current_seg.nii.gz")
        prior_seg = await _fetch(prior.segmentation_path, "prior_seg.nii.gz")

        config = get_config()

        long_result = compare_timepoints(
            current_t1c=current_t1c,
            current_seg=current_seg,
            prior_t1c=prior_t1c,
            prior_seg=prior_seg,
            output_dir=work_dir / "output",
            config=config,
        )

        # Persist back to database
        async with AsyncSessionLocal() as db:
            current = await db.get(Case, case_id)
            if current is None:
                raise ValueError(f"Current case {case_id} not found")
            current.longitudinal_result = {
                "prior_case_id": prior_case_id,
                "rano_response": long_result.rano_response,
                "registration_quality": long_result.registration_quality,
                "volume_changes": {
                    k: {
                        "prior_cc": v.prior_cc,
                        "current_cc": v.current_cc,
                        "abs_change_cc": v.abs_change_cc,
                        "pct_change": v.pct_change,
                        "status": v.status,
                    }
                    for k, v in long_result.volume_changes.items()
                },
            }
            current.updated_at = datetime.utcnow()

            job = await db.get(Job, self.request.id)
            if job:
                job.status = "success"
                job.progress = 100
                job.completed_at = datetime.utcnow()
                job.result = current.longitudinal_result
            await db.commit()

        return current.longitudinal_result

    return asyncio.run(_run())


@celery_app.task(bind=True)
def export_dicom(self, case_id: str, formats: list[str]) -> Dict[str, Any]:
    """
    Export case results as DICOM files.

    Args:
        case_id: Case ID to export
        formats: List of export formats (e.g., ["dicom-seg", "dicom-sr"])

    Returns:
        Result dictionary with export paths

    Raises:
        NotImplementedError: DICOM export requires source DICOM series; not yet wired up via API
    """
    raise NotImplementedError("DICOM export requires source DICOM series; not yet wired up via API")
