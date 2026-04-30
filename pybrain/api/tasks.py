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
    """
    Run longitudinal comparison between two cases.

    Args:
        case_id: Current case ID
        prior_case_id: Prior case ID

    Returns:
        Result dictionary with comparison metrics
    """
    import asyncio
    from sqlalchemy import select

    async def _run_comparison():
        async with AsyncSessionLocal() as db:
            # Update job status to started
            job = await db.get(Job, self.request.id)
            if job:
                job.status = "started"
                job.started_at = datetime.utcnow()
                await db.commit()

            # Get cases
            current_case = await db.get(Case, case_id)
            prior_case = await db.get(Case, prior_case_id)

            if not current_case or not prior_case:
                raise ValueError("One or both cases not found")

            try:
                # TODO: Run actual longitudinal comparison
                # Integrate with pybrain.analysis.longitudinal
                logger.info(f"Comparing case {case_id} with prior {prior_case_id}")

                # Simulate processing
                if job:
                    job.progress = 50
                    await db.commit()

                # Mock result
                result = {
                    "rano_response": "stable",
                    "volume_changes": {
                        "wt_cc": {
                            "prior_cc": 10.0,
                            "current_cc": 10.5,
                            "pct_change": 5.0,
                            "status": "stable",
                        }
                    },
                    "registration_quality": 0.95,
                }

                # Update current case with results
                current_case.longitudinal_result = result
                current_case.updated_at = datetime.utcnow()

                # Update job
                if job:
                    job.status = "success"
                    job.progress = 100
                    job.completed_at = datetime.utcnow()
                    job.result = result

                await db.commit()

                return result

            except Exception as e:
                logger.error(f"Longitudinal comparison failed: {e}")

                if job:
                    job.status = "failure"
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()

                await db.commit()
                raise

    return asyncio.run(_run_comparison())


@celery_app.task(bind=True)
def export_dicom(self, case_id: str, formats: list[str]) -> Dict[str, Any]:
    """
    Export case results as DICOM files.

    Args:
        case_id: Case ID to export
        formats: List of export formats (e.g., ["dicom-seg", "dicom-sr"])

    Returns:
        Result dictionary with export paths
    """
    import asyncio

    async def _run_export():
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

            try:
                # TODO: Run actual DICOM export
                # Integrate with pybrain.io.dicom_seg_writer, etc.
                logger.info(f"Exporting case {case_id} as DICOM: {formats}")

                result = {}

                if "dicom-seg" in formats:
                    result["dicom_seg_path"] = f"cases/{case_id}/segmentation.dcm"
                    case.dicom_seg_path = result["dicom_seg_path"]

                if "dicom-sr" in formats:
                    result["dicom_sr_path"] = f"cases/{case_id}/report.dcm"
                    case.dicom_sr_path = result["dicom_sr_path"]

                case.updated_at = datetime.utcnow()

                # Update job
                if job:
                    job.status = "success"
                    job.progress = 100
                    job.completed_at = datetime.utcnow()
                    job.result = result

                await db.commit()

                return result

            except Exception as e:
                logger.error(f"DICOM export failed for case {case_id}: {e}")

                if job:
                    job.status = "failure"
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()

                await db.commit()
                raise

    return asyncio.run(_run_export())
