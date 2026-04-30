"""
Celery tasks for async job processing.
"""

from datetime import datetime
from typing import Optional, Dict, Any
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
                # TODO: Run actual PY-BRAIN pipeline
                # This is a placeholder - integrate with pybrain.pipeline.run()
                # For now, simulate processing
                logger.info(f"Segmenting case {case_id}")

                # Simulate progress updates
                for i in range(0, 101, 25):
                    if job:
                        job.progress = i
                        await db.commit()

                # Mock result
                result = {
                    "volumes": {"wt_cc": 10.5, "tc_cc": 5.2, "et_cc": 3.1, "nc_cc": 5.3},
                    "segmentation_path": f"cases/{case_id}/segmentation.nii.gz",
                    "report_path": f"cases/{case_id}/report.pdf",
                }

                # Update case with results
                case.volumes = result["volumes"]
                case.segmentation_path = result["segmentation_path"]
                case.report_path = result["report_path"]
                case.status = "completed"
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
