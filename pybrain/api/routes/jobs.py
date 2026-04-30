"""
Job management routes.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from pybrain.api.db.base import get_db
from pybrain.api.db.models import Job, Case
from pybrain.api.tasks import segment_case, longitudinal_compare, export_dicom
from pybrain.api.audit import log_api_call
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/segment", response_model=Dict[str, Any])
async def create_segmentation_job(
    case_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Enqueue a segmentation job for a case.

    Args:
        case_id: Case ID to segment

    Returns:
        Job ID and initial status
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Create job record
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        case_id=case_id,
        user_id=1,  # TODO: Get from auth context
        job_type="segment",
        status="pending",
        progress=0,
        parameters={"case_id": case_id},
    )
    db.add(job)
    await db.commit()

    # Enqueue Celery task
    task = segment_case.delay(case_id)
    job.celery_task_id = task.id
    await db.commit()

    # Log audit
    await log_api_call(
        db=db,
        user_id=1,  # TODO: Get from auth context
        action="create",
        resource_type="job",
        resource_id=job_id,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "celery_task_id": task.id,
    }


@router.get("/{job_id}", response_model=Dict[str, Any])
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get job status and progress.

    Args:
        job_id: Job ID

    Returns:
        Job status, progress, and result (if complete)
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Log audit
    await log_api_call(
        db=db,
        user_id=1,  # TODO: Get from auth context
        action="read",
        resource_type="job",
        resource_id=job_id,
    )

    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("/longitudinal", response_model=Dict[str, Any])
async def create_longitudinal_job(
    case_id: str,
    prior_case_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Enqueue a longitudinal comparison job.

    Args:
        case_id: Current case ID
        prior_case_id: Prior case ID

    Returns:
        Job ID and initial status
    """
    # Verify cases exist
    current_result = await db.execute(select(Case).where(Case.id == case_id))
    current_case = current_result.scalar_one_or_none()
    if not current_case:
        raise HTTPException(status_code=404, detail="Current case not found")

    prior_result = await db.execute(select(Case).where(Case.id == prior_case_id))
    prior_case = prior_result.scalar_one_or_none()
    if not prior_case:
        raise HTTPException(status_code=404, detail="Prior case not found")

    # Create job record
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        case_id=case_id,
        user_id=1,  # TODO: Get from auth context
        job_type="longitudinal",
        status="pending",
        progress=0,
        parameters={
            "case_id": case_id,
            "prior_case_id": prior_case_id,
        },
    )
    db.add(job)
    await db.commit()

    # Enqueue Celery task
    task = longitudinal_compare.delay(case_id, prior_case_id)
    job.celery_task_id = task.id
    await db.commit()

    # Log audit
    await log_api_call(
        db=db,
        user_id=1,  # TODO: Get from auth context
        action="create",
        resource_type="job",
        resource_id=job_id,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "celery_task_id": task.id,
    }


@router.post("/export-dicom", response_model=Dict[str, Any])
async def create_export_job(
    case_id: str,
    formats: list[str],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Enqueue a DICOM export job.

    Args:
        case_id: Case ID to export
        formats: List of export formats (e.g., ["dicom-seg", "dicom-sr"])

    Returns:
        Job ID and initial status
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Create job record
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        case_id=case_id,
        user_id=1,  # TODO: Get from auth context
        job_type="export_dicom",
        status="pending",
        progress=0,
        parameters={
            "case_id": case_id,
            "formats": formats,
        },
    )
    db.add(job)
    await db.commit()

    # Enqueue Celery task
    task = export_dicom.delay(case_id, formats)
    job.celery_task_id = task.id
    await db.commit()

    # Log audit
    await log_api_call(
        db=db,
        user_id=1,  # TODO: Get from auth context
        action="create",
        resource_type="job",
        resource_id=job_id,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "celery_task_id": task.id,
    }
