"""
Cases API routes for managing segmentation cases.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
import uuid
import shutil
import zipfile
import tempfile

from pybrain.api.db.base import get_db
from pybrain.api.db.models import Case, Job, LongitudinalLink, User
from pybrain.api.storage import storage
from pybrain.api.audit import log_patient_data_access, log_patient_data_modification, log_api_call
from pybrain.api.routes.jobs import create_segmentation_job
from pybrain.api.auth import verify_auth
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=Dict[str, Any])
async def create_case(
    files: UploadFile = File(...),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    analysis_mode: str = Form("auto"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    Upload DICOM zip or NIfTI files and create a case.

    Args:
        files: Uploaded file (DICOM zip or NIfTI files)
        patient_name: Patient name (optional)
        patient_age: Patient age (optional)
        patient_sex: Patient sex (optional)
        analysis_mode: Analysis mode (glioma, mets, auto)

    Returns:
        Case ID and initial status
    """
    # Generate case ID
    case_id = str(uuid.uuid4())

    # Create storage path for this case
    case_storage_path = f"cases/{case_id}"

    # Save uploaded file
    temp_path = Path(tempfile.mktemp())
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(files.file, f)

        # Extract if zip file
        if files.filename and files.filename.endswith(".zip"):
            extract_path = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(temp_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Store all extracted files
            for file_path in extract_path.rglob("*"):
                if file_path.is_file():
                    relative_path = f"{case_storage_path}/{file_path.relative_to(extract_path)}"
                    with open(file_path, "rb") as f:
                        await storage.save_file(relative_path, f)
        else:
            # Store single file
            relative_path = f"{case_storage_path}/{files.filename}"
            with open(temp_path, "rb") as f:
                await storage.save_file(relative_path, f)

    finally:
        # Cleanup temp files
        if temp_path.exists():
            temp_path.unlink()

    # Create case record
    case = Case(
        id=case_id,
        user_id=current_user.id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_sex=patient_sex,
        status="pending",
        analysis_mode=analysis_mode,
        storage_path=case_storage_path,
    )
    db.add(case)
    await db.commit()

    # Log audit
    await log_patient_data_modification(
        db=db,
        user_id=current_user.id,
        case_id=case_id,
        action="create",
        new_values={
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_sex": patient_sex,
            "analysis_mode": analysis_mode,
        },
    )

    return {
        "case_id": case_id,
        "status": "pending",
        "analysis_mode": analysis_mode,
    }


@router.get("/{case_id}", response_model=Dict[str, Any])
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    Get case status and results.

    Args:
        case_id: Case ID

    Returns:
        Case information including status and results
    """
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Log audit
    await log_patient_data_access(
        db=db,
        user_id=current_user.id,
        case_id=case_id,
    )

    return {
        "case_id": case.id,
        "status": case.status,
        "analysis_mode": case.analysis_mode,
        "patient_name": case.patient_name,
        "patient_age": case.patient_age,
        "patient_sex": case.patient_sex,
        "volumes": case.volumes,
        "mets_result": case.mets_result,
        "longitudinal_result": case.longitudinal_result,
        "error_message": case.error_message,
        "created_at": case.created_at.isoformat(),
        "updated_at": case.updated_at.isoformat(),
    }


@router.get("/{case_id}/segmentation")
async def get_segmentation(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> FileResponse:
    """
    Download segmentation as NIfTI.

    Args:
        case_id: Case ID

    Returns:
        Segmentation NIfTI file
    """
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not case.segmentation_path:
        raise HTTPException(status_code=404, detail="Segmentation not available")

    # Log audit
    await log_patient_data_access(
        db=db,
        user_id=current_user.id,
        case_id=case_id,
    )

    # Get file from storage
    file_content = await storage.get_file(case.segmentation_path)
    
    # Write to temp file for FileResponse
    temp_path = Path(tempfile.mktemp(suffix=".nii.gz"))
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file_content, f)

    return FileResponse(
        temp_path,
        media_type="application/gzip",
        filename=f"{case_id}_segmentation.nii.gz",
    )


@router.get("/{case_id}/report")
async def get_report(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> FileResponse:
    """
    Download PDF report.

    Args:
        case_id: Case ID

    Returns:
        PDF report file
    """
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not case.report_path:
        raise HTTPException(status_code=404, detail="Report not available")

    # Log audit
    await log_patient_data_access(
        db=db,
        user_id=current_user.id,
        case_id=case_id,
    )

    # Get file from storage
    file_content = await storage.get_file(case.report_path)
    
    # Write to temp file for FileResponse
    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file_content, f)

    return FileResponse(
        temp_path,
        media_type="application/pdf",
        filename=f"{case_id}_report.pdf",
    )


@router.get("/{case_id}/dicom-seg")
async def get_dicom_seg(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> FileResponse:
    """
    Stream DICOM-SEG file.

    Args:
        case_id: Case ID

    Returns:
        DICOM-SEG file
    """
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not case.dicom_seg_path:
        raise HTTPException(status_code=404, detail="DICOM-SEG not available")

    # Log audit
    await log_patient_data_access(
        db=db,
        user_id=current_user.id,
        case_id=case_id,
    )

    # Get file from storage
    file_content = await storage.get_file(case.dicom_seg_path)
    
    # Write to temp file for FileResponse
    temp_path = Path(tempfile.mktemp(suffix=".dcm"))
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file_content, f)

    return FileResponse(
        temp_path,
        media_type="application/dicom",
        filename=f"{case_id}_segmentation.dcm",
    )


@router.post("/{case_id}/segment")
async def trigger_segmentation(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    Trigger segmentation for a case.

    Args:
        case_id: Case ID

    Returns:
        Job ID and status
    """
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Create segmentation job
    job_result = await create_segmentation_job(case_id, db)

    return job_result


@router.post("/{case_id}/longitudinal/{prior_id}")
async def trigger_longitudinal(
    case_id: str,
    prior_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    Trigger longitudinal comparison between current and prior case.

    Args:
        case_id: Current case ID
        prior_id: Prior case ID

    Returns:
        Job ID and status
    """
    from pybrain.api.routes.jobs import create_longitudinal_job

    result = await create_longitudinal_job(case_id, prior_id, db)

    # Create longitudinal link
    link = LongitudinalLink(
        prior_case_id=prior_id,
        current_case_id=case_id,
    )
    db.add(link)
    await db.commit()

    return result


@router.delete("/{case_id}", response_model=Dict[str, Any])
async def delete_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_auth),
) -> Dict[str, Any]:
    """
    Soft delete a case (audit trail preserved).

    Args:
        case_id: Case ID

    Returns:
        Success message
    """
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Soft delete
    case.deleted_at = datetime.utcnow()
    case.updated_at = datetime.utcnow()

    # Log audit
    await log_patient_data_modification(
        db=db,
        user_id=current_user.id,
        case_id=case_id,
        action="delete",
        old_values={"status": case.status},
    )

    await db.commit()

    return {
        "message": "Case soft deleted",
        "case_id": case_id,
    }
