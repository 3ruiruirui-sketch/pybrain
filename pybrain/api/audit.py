"""
Audit logging for HIPAA compliance.
Every read/write of patient data is logged to the AuditLog table.
Every API call is logged with user, timestamp, IP, action, resource.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from pybrain.api.db.models import AuditLog
from pybrain.api.db.models import User
import logging

logger = logging.getLogger(__name__)


async def log_audit(
    db: AsyncSession,
    user_id: Optional[int],
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """
    Create an audit log entry.

    Args:
        db: Database session
        user_id: User ID (None for anonymous/system actions)
        action: Action performed (create, read, update, delete)
        resource_type: Type of resource (case, job, user)
        resource_id: ID of the resource
        ip_address: Client IP address
        user_agent: Client user agent string
        old_values: Previous values (for update/delete)
        new_values: New values (for create/update)

    Returns:
        Created AuditLog entry
    """
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        old_values=old_values,
        new_values=new_values,
        timestamp=datetime.utcnow(),
    )

    db.add(audit_log)
    await db.flush()

    logger.info(
        f"Audit log: {action} {resource_type}/{resource_id} by user {user_id} from {ip_address}"
    )

    return audit_log


async def log_api_call(
    db: AsyncSession,
    user_id: Optional[int],
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """
    Log an API call (simplified version without value tracking).

    Args:
        db: Database session
        user_id: User ID
        action: API action
        resource_type: Type of resource
        resource_id: Resource ID
        ip_address: Client IP
        user_agent: Client user agent
    """
    await log_audit(
        db=db,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )


async def log_patient_data_access(
    db: AsyncSession,
    user_id: Optional[int],
    case_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """
    Log access to patient data (HIPAA requirement).

    Args:
        db: Database session
        user_id: User ID
        case_id: Case ID being accessed
        ip_address: Client IP
        user_agent: Client user agent
    """
    await log_audit(
        db=db,
        user_id=user_id,
        action="read",
        resource_type="case",
        resource_id=case_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )


async def log_patient_data_modification(
    db: AsyncSession,
    user_id: Optional[int],
    case_id: str,
    action: str,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """
    Log modification of patient data (HIPAA requirement).

    Args:
        db: Database session
        user_id: User ID
        case_id: Case ID being modified
        action: Type of modification (create, update, delete)
        old_values: Previous values
        new_values: New values
        ip_address: Client IP
        user_agent: Client user agent
    """
    await log_audit(
        db=db,
        user_id=user_id,
        action=action,
        resource_type="case",
        resource_id=case_id,
        old_values=old_values,
        new_values=new_values,
        ip_address=ip_address,
        user_agent=user_agent,
    )


async def get_user_audit_trail(
    db: AsyncSession,
    user_id: int,
    limit: int = 100,
) -> list[AuditLog]:
    """
    Get audit trail for a specific user.

    Args:
        db: Database session
        user_id: User ID
        limit: Maximum number of entries to return

    Returns:
        List of AuditLog entries
    """
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_case_audit_trail(
    db: AsyncSession,
    case_id: str,
    limit: int = 100,
) -> list[AuditLog]:
    """
    Get audit trail for a specific case.

    Args:
        db: Database session
        case_id: Case ID
        limit: Maximum number of entries to return

    Returns:
        List of AuditLog entries
    """
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.resource_id == case_id)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
