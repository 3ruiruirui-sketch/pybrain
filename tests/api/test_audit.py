"""
Audit logging tests: every modifying action produces an audit row.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select


@pytest.mark.asyncio
async def test_audit_log_on_case_creation():
    """Test that case creation creates an audit log entry."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import AuditLog

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        from io import BytesIO
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            file_content = BytesIO(b"test data")
            files = {"files": ("test.nii.gz", file_content, "application/octet-stream")}
            data = {"analysis_mode": "auto"}

            await client.post(
                "/cases",
                files=files,
                data=data,
                headers={"Authorization": "Bearer dev-api-key-123"},
            )

        # Check audit log
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(AuditLog).order_by(AuditLog.timestamp.desc()))
            audit_logs = result.scalars().all()
            assert len(audit_logs) > 0
            latest_log = audit_logs[0]
            assert latest_log.action == "create"
            assert latest_log.resource_type == "case"
    finally:
        settings.environment = original_env


@pytest.mark.asyncio
async def test_audit_log_on_case_deletion():
    """Test that case deletion creates an audit log entry."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import Case, AuditLog
    from sqlalchemy import select

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create a test case
        async with AsyncSessionLocal() as db:
            import uuid
            case_id = str(uuid.uuid4())
            case = Case(
                id=case_id,
                user_id=1,
                status="pending",
                analysis_mode="auto",
                storage_path=f"cases/{case_id}",
            )
            db.add(case)
            await db.commit()

        # Delete the case
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.delete(
                f"/cases/{case_id}",
                headers={"Authorization": "Bearer dev-api-key-123"},
            )

        # Check audit log
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(AuditLog)
                .where(AuditLog.action == "delete")
                .where(AuditLog.resource_id == case_id)
            )
            audit_logs = result.scalars().all()
            assert len(audit_logs) > 0
            assert audit_logs[0].action == "delete"
    finally:
        settings.environment = original_env


@pytest.mark.asyncio
async def test_audit_log_on_case_access():
    """Test that case access creates an audit log entry."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import Case, AuditLog
    from sqlalchemy import select

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create a test case
        async with AsyncSessionLocal() as db:
            import uuid
            case_id = str(uuid.uuid4())
            case = Case(
                id=case_id,
                user_id=1,
                status="pending",
                analysis_mode="auto",
                storage_path=f"cases/{case_id}",
            )
            db.add(case)
            await db.commit()

        # Access the case
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get(
                f"/cases/{case_id}",
                headers={"Authorization": "Bearer dev-api-key-123"},
            )

        # Check audit log
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(AuditLog)
                .where(AuditLog.action == "read")
                .where(AuditLog.resource_id == case_id)
            )
            audit_logs = result.scalars().all()
            assert len(audit_logs) > 0
            assert audit_logs[0].action == "read"
    finally:
        settings.environment = original_env
