"""
Case lifecycle tests: upload → segment → fetch results → delete.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from io import BytesIO


@pytest.mark.asyncio
async def test_create_case():
    """Test creating a case with file upload."""
    from pybrain.api.main import app
    from pybrain.api.config import settings

    # Set development mode for testing
    original_env = settings.environment
    settings.environment = "development"

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Create a dummy file for upload
            file_content = BytesIO(b"test data")
            files = {"files": ("test.nii.gz", file_content, "application/octet-stream")}
            data = {"analysis_mode": "auto"}

            response = await client.post(
                "/cases",
                files=files,
                data=data,
                headers={"Authorization": "Bearer dev-api-key-123"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "case_id" in data
            assert data["status"] == "pending"
            return data["case_id"]
    finally:
        settings.environment = original_env


@pytest.mark.asyncio
async def test_get_case():
    """Test retrieving case information."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import Case

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncSessionLocal() as db:
            # Create a test case
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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                f"/cases/{case_id}",
                headers={"Authorization": "Bearer dev-api-key-123"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["case_id"] == case_id
            assert data["status"] == "pending"
    finally:
        settings.environment = original_env


@pytest.mark.asyncio
async def test_delete_case():
    """Test soft deleting a case."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import Case
    from sqlalchemy import select

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncSessionLocal() as db:
            # Create a test case
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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete(
                f"/cases/{case_id}",
                headers={"Authorization": "Bearer dev-api-key-123"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["case_id"] == case_id

        # Verify soft delete
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Case).where(Case.id == case_id))
            case = result.scalar_one()
            assert case.deleted_at is not None
    finally:
        settings.environment = original_env


@pytest.mark.asyncio
async def test_trigger_segmentation():
    """Test triggering segmentation for a case."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import Case

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncSessionLocal() as db:
            # Create a test case
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

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/cases/{case_id}/segment",
                headers={"Authorization": "Bearer dev-api-key-123"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "pending"
    finally:
        settings.environment = original_env
