"""
Concurrent jobs test: 5 parallel segmentations don't deadlock workers.
"""

import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy import select


@pytest.mark.asyncio
async def test_concurrent_segmentation_jobs():
    """Test that 5 parallel segmentation jobs don't deadlock workers."""
    from pybrain.api.main import app
    from pybrain.api.config import settings
    from pybrain.api.db.base import engine, Base, AsyncSessionLocal
    from pybrain.api.db.models import Case, Job

    original_env = settings.environment
    settings.environment = "development"

    try:
        # Initialize test database
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create 5 test cases
        case_ids = []
        async with AsyncSessionLocal() as db:
            import uuid
            for i in range(5):
                case_id = str(uuid.uuid4())
                case = Case(
                    id=case_id,
                    user_id=1,
                    status="pending",
                    analysis_mode="auto",
                    storage_path=f"cases/{case_id}",
                )
                db.add(case)
                case_ids.append(case_id)
            await db.commit()

        # Trigger 5 concurrent segmentation jobs
        async with AsyncClient(app=app, base_url="http://test") as client:
            tasks = []
            for case_id in case_ids:
                task = client.post(
                    f"/cases/{case_id}/segment",
                    headers={"Authorization": "Bearer dev-api-key-123"},
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            # All should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "job_id" in data
                assert data["status"] == "pending"

        # Verify all jobs were created
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Job))
            jobs = result.scalars().all()
            assert len(jobs) == 5

    finally:
        settings.environment = original_env
