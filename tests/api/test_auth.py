"""
Authentication tests: missing/invalid/valid JWT and API keys.
"""

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_missing_auth():
    """Test that requests without authentication fail."""
    from pybrain.api.main import app
    from pybrain.api.config import settings

    original_env = settings.environment
    settings.environment = "production"  # Not development mode

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/cases/test-case")
            assert response.status_code == 401
    finally:
        settings.environment = original_env


@pytest.mark.asyncio
async def test_invalid_api_key():
    """Test that invalid API keys are rejected."""
    from pybrain.api.main import app
    from pybrain.api.config import settings

    original_env = settings.environment
    settings.environment = "production"
    original_keys = settings.api_keys.copy()
    settings.api_keys = ["valid-key-123"]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/cases/test-case",
                headers={"Authorization": "Bearer invalid-key"},
            )
            assert response.status_code == 401
    finally:
        settings.environment = original_env
        settings.api_keys = original_keys


@pytest.mark.asyncio
async def test_valid_api_key():
    """Test that valid API keys are accepted."""
    from pybrain.api.main import app
    from pybrain.api.config import settings

    original_env = settings.environment
    settings.environment = "production"
    original_keys = settings.api_keys.copy()
    settings.api_keys = ["valid-key-123"]

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/cases/test-case",
                headers={"Authorization": "Bearer valid-key-123"},
            )
            # Will return 404 (case not found) but not 401 (auth failed)
            assert response.status_code == 404
    finally:
        settings.environment = original_env
        settings.api_keys = original_keys


@pytest.mark.asyncio
async def test_development_mode_any_token():
    """Test that development mode accepts any token."""
    from pybrain.api.main import app
    from pybrain.api.config import settings

    original_env = settings.environment
    settings.environment = "development"

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/cases/test-case",
                headers={"Authorization": "Bearer any-token"},
            )
            # Will return 404 (case not found) but not 401 (auth failed)
            assert response.status_code == 404
    finally:
        settings.environment = original_env
