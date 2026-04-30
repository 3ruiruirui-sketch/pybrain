"""
FastAPI application for PY-BRAIN API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from pybrain.api.config import settings
from pybrain.api.db.base import engine, init_db
from pybrain.api.routes import cases, jobs

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting PY-BRAIN API...")
    await init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down PY-BRAIN API...")
    await engine.dispose()
    logger.info("Database connection closed")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="PY-BRAIN Brain Tumor Segmentation API",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)


# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check (always returns 200)."""
    return {"status": "healthy", "version": settings.app_version}


@app.get("/ready")
async def readiness_check():
    """
    Readiness check (checks Redis and DB connectivity).
    """
    # Check Redis
    try:
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"

    # Check Database
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    if redis_status == "healthy" and db_status == "healthy":
        return {
            "status": "ready",
            "redis": redis_status,
            "database": db_status,
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "redis": redis_status,
                "database": db_status,
            },
        )


# Auth middleware
async def verify_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Verify JWT or API key authentication.
    In research mode, API keys are accepted.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Check if it's an API key (research mode)
    if settings.api_keys and token in settings.api_keys:
        return {"type": "api_key", "key": token}

    # TODO: Verify JWT token
    # For now, accept any token in development mode
    if settings.environment == "development":
        return {"type": "jwt", "token": token}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Include routers
app.include_router(cases.router, dependencies=[Depends(verify_auth)])
app.include_router(jobs.router, dependencies=[Depends(verify_auth)])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
    }


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "pybrain.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
