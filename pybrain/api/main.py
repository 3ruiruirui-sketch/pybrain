"""
FastAPI application for PY-BRAIN API.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from pybrain.api.config import settings
from pybrain.api.db.base import engine, init_db, get_db
from pybrain.api.db.models import User
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
            await conn.execute(text("SELECT 1"))
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
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Verify JWT or API key authentication and return User object.
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
        # Look up user by api_key
        result = await db.execute(
            select(User).where(User.api_key == token, User.is_active == True)
        )
        user = result.scalar_one_or_none()
        if user is None:
            # Create a default user for research mode
            user = User(
                username="research_user",
                api_key=token,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user

    # Try JWT verification
    try:
        from jose import jwt, JWTError

        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise JWTError("Missing sub claim")

        user = await db.get(User, int(user_id))
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )
        return user
    except ImportError:
        # python-jose not installed, fall back to development mode
        pass
    except JWTError as e:
        if settings.environment == "development":
            # Dev fallback: create default user
            result = await db.execute(select(User).where(User.username == "dev"))
            user = result.scalar_one_or_none()
            if user is None:
                user = User(
                    username="dev",
                    api_key="dev",
                    is_active=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(user)
                await db.commit()
                await db.refresh(user)
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )

    # Development mode fallback
    if settings.environment == "development":
        result = await db.execute(select(User).where(User.username == "dev"))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                username="dev",
                api_key="dev",
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Include routers
app.include_router(cases.router)
app.include_router(jobs.router)


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
