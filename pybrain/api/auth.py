"""
Authentication utilities for PY-BRAIN API.
"""

from datetime import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pybrain.api.config import settings
from pybrain.api.db.base import get_db
from pybrain.api.db.models import User

# Security
security = HTTPBearer(auto_error=False)


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
