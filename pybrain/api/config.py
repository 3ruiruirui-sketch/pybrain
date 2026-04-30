"""
API configuration using Pydantic Settings.
Reads from environment variables following 12-factor app principles.
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Async database drivers - at module scope to avoid Pydantic v2 ModelPrivateAttr
_ASYNC_DRIVERS = (
    "postgresql+asyncpg://",
    "sqlite+aiosqlite://",
    "mysql+aiomysql://",
    "mysql+asyncmy://",
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "PY-BRAIN API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://pybrain:pybrain@localhost:5432/pybrain",
        description="Database connection URL (SQLAlchemy async format)",
    )

    # Redis (for Celery)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Celery broker and backend",
    )

    # Storage
    storage_backend: str = Field(
        default="local",
        description="Storage backend: local or s3",
    )
    storage_path: str = Field(
        default="/data/pybrain",
        description="Local filesystem storage path",
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket name (if using s3 backend)",
    )
    s3_region: str = Field(
        default="us-east-1",
        description="S3 region",
    )
    s3_endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom S3 endpoint URL (for MinIO or other S3-compatible services)",
    )

    # JWT Authentication
    jwt_secret: str = Field(
        default="change-me-in-production",
        description="JWT secret key for token signing",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm",
    )
    jwt_expiration_hours: int = Field(
        default=24,
        description="JWT token expiration time in hours",
    )

    # API Keys (research mode)
    api_keys: List[str] = Field(
        default_factory=list,
        description="List of valid API keys for research mode authentication",
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        """Parse comma-separated API keys string."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    # CORS
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins",
    )
    allow_credentials: bool = True
    allow_methods: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed methods",
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed headers",
    )

    # API Configuration
    max_upload_size_mb: int = Field(
        default=500,
        description="Maximum upload size in MB",
    )
    max_concurrent_jobs: int = Field(
        default=10,
        description="Maximum concurrent segmentation jobs",
    )
    job_timeout_minutes: int = Field(
        default=30,
        description="Job timeout in minutes",
    )

    # Orthanc (PACS) Configuration
    orthanc_url: Optional[str] = Field(
        default=None,
        description="Orthanc PACS URL",
    )
    orthanc_username: str = Field(
        default="orthanc",
        description="Orthanc username",
    )
    orthanc_password: str = Field(
        default="orthanc",
        description="Orthanc password",
    )

    # Pipeline Configuration
    pipeline_device: str = Field(
        default="cuda",
        description="Device for pipeline inference (cuda, cpu, mps)",
    )
    pipeline_analysis_mode: str = Field(
        default="auto",
        description="Analysis mode: glioma, mets, or auto",
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database URL uses an async driver."""
        if not any(v.startswith(d) for d in _ASYNC_DRIVERS):
            raise ValueError(
                f"DATABASE_URL must use one of: {', '.join(_ASYNC_DRIVERS)}"
            )
        return v

    @field_validator("storage_backend")
    @classmethod
    def validate_storage_backend(cls, v: str) -> str:
        """Ensure storage backend is valid."""
        if v not in ["local", "s3"]:
            raise ValueError("STORAGE_BACKEND must be 'local' or 's3'")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is valid."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("ENVIRONMENT must be 'development', 'staging', or 'production'")
        return v


# Global settings instance
settings = Settings()
