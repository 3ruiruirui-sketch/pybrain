"""
Storage abstraction for local filesystem and S3.
All patient data goes here, never in the database.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO
import logging

from pybrain.api.config import settings

logger = logging.getLogger(__name__)


class StorageBackend:
    """Abstract base class for storage backends."""

    async def save_file(self, path: str, content: BinaryIO) -> str:
        """Save a file to storage."""
        raise NotImplementedError

    async def get_file(self, path: str) -> BinaryIO:
        """Get a file from storage."""
        raise NotImplementedError

    async def delete_file(self, path: str) -> None:
        """Delete a file from storage."""
        raise NotImplementedError

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in storage."""
        raise NotImplementedError

    async def get_file_url(self, path: str) -> str:
        """Get a URL for the file (if applicable)."""
        raise NotImplementedError


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, path: str) -> Path:
        """Get the full filesystem path for a relative path."""
        return self.base_path / path

    async def save_file(self, path: str, content: BinaryIO) -> str:
        """Save a file to local storage."""
        full_path = self._get_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "wb") as f:
            shutil.copyfileobj(content, f)
        
        logger.info(f"Saved file to local storage: {full_path}")
        return str(full_path)

    async def get_file(self, path: str) -> BinaryIO:
        """Get a file from local storage."""
        full_path = self._get_full_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        return open(full_path, "rb")

    async def delete_file(self, path: str) -> None:
        """Delete a file from local storage."""
        full_path = self._get_full_path(path)
        if full_path.exists():
            full_path.unlink()
            logger.info(f"Deleted file from local storage: {full_path}")

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in local storage."""
        return self._get_full_path(path).exists()

    async def get_file_url(self, path: str) -> str:
        """Get a URL for the file (not applicable for local storage)."""
        return f"file://{self._get_full_path(path)}"


class S3Storage(StorageBackend):
    """S3 storage backend using boto3."""

    def __init__(
        self,
        bucket: str,
        region: str,
        endpoint_url: Optional[str] = None,
    ):
        try:
            import boto3
            from botocore.client import Config
        except ImportError:
            raise ImportError("boto3 is required for S3 storage")

        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url

        # Initialize S3 client
        config_kwargs = {"region_name": region}
        if endpoint_url:
            config_kwargs["endpoint_url"] = endpoint_url
            # Use path-style addressing for MinIO and other S3-compatible services
            config_kwargs["config"] = Config(s3={"addressing_style": "path"})

        self.s3_client = boto3.client("s3", **config_kwargs)

    async def save_file(self, path: str, content: BinaryIO) -> str:
        """Save a file to S3 storage."""
        import asyncio

        def _upload():
            self.s3_client.upload_fileobj(
                content,
                self.bucket,
                path,
            )

        await asyncio.to_thread(_upload)
        logger.info(f"Saved file to S3: s3://{self.bucket}/{path}")
        return f"s3://{self.bucket}/{path}"

    async def get_file(self, path: str) -> BinaryIO:
        """Get a file from S3 storage."""
        import asyncio
        from io import BytesIO

        def _download():
            buffer = BytesIO()
            self.s3_client.download_fileobj(self.bucket, path, buffer)
            buffer.seek(0)
            return buffer

        return await asyncio.to_thread(_download)

    async def delete_file(self, path: str) -> None:
        """Delete a file from S3 storage."""
        import asyncio

        def _delete():
            self.s3_client.delete_object(Bucket=self.bucket, Key=path)

        await asyncio.to_thread(_delete)
        logger.info(f"Deleted file from S3: s3://{self.bucket}/{path}")

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in S3 storage."""
        import asyncio

        def _exists():
            try:
                self.s3_client.head_object(Bucket=self.bucket, Key=path)
                return True
            except self.s3_client.exceptions.ClientError:
                return False

        return await asyncio.to_thread(_exists)

    async def get_file_url(self, path: str) -> str:
        """Get a presigned URL for the file."""
        import asyncio

        def _generate_url():
            from botocore.client import Config
            from datetime import timedelta

            config_kwargs = {}
            if self.endpoint_url:
                config_kwargs["Config"] = Config(s3={"addressing_style": "path"})

            return self.s3_client.generate_presigned_url(
                "get_object",
                Bucket=self.bucket,
                Key=path,
                ExpiresIn=3600,  # 1 hour
                **config_kwargs,
            )

        return await asyncio.to_thread(_generate_url)


def get_storage() -> StorageBackend:
    """Get the configured storage backend."""
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET must be set when STORAGE_BACKEND is 's3'")
        return S3Storage(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
        )
    else:
        return LocalStorage(base_path=settings.storage_path)


# Global storage instance
storage = get_storage()
