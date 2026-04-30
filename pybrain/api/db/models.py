"""SQLAlchemy database models."""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, JSON, Text, Boolean, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pybrain.api.db.base import Base


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    cases: Mapped[list["Case"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    jobs: Mapped[list["Job"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[list["AuditLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Case(Base):
    """Case model for patient data and analysis results."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    patient_name: Mapped[Optional[str]] = mapped_column(String(255))
    patient_age: Mapped[Optional[int]] = mapped_column(Integer)
    patient_sex: Mapped[Optional[str]] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)  # pending, processing, completed, failed
    analysis_mode: Mapped[str] = mapped_column(String(50), default="auto")  # glioma, mets, auto

    # Storage paths (references to storage system, not actual data)
    storage_path: Mapped[str] = mapped_column(String(500))
    segmentation_path: Mapped[Optional[str]] = mapped_column(String(500))
    report_path: Mapped[Optional[str]] = mapped_column(String(500))
    dicom_seg_path: Mapped[Optional[str]] = mapped_column(String(500))
    dicom_sr_path: Mapped[Optional[str]] = mapped_column(String(500))

    # Results
    volumes: Mapped[Optional[dict]] = mapped_column(JSON)  # {"wt_cc": ..., "tc_cc": ..., "et_cc": ...}
    mets_result: Mapped[Optional[dict]] = mapped_column(JSON)  # Mets-specific results
    longitudinal_result: Mapped[Optional[dict]] = mapped_column(JSON)  # Longitudinal comparison results

    # Metadata
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)  # Soft delete

    # Relationships
    user: Mapped["User"] = relationship(back_populates="cases")
    jobs: Mapped[list["Job"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    prior_cases: Mapped[list["LongitudinalLink"]] = relationship(
        foreign_keys="LongitudinalLink.current_case_id",
        back_populates="current_case",
    )
    subsequent_cases: Mapped[list["LongitudinalLink"]] = relationship(
        foreign_keys="LongitudinalLink.prior_case_id",
        back_populates="prior_case",
    )


class Job(Base):
    """Job model for tracking async tasks."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    case_id: Mapped[Optional[str]] = mapped_column(ForeignKey("cases.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    job_type: Mapped[str] = mapped_column(String(50), index=True)  # segment, longitudinal, export_dicom
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)  # pending, started, success, failure
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100

    # Task-specific parameters
    parameters: Mapped[Optional[dict]] = mapped_column(JSON)

    # Results
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Celery task info
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="jobs")
    case: Mapped["Case"] = relationship(back_populates="jobs")


class LongitudinalLink(Base):
    """Model linking prior and current cases for longitudinal analysis."""

    __tablename__ = "longitudinal_links"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prior_case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    current_case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    comparison_result: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    prior_case: Mapped["Case"] = relationship(
        foreign_keys=[prior_case_id],
        back_populates="subsequent_cases",
    )
    current_case: Mapped["Case"] = relationship(
        foreign_keys=[current_case_id],
        back_populates="prior_cases",
    )


class AuditLog(Base):
    """Audit log model for HIPAA compliance."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(50), index=True)  # create, read, update, delete
    resource_type: Mapped[str] = mapped_column(String(50), index=True)  # case, job, user
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)

    # Request info
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(String(500))

    # Changes
    old_values: Mapped[Optional[dict]] = mapped_column(JSON)
    new_values: Mapped[Optional[dict]] = mapped_column(JSON)

    # Metadata
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="audit_logs")
