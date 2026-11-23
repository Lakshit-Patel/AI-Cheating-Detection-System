from datetime import datetime, UTC

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


def utc_now():
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(320), nullable=False, unique=True, index=True)
    role = Column(String(20), nullable=False, default="student")
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    __table_args__ = (
        CheckConstraint("role IN ('student', 'admin')", name="ck_users_role"),
        CheckConstraint("is_active IN (0, 1)", name="ck_users_active"),
    )

    sessions = relationship(
        "ExamSession",
        back_populates="student",
        cascade="all, delete-orphan",
    )



class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    family_id = Column(String(36), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    replaced_by_hash = Column(String(64), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user = relationship("User")

class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True)
    student_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    start_time = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="ACTIVE")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    __table_args__ = (
        CheckConstraint(
            "status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')",
            name="ck_exam_sessions_status",
        ),
        Index("ix_exam_sessions_student_status", "student_id", "status"),
        Index("ix_exam_sessions_start_time", "start_time"),
    )

    student = relationship("User", back_populates="sessions")
    violations = relationship(
        "Violation",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        Integer,
        ForeignKey("exam_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    violation_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    duration_seconds = Column(Float, nullable=True)
    details = Column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_violations_confidence",
        ),
        Index("ix_violations_session_timestamp", "session_id", "timestamp"),
        Index("ix_violations_type_timestamp", "violation_type", "timestamp"),
    )

    session = relationship("ExamSession", back_populates="violations")
