from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True)
    role = Column(String)
    password_hash = Column(String)
    
    sessions = relationship("ExamSession", back_populates="student")


class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True, index=True)

    student_id = Column(Integer, ForeignKey("users.id"))

    start_time = Column(DateTime)

    end_time = Column(DateTime, nullable=True)

    status = Column(String, default="ACTIVE")

    student = relationship("User", back_populates="sessions")

    violations = relationship(
        "Violation",
        back_populates="session"
    )


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(Integer, ForeignKey("exam_sessions.id"))

    violation_type = Column(String)

    confidence = Column(Float)

    timestamp = Column(DateTime)

    session = relationship(
        "ExamSession",
        back_populates="violations"
    )