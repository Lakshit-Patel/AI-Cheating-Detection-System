from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field

class UserRegister(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: EmailStr
    role: str
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20)

class LogoutRequest(BaseModel):
    refresh_token: str = Field(min_length=20)

class SessionListItem(BaseModel):
    id: int
    student: str
    student_email: EmailStr
    start_time: datetime
    end_time: datetime | None = None
    status: str

class SessionCreateResponse(BaseModel):
    session_id: int
    student_id: int
    start_time: datetime
    status: str

class SessionDetailResponse(BaseModel):
    session_id: int
    student_id: int
    start_time: datetime
    end_time: datetime | None = None
    status: str
    violation_count: int

class SessionEndResponse(BaseModel):
    session_id: int
    status: str
    end_time: datetime | None = None

class ViolationItem(BaseModel):
    id: int
    type: str
    confidence: float | None = None
    timestamp: datetime
    duration_seconds: float | None = None
    details: str | None = None

class EventsResponse(BaseModel):
    session_id: int
    event_count: int
    events: list[ViolationItem]

class ReportResponse(BaseModel):
    session_id: int
    student: str
    student_email: EmailStr
    status: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: int | None = None
    total_violations: int
    violations: dict[str, int]
    risk_score: float | None = None
    risk_level: str | None = None
    risk_note: str

class StatisticsResponse(BaseModel):
    total_sessions: int
    total_violations: int
    phone_detections: int
    no_face_detections: int
    multiple_face_detections: int
    looking_left: int
    looking_right: int
    looking_up: int
    looking_down: int
