import base64
import logging
import traceback

import cv2
import numpy as np

from datetime import datetime, UTC

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    File,
    UploadFile,
    Request,
)
from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy import func
from sqlalchemy.orm import joinedload

from .database import engine, SessionLocal
from .models import Base, User, ExamSession, Violation
from .detector import Detector
from .schemas import UserRegister

from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    verify_access_token,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proctor")

app = FastAPI(
    title="Online Cheating Preventer API",
    docs_url=None,
    redoc_url=None
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

Base.metadata.create_all(bind=engine)


# ============================================================
# DEFAULT USERS
# ============================================================

db = SessionLocal()

student = (
    db.query(User)
    .filter(User.email == "student@test.com")
    .first()
)

if student is None:
    student = User(
        name="Lakshit",
        email="student@test.com",
        role="student",
        password_hash=hash_password("password123")
    )

    db.add(student)
    db.commit()


proctor = (
    db.query(User)
    .filter(User.email == "admin@test.com")
    .first()
)

if proctor is None:
    proctor = User(
        name="Admin",
        email="admin@test.com",
        role="admin",
        password_hash=hash_password("admin123")
    )

    db.add(proctor)
    db.commit()

db.close()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AUTH HELPERS
# ============================================================

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    return payload


def require_admin(current_user=Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user

@app.get("/docs", include_in_schema=False)
async def custom_docs(request: Request):
    token = request.query_params.get("token")

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    payload = verify_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Online Cheating Preventer API - Swagger"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ============================================================
# REGISTER
# ============================================================

@app.post("/register")
def register(user: UserRegister):
    db = SessionLocal()

    existing = (
        db.query(User)
        .filter(User.email == user.email)
        .first()
    )

    if existing:
        db.close()

        return {
            "message": "Email already exists"
        }

    new_user = User(
        name=user.name,
        email=user.email,
        role="student",
        password_hash=hash_password(user.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    db.close()

    return {
        "message": "User registered successfully",
        "user_id": new_user.id
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    db = SessionLocal()

    existing = (
        db.query(User)
        .filter(User.email == form_data.username)
        .first()
    )

    if existing is None:
        db.close()

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        form_data.password,
        existing.password_hash
    ):
        db.close()

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(
        {
            "sub": str(existing.id),
            "role": existing.role
        }
    )

    db.close()

    return {
        "access_token": token,
        "token_type": "bearer"
    }


# ============================================================
# ADMIN — ALL SESSIONS
# ============================================================

@app.get("/sessions")
def get_sessions(
    current_user=Depends(require_admin)
):
    db = SessionLocal()

    sessions = (
        db.query(ExamSession)
        .options(joinedload(ExamSession.student))
        .all()
    )

    result = []

    for session in sessions:
        result.append({
            "id": session.id,
            "student": session.student.name,
            "student_email": session.student.email,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "status": session.status
        })

    db.close()

    return result


# ============================================================
# CREATE SESSION
# ============================================================

@app.post("/api/v1/sessions")
def create_session(
    current_user=Depends(get_current_user)
):
    if current_user["role"] != "student":
        raise HTTPException(
            status_code=403,
            detail="Only students can create exam sessions"
        )

    db = SessionLocal()

    session = ExamSession(
        student_id=int(current_user["sub"]),
        start_time=datetime.now(UTC),
        status="ACTIVE"
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    result = {
        "session_id": session.id,
        "student_id": session.student_id,
        "start_time": session.start_time,
        "status": session.status
    }

    db.close()

    return result


# ============================================================
# GET SESSION
# ============================================================

@app.get("/api/v1/sessions/{session_id}")
def get_session(
    session_id: int,
    current_user=Depends(get_current_user)
):
    db = SessionLocal()

    session = (
        db.query(ExamSession)
        .filter(ExamSession.id == session_id)
        .first()
    )

    if session is None:
        db.close()

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    if (
        current_user["role"] != "admin"
        and session.student_id != int(current_user["sub"])
    ):
        db.close()

        raise HTTPException(
            status_code=403,
            detail="Not authorized to view this session"
        )

    result = {
        "session_id": session.id,
        "student_id": session.student_id,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "status": session.status,
        "violation_count": len(session.violations)
    }

    db.close()

    return result


# ============================================================
# END SESSION
# ============================================================

@app.delete("/api/v1/sessions/{session_id}")
def end_session(
    session_id: int,
    current_user=Depends(get_current_user)
):
    db = SessionLocal()

    session = (
        db.query(ExamSession)
        .filter(ExamSession.id == session_id)
        .first()
    )

    if session is None:
        db.close()

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    if (
        current_user["role"] != "admin"
        and session.student_id != int(current_user["sub"])
    ):
        db.close()

        raise HTTPException(
            status_code=403,
            detail="Not authorized to end this session"
        )

    session.status = "COMPLETED"
    session.end_time = datetime.now(UTC)

    db.commit()

    result = {
        "session_id": session.id,
        "status": session.status,
        "end_time": session.end_time
    }

    db.close()

    return result


# ============================================================
# SINGLE FRAME DETECTION API
# ============================================================

@app.post("/api/v1/detect")
async def detect_frame(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    contents = await file.read()

    frame = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    detector = Detector()

    try:
        processed_frame, data = detector.process_frame(frame)

        _, buffer = cv2.imencode(
            ".jpg",
            processed_frame
        )

        data["frame"] = base64.b64encode(
            buffer
        ).decode("utf-8")

        return data

    finally:
        detector.close()


# ============================================================
# EVENTS
# ============================================================

@app.get("/api/v1/events/{session_id}")
def get_events(
    session_id: int,
    current_user=Depends(get_current_user)
):
    db = SessionLocal()

    session = (
        db.query(ExamSession)
        .filter(ExamSession.id == session_id)
        .first()
    )

    if session is None:
        db.close()

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    if (
        current_user["role"] != "admin"
        and session.student_id != int(current_user["sub"])
    ):
        db.close()

        raise HTTPException(
            status_code=403,
            detail="Not authorized to view this session"
        )

    violations = (
        db.query(Violation)
        .filter(Violation.session_id == session_id)
        .order_by(Violation.timestamp.asc())
        .all()
    )

    result = []

    for violation in violations:
        result.append({
            "id": violation.id,
            "type": violation.violation_type,
            "confidence": violation.confidence,
            "timestamp": violation.timestamp
        })

    db.close()

    return {
        "session_id": session_id,
        "event_count": len(result),
        "events": result
    }


# ============================================================
# REPORT
# ============================================================

@app.get("/api/v1/reports/{session_id}")
def get_report(
    session_id: int,
    current_user=Depends(get_current_user)
):
    db = SessionLocal()

    session = (
        db.query(ExamSession)
        .options(joinedload(ExamSession.student))
        .filter(ExamSession.id == session_id)
        .first()
    )

    if session is None:
        db.close()

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    if (
        current_user["role"] != "admin"
        and session.student_id != int(current_user["sub"])
    ):
        db.close()

        raise HTTPException(
            status_code=403,
            detail="Not authorized to view this report"
        )

    violations = (
        db.query(Violation)
        .filter(Violation.session_id == session_id)
        .all()
    )

    violation_counts = {}

    for violation in violations:

        violation_type = violation.violation_type

        clean_type = violation_type.replace(
            "FLAGGED: ",
            ""
        )

        violation_counts[clean_type] = (
            violation_counts.get(clean_type, 0) + 1
        )

    duration_seconds = None

    if session.start_time:

        end_time = (
            session.end_time
            or datetime.now(UTC)
        )

        duration_seconds = int(
            (
                end_time - session.start_time
            ).total_seconds()
        )

    total_violations = len(violations)

    risk_score = min(
        total_violations * 10,
        100
    )

    if risk_score >= 70:
        risk_level = "HIGH"

    elif risk_score >= 30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    result = {
        "session_id": session.id,
        "student": session.student.name,
        "student_email": session.student.email,
        "status": session.status,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "duration_seconds": duration_seconds,
        "total_violations": total_violations,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "violations": violation_counts
    }

    db.close()

    return result


# ============================================================
# VIOLATIONS
# ============================================================

@app.get("/api/v1/sessions/{session_id}/violations")
def get_violations(
    session_id: int,
    current_user=Depends(require_admin)
):
    db = SessionLocal()

    violations = (
        db.query(Violation)
        .filter(Violation.session_id == session_id)
        .all()
    )

    result = []

    for violation in violations:
        result.append({
            "id": violation.id,
            "type": violation.violation_type,
            "confidence": violation.confidence,
            "timestamp": violation.timestamp
        })

    db.close()

    return result


# ============================================================
# STATISTICS
# ============================================================

@app.get("/statistics")
def get_statistics(
    current_user=Depends(require_admin)
):
    db = SessionLocal()

    total_sessions = (
        db.query(ExamSession).count()
    )

    total_violations = (
        db.query(Violation).count()
    )

    phone_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("PHONE")
        )
        .count()
    )

    no_face_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("NO FACE")
        )
        .count()
    )

    multiple_face_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("MULTIPLE")
        )
        .count()
    )

    looking_left_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("LOOKING LEFT")
        )
        .count()
    )

    looking_right_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("LOOKING RIGHT")
        )
        .count()
    )

    looking_up_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("LOOKING UP")
        )
        .count()
    )

    looking_down_count = (
        db.query(Violation)
        .filter(
            Violation.violation_type.contains("LOOKING DOWN")
        )
        .count()
    )

    db.close()

    return {
        "total_sessions": total_sessions,
        "total_violations": total_violations,
        "phone_detections": phone_count,
        "no_face_detections": no_face_count,
        "multiple_face_detections": multiple_face_count,
        "looking_left": looking_left_count,
        "looking_right": looking_right_count,
        "looking_up": looking_up_count,
        "looking_down": looking_down_count
    }


# ============================================================
# WEBSOCKET PROCTORING
# ============================================================

@app.websocket("/ws/{session_id}")
async def proctor_stream(
    websocket: WebSocket,
    session_id: str,
    token: str
):
    """
    Real-time proctoring stream.

    The session_id must correspond to an existing
    ExamSession created through POST /api/v1/sessions.
    """

    await websocket.accept()

    payload = verify_access_token(token)

    if payload is None:
        await websocket.close(code=1008)
        return

    logger.info(
        f"Session {session_id} connected"
    )

    detector = Detector()
    db = SessionLocal()

    # --------------------------------------------------------
    # FIND EXISTING SESSION
    # --------------------------------------------------------

    try:
        session_id_int = int(session_id)
    except ValueError:
        db.close()
        detector.close()

        await websocket.close(code=1008)
        return

    exam_session = (
        db.query(ExamSession)
        .filter(
            ExamSession.id == session_id_int
        )
        .first()
    )

    if exam_session is None:
        logger.warning(
            f"Session {session_id} does not exist"
        )

        db.close()
        detector.close()

        await websocket.close(code=1008)
        return

    # --------------------------------------------------------
    # CHECK OWNERSHIP
    # --------------------------------------------------------

    if (
        payload["role"] != "admin"
        and exam_session.student_id != int(payload["sub"])
    ):
        logger.warning(
            f"Unauthorized access to session {session_id}"
        )

        db.close()
        detector.close()

        await websocket.close(code=1008)
        return

    # --------------------------------------------------------
    # VIOLATION COOLDOWN
    # --------------------------------------------------------

    last_logged = {}

    COOLDOWN_SECONDS = 5

    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    try:

        while True:

            raw_bytes = (
                await websocket.receive_bytes()
            )

            logger.info(
                f"Session {session_id}: "
                f"received frame ({len(raw_bytes)} bytes)"
            )

            frame = cv2.imdecode(
                np.frombuffer(
                    raw_bytes,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is None:

                await websocket.send_json({
                    "error": "invalid frame"
                })

                continue

            try:

                processed_frame, data = (
                    detector.process_frame(frame)
                )

            except Exception:

                logger.exception(
                    f"Session {session_id}: "
                    f"frame processing failed"
                )

                await websocket.send_json({
                    "error": "frame processing failed"
                })

                continue

            # ------------------------------------------------
            # ENCODE PROCESSED FRAME
            # ------------------------------------------------

            _, buffer = cv2.imencode(
                ".jpg",
                processed_frame
            )

            data["frame"] = base64.b64encode(
                buffer
            ).decode("utf-8")

            # ------------------------------------------------
            # LOG VIOLATIONS
            # ------------------------------------------------

            status = data.get(
                "status",
                ""
            )

            if status.startswith("FLAGGED"):

                now = datetime.now(UTC)

                should_log = (
                    status not in last_logged
                    or
                    (
                        now - last_logged[status]
                    ).total_seconds()
                    >= COOLDOWN_SECONDS
                )

                if should_log:

                    violation = Violation(
                        session_id=exam_session.id,
                        violation_type=status,
                        confidence=1.0,
                        timestamp=now
                    )

                    db.add(violation)
                    db.commit()

                    last_logged[status] = now

            # ------------------------------------------------
            # SEND RESULT
            # ------------------------------------------------

            await websocket.send_json(data)

    except WebSocketDisconnect:

        logger.info(
            f"Session {session_id} disconnected"
        )

    except Exception:

        logger.error(
            f"Session {session_id} crashed:\n"
            f"{traceback.format_exc()}"
        )

        try:
            await websocket.close(
                code=1011
            )

        except RuntimeError:
            pass

    finally:

        if db:
            db.close()

        detector.close()

        logger.info(
            f"Session {session_id} WebSocket cleanup complete"
        )