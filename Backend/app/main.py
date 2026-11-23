from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.openapi.docs import get_swagger_ui_html
from sqlalchemy import func, text
from sqlalchemy.orm import Session, joinedload

from .auth import decode_access_token
from .config import APP_ENV, APP_NAME, CORS_ORIGINS, MAX_FRAME_BYTES
from .database import SessionLocal, get_db
from .detector import Detector
from .model_manager import model_manager
from .models import ExamSession, User, Violation
from .security_dependencies import get_current_user, require_admin
from .auth_routes import router as auth_router
from .violation_engine import ViolationEngine
from .rate_limit import limiter
from .schemas import (
    EventsResponse,
    ReportResponse,
    SessionCreateResponse,
    SessionDetailResponse,
    SessionEndResponse,
    SessionListItem,
    StatisticsResponse,
    ViolationItem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proctor")


# Sessions that currently have a live monitoring WebSocket (per process; single-worker design).
LIVE_SESSIONS: set[int] = set()
NEW_SESSION_GRACE_SECONDS = 15
MAX_CONSECUTIVE_FRAME_FAILURES = 10


def _close_open_violations(db: Session, session_id: int, end: datetime) -> int:
    """Give every still-open violation of a session a duration. Returns rows closed."""
    rows = db.query(Violation).filter(
        Violation.session_id == session_id, Violation.duration_seconds.is_(None)
    ).all()
    for row in rows:
        start = _to_utc(row.timestamp)
        row.duration_seconds = max(0.0, (end - start).total_seconds()) if start else 0.0
    return len(rows)


def _abandon_session(db: Session, session: ExamSession, now: datetime) -> None:
    session.status = "CANCELLED"
    session.end_time = now
    _close_open_violations(db, session.id, now)


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        # A fresh process has no live sockets, so any ACTIVE row is an orphan from a
        # crash/restart. Close it (CANCELLED) so students are not locked out by a 409.
        now = datetime.now(UTC)
        orphans = db.query(ExamSession).filter(ExamSession.status == "ACTIVE").all()
        for orphan in orphans:
            _abandon_session(db, orphan, now)
        if orphans:
            db.commit()
            logger.warning("Closed %d orphaned ACTIVE session(s) at startup", len(orphans))
        yield
    finally:
        db.close()
        model_manager.close()


app = FastAPI(
    title=APP_NAME,
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(auth_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    import uuid
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "environment": APP_ENV}


@app.get("/ready", tags=["system"])
def ready_check(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ready", "phone_model": model_manager.status()}


@app.get("/docs", include_in_schema=False)
async def custom_docs(request: Request, current_user: User = Depends(require_admin)):
    if APP_ENV == "production":
        raise HTTPException(status_code=404, detail="Not found")
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{APP_NAME} - Swagger")


@app.get("/sessions", response_model=list[SessionListItem], tags=["admin"])
def get_sessions(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    sessions = db.query(ExamSession).options(joinedload(ExamSession.student)).order_by(ExamSession.start_time.desc()).all()
    return [
        {
            "id": s.id,
            "student": s.student.name,
            "student_email": s.student.email,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "status": s.status,
        }
        for s in sessions
    ]


@app.post("/api/v1/sessions", response_model=SessionCreateResponse, tags=["sessions"])
def create_session(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can create exam sessions")

    now = datetime.now(UTC)
    for active in db.query(ExamSession).filter(
        ExamSession.student_id == current_user.id,
        ExamSession.status == "ACTIVE",
    ).all():
        started = _to_utc(active.start_time) or now
        is_live = active.id in LIVE_SESSIONS or (now - started).total_seconds() < NEW_SESSION_GRACE_SECONDS
        if is_live:
            raise HTTPException(status_code=409, detail=f"Student already has active session {active.id}")
        _abandon_session(db, active, now)   # no live connection: the browser died / WS failed

    session = ExamSession(student_id=current_user.id, start_time=now, status="ACTIVE")
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "session_id": session.id,
        "student_id": session.student_id,
        "start_time": session.start_time,
        "status": session.status,
    }


def _get_session_for_user(session_id: int, user: User, db: Session) -> ExamSession:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if user.role != "admin" and session.student_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    return session


@app.get("/api/v1/sessions/{session_id}", response_model=SessionDetailResponse, tags=["sessions"])
def get_session(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = _get_session_for_user(session_id, current_user, db)
    violation_count = db.query(func.count(Violation.id)).filter(Violation.session_id == session.id).scalar() or 0
    return {
        "session_id": session.id,
        "student_id": session.student_id,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "status": session.status,
        "violation_count": violation_count,
    }


@app.delete("/api/v1/sessions/{session_id}", response_model=SessionEndResponse, tags=["sessions"])
def end_session(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = _get_session_for_user(session_id, current_user, db)
    if session.status != "ACTIVE":
        return {"session_id": session.id, "status": session.status, "end_time": session.end_time}
    session.status = "COMPLETED"
    session.end_time = datetime.now(UTC)
    _close_open_violations(db, session.id, session.end_time)
    db.commit()
    return {"session_id": session.id, "status": session.status, "end_time": session.end_time}


@app.post("/api/v1/detect")
async def detect_frame(file: UploadFile = File(...), _: User = Depends(get_current_user)):
    contents = await file.read()
    if len(contents) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    detector = Detector()
    try:
        processed_frame, data = await asyncio.to_thread(detector.process_frame, frame)
        ok, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode result")
        data["frame"] = base64.b64encode(buffer).decode("ascii")
        return data
    finally:
        detector.close()


@app.get("/api/v1/events/{session_id}", response_model=EventsResponse, tags=["events"])
def get_events(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _get_session_for_user(session_id, current_user, db)
    violations = db.query(Violation).filter(Violation.session_id == session_id).order_by(Violation.timestamp.asc()).all()
    return {
        "session_id": session_id,
        "event_count": len(violations),
        "events": [
            {
                "id": v.id,
                "type": v.violation_type,
                "confidence": v.confidence,
                "timestamp": v.timestamp,
                "duration_seconds": v.duration_seconds,
                "details": v.details,
            }
            for v in violations
        ],
    }


@app.get("/api/v1/reports/{session_id}", response_model=ReportResponse, tags=["reports"])
def get_report(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ExamSession).options(joinedload(ExamSession.student)).filter(ExamSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if current_user.role != "admin" and session.student_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this report")

    violations = db.query(Violation).filter(Violation.session_id == session_id).all()
    violation_counts = {}
    for v in violations:
        violation_counts[v.violation_type] = violation_counts.get(v.violation_type, 0) + 1

    start_utc = _to_utc(session.start_time)
    end_utc = _to_utc(session.end_time) or datetime.now(UTC)
    duration = int((end_utc - start_utc).total_seconds()) if start_utc else None
    return {
        "session_id": session.id,
        "student": session.student.name,
        "student_email": session.student.email,
        "status": session.status,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "duration_seconds": duration,
        "total_violations": len(violations),
        "violations": violation_counts,
        "risk_score": None,
        "risk_level": None,
        "risk_note": "No composite cheating score is assigned; violations are reported as observed events.",
    }


@app.get("/api/v1/sessions/{session_id}/violations", response_model=list[ViolationItem], tags=["admin"])
def get_violations(session_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    session = db.query(ExamSession.id).filter(ExamSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    violations = db.query(Violation).filter(Violation.session_id == session_id).order_by(Violation.timestamp.asc()).all()
    return [
        {
            "id": v.id,
            "type": v.violation_type,
            "confidence": v.confidence,
            "timestamp": v.timestamp,
            "duration_seconds": v.duration_seconds,
            "details": v.details,
        }
        for v in violations
    ]


@app.get("/statistics", response_model=StatisticsResponse, tags=["admin"])
def get_statistics(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    total_sessions = db.query(ExamSession).count()
    total_violations = db.query(Violation).count()
    def count_like(value: str) -> int:
        return db.query(Violation).filter(Violation.violation_type.contains(value)).count()
    return {
        "total_sessions": total_sessions,
        "total_violations": total_violations,
        "phone_detections": count_like("MOBILE PHONE"),
        "no_face_detections": count_like("NO FACE"),
        "multiple_face_detections": count_like("MULTIPLE"),
        "looking_left": count_like("LOOKING LEFT"),
        "looking_right": count_like("LOOKING RIGHT"),
        "looking_up": count_like("LOOKING UP"),
        "looking_down": count_like("LOOKING DOWN"),
    }


async def _authenticate_websocket(websocket: WebSocket) -> User | None:
    """Authenticate with a first JSON message so the JWT is not placed in the URL."""
    try:
        message = await asyncio.wait_for(websocket.receive_json(), timeout=10)
    except (asyncio.TimeoutError, ValueError):
        return None
    if message.get("type") != "auth" or not message.get("token"):
        return None
    payload = decode_access_token(message["token"])
    if payload is None:
        return None
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        return None
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id, User.is_active == 1).first()
    finally:
        db.close()


@app.websocket("/ws/{session_id}")
async def proctor_stream(websocket: WebSocket, session_id: int):
    await websocket.accept()

    user = await _authenticate_websocket(websocket)
    if user is None:
        await websocket.close(code=1008, reason="Authentication required")
        return

    db = SessionLocal()
    detector = None
    receiver_task = None
    processor_task = None
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    stop_event = asyncio.Event()
    violation_engine = ViolationEngine(start_frames=2, clear_frames=3)

    try:
        exam_session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if exam_session is None:
            await websocket.close(code=1008, reason="Session not found")
            return
        if user.role != "admin" and exam_session.student_id != user.id:
            await websocket.close(code=1008, reason="Not authorized")
            return
        if exam_session.status != "ACTIVE":
            await websocket.close(code=1008, reason="Session is not active")
            return

        try:
            detector = await asyncio.to_thread(Detector)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Detector initialisation failed for session %s", session_id)
            await websocket.send_json({"type": "error", "code": "detector_init", "message": str(exc)[:300]})
            await websocket.close(code=1011, reason="Detector unavailable")
            return
        LIVE_SESSIONS.add(session_id)
        await websocket.send_json({"type": "ready", "session_id": session_id})

        async def receive_frames():
            while not stop_event.is_set():
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(msg.get("code", 1000))
                raw = msg.get("bytes")
                if raw is None:
                    await websocket.send_json({"type": "error", "code": "binary_only", "message": "Send JPEG frames as binary messages"})
                    continue
                if len(raw) > MAX_FRAME_BYTES:
                    await websocket.send_json({"type": "error", "code": "frame_too_large", "error": "frame too large", "max_bytes": MAX_FRAME_BYTES})
                    continue
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(raw)

        async def process_frames():
            last_process = 0.0
            min_interval = 0.15  # target <= ~6.7 processed frames/sec per session
            failures = 0
            while not stop_event.is_set():
                raw = await queue.get()
                now_mono = asyncio.get_running_loop().time()
                remaining = min_interval - (now_mono - last_process)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                last_process = asyncio.get_running_loop().time()

                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_json({"type": "error", "code": "invalid_frame", "error": "invalid frame"})
                    continue

                try:
                    processed_frame, data = await asyncio.to_thread(detector.process_frame, frame)
                    failures = 0
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.exception("process_frame failed (%d consecutive) for session %s", failures, session_id)
                    await websocket.send_json({"type": "error", "code": "processing_failed", "message": f"{type(exc).__name__}: {exc}"[:300]})
                    if failures >= MAX_CONSECUTIVE_FRAME_FAILURES:
                        raise RuntimeError("detector keeps failing") from exc
                    continue
                logger.debug("WS processed frame: status=%s", data.get("status"))
                now = datetime.now(UTC)
                started_events, ended_events = violation_engine.update(data, now)

                for event in started_events:
                    row = Violation(
                        session_id=exam_session.id,
                        violation_type=event.violation_type,
                        confidence=event.confidence,
                        timestamp=event.started_at,
                        details=event.details,
                    )
                    db.add(row)
                if started_events:
                    try:
                        db.commit()
                    except Exception:
                        db.rollback()
                        logger.exception("Failed to commit started violations for session %s", session_id)

                for event in ended_events:
                    started_utc = _to_utc(event.started_at)
                    dur = max(0.0, (now - started_utc).total_seconds()) if started_utc else 0.0
                    # Match the currently active DB event by type and open duration.
                    row = (
                        db.query(Violation)
                        .filter(
                            Violation.session_id == exam_session.id,
                            Violation.violation_type == event.violation_type,
                            Violation.duration_seconds.is_(None),
                        )
                        .order_by(Violation.id.desc())
                        .first()
                    )
                    if row is not None:
                        row.duration_seconds = dur
                        row.details = event.details
                if ended_events:
                    try:
                        db.commit()
                    except Exception:
                        db.rollback()
                        logger.exception("Failed to commit ended violations for session %s", session_id)

                ok, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok:
                    await websocket.send_json({"type": "error", "code": "encode_failed", "error": "failed to encode frame"})
                    continue
                data["frame"] = base64.b64encode(buffer).decode("ascii")
                data["type"] = "detection"
                await websocket.send_json(data)

        receiver_task = asyncio.create_task(receive_frames())
        processor_task = asyncio.create_task(process_frames())
        done, pending = await asyncio.wait(
            {receiver_task, processor_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
        stop_event.set()
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception:
        logger.exception("Session %s websocket failed", session_id)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except RuntimeError:
            pass
    finally:
        LIVE_SESSIONS.discard(session_id)
        stop_event.set()
        for task in (receiver_task, processor_task):
            if task and not task.done():
                task.cancel()
        if detector:
            detector.close()
        now = datetime.now(UTC)
        try:
            ended_events = violation_engine.finalize(now)
            for event in ended_events:
                started_utc = _to_utc(event.started_at)
                dur = max(0.0, (now - started_utc).total_seconds()) if started_utc else 0.0
                row = (
                    db.query(Violation)
                    .filter(
                        Violation.session_id == session_id,
                        Violation.violation_type == event.violation_type,
                        Violation.duration_seconds.is_(None),
                    )
                    .order_by(Violation.id.desc())
                    .first()
                )
                if row is not None:
                    row.duration_seconds = dur
                    row.details = event.details
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Error finalizing events for session %s", session_id)
        finally:
            db.close()
            logger.info("Session %s WebSocket cleanup complete", session_id)
