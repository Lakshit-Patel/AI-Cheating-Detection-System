"""Session lifecycle + websocket resilience. NOTE: written against the real FastAPI app;
requires the pinned requirements to be installed (not runnable in the offline authoring sandbox)."""
import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-change-me")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_proctor.db")

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.auth import create_access_token, hash_password
from app.database import Base, SessionLocal, engine
from app.main import LIVE_SESSIONS, app
from app.models import ExamSession, User, Violation


def _setup():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    u = User(name="S", email="s@example.com", role="student", password_hash=hash_password("password123"), is_active=1)
    db.add(u)
    db.commit()
    db.refresh(u)
    tok = create_access_token(u)
    db.close()
    return u.id, tok


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
    return buf.tobytes()


def test_abandoned_active_session_is_cancelled_so_student_is_not_locked_out():
    uid, tok = _setup()
    db = SessionLocal()
    old = ExamSession(student_id=uid, status="ACTIVE", start_time=datetime.now(UTC) - timedelta(minutes=5))
    db.add(old)
    db.commit()
    db.add(Violation(session_id=old.id, violation_type="NO FACE DETECTED", timestamp=datetime.now(UTC) - timedelta(minutes=4)))
    db.commit()
    old_id = old.id
    db.close()

    r = TestClient(app).post("/api/v1/sessions", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["session_id"] != old_id

    db = SessionLocal()
    old = db.get(ExamSession, old_id)
    assert old.status == "CANCELLED" and old.end_time is not None
    v = db.query(Violation).filter_by(session_id=old_id).one()
    assert v.duration_seconds is not None and v.duration_seconds >= 0   # open violation was finalised
    db.close()


def test_live_session_still_returns_409():
    uid, tok = _setup()
    h = {"Authorization": f"Bearer {tok}"}
    c = TestClient(app)
    first = c.post("/api/v1/sessions", headers=h).json()["session_id"]
    LIVE_SESSIONS.add(first)
    try:
        assert c.post("/api/v1/sessions", headers=h).status_code == 409
    finally:
        LIVE_SESSIONS.discard(first)


class ExplodingDetector:
    def process_frame(self, frame):
        raise ValueError("boom")

    def close(self):
        pass


def test_detector_failure_is_reported_to_browser_and_socket_survives():
    uid, tok = _setup()
    sid = TestClient(app).post("/api/v1/sessions", headers={"Authorization": f"Bearer {tok}"}).json()["session_id"]
    with patch("app.main.Detector", ExplodingDetector):
        with TestClient(app).websocket_connect(f"/ws/{sid}") as ws:
            ws.send_json({"type": "auth", "token": tok})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_jpeg())
            msg = ws.receive_json()
            assert msg["type"] == "error" and msg["code"] == "processing_failed"
            ws.send_bytes(_jpeg())                     # still alive after a failure
            assert ws.receive_json()["code"] == "processing_failed"
    assert sid not in LIVE_SESSIONS


def test_text_frame_gets_clear_error():
    uid, tok = _setup()
    sid = TestClient(app).post("/api/v1/sessions", headers={"Authorization": f"Bearer {tok}"}).json()["session_id"]
    with patch("app.main.Detector", ExplodingDetector):
        with TestClient(app).websocket_connect(f"/ws/{sid}") as ws:
            ws.send_json({"type": "auth", "token": tok})
            ws.receive_json()
            ws.send_text("hello")
            assert ws.receive_json()["code"] == "binary_only"
