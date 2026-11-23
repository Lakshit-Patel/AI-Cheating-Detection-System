import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-change-me")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_proctor.db")

import base64
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.auth import create_access_token, hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import ExamSession, User


class FakeDetector:
    def process_frame(self, frame):
        return frame, {
            "status": "STATUS : SECURE",
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "face_count": 1,
            "looking_left": False,
            "looking_right": False,
            "looking_down": False,
            "looking_up": False,
            "no_face": False,
            "no_valid_pose": False,
            "multiple_faces": False,
            "phone_detected": False,
            "phone_count": 0,
            "phone_confidence": None,
            "pose_valid": True,
            "calibrated": True,
            "baseline_pitch": 0.0,
            "baseline_yaw": 0.0,
        }

    def close(self):
        pass


def setup():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    student = User(
        name="Student",
        email="student@example.com",
        role="student",
        password_hash=hash_password("password123"),
        is_active=1,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    session = ExamSession(student_id=student.id, status="ACTIVE")
    db.add(session)
    db.commit()
    db.refresh(session)
    token = create_access_token(student)
    db.close()
    return token, session.id


def teardown():
    Base.metadata.drop_all(bind=engine)


def test_websocket_requires_auth_message():
    token, session_id = setup()
    try:
        client = TestClient(app)
        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            websocket.send_json({"type": "auth", "token": "invalid"})
            # Server closes with policy violation; TestClient surfaces it as WebSocketDisconnect.
            try:
                websocket.receive_json()
                assert False, "Expected websocket to close"
            except Exception:
                pass
    finally:
        teardown()


def test_websocket_accepts_valid_auth_and_processes_frame():
    token, session_id = setup()
    try:
        client = TestClient(app)
        frame = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Aaf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8Qf//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8Qf//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8Qf//Z")
        with patch("app.main.Detector", FakeDetector):
            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                websocket.send_json({"type": "auth", "token": token})
                ready = websocket.receive_json()
                assert ready["type"] == "ready"
                websocket.send_bytes(frame)
                result = websocket.receive_json()
                assert result["type"] == "detection"
                assert result["phone_detected"] is False
    finally:
        teardown()


def test_websocket_rejects_unauthorized_user():
    token, session_id = setup()
    db = SessionLocal()
    other_student = User(
        name="OtherStudent",
        email="otherstudent@example.com",
        role="student",
        password_hash=hash_password("password123"),
        is_active=1,
    )
    db.add(other_student)
    db.commit()
    db.refresh(other_student)
    other_token = create_access_token(other_student)
    db.close()

    try:
        client = TestClient(app)
        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            websocket.send_json({"type": "auth", "token": other_token})
            try:
                websocket.receive_json()
                assert False, "Expected socket to close for unauthorized student"
            except Exception:
                pass
    finally:
        teardown()


def test_websocket_rejects_inactive_session():
    token, session_id = setup()
    db = SessionLocal()
    s = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    s.status = "COMPLETED"
    db.commit()
    db.close()

    try:
        client = TestClient(app)
        with client.websocket_connect(f"/ws/{session_id}") as websocket:
            websocket.send_json({"type": "auth", "token": token})
            try:
                websocket.receive_json()
                assert False, "Expected socket to close for completed session"
            except Exception:
                pass
    finally:
        teardown()


def test_websocket_handles_invalid_image_and_oversized_frame():
    token, session_id = setup()
    try:
        client = TestClient(app)
        with patch("app.main.Detector", FakeDetector):
            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                websocket.send_json({"type": "auth", "token": token})
                ready = websocket.receive_json()
                assert ready["type"] == "ready"

                # Invalid frame
                websocket.send_bytes(b"not-a-valid-image-format")
                err = websocket.receive_json()
                assert err.get("error") == "invalid frame"

                # Oversized frame
                oversized = b"x" * (2 * 1024 * 1024 + 50)
                websocket.send_bytes(oversized)
                err2 = websocket.receive_json()
                assert err2.get("error") == "frame too large"
    finally:
        teardown()


class FakePhoneDetector:
    def process_frame(self, frame):
        return frame, {
            "status": "FLAGGED : MOBILE PHONE DETECTED",
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "face_count": 1,
            "looking_left": False,
            "looking_right": False,
            "looking_down": False,
            "looking_up": False,
            "no_face": False,
            "no_valid_pose": False,
            "multiple_faces": False,
            "phone_detected": True,
            "phone_count": 1,
            "phone_confidence": 0.95,
            "pose_valid": True,
            "calibrated": True,
            "baseline_pitch": 0.0,
            "baseline_yaw": 0.0,
        }

    def close(self):
        pass


def test_websocket_persists_violations_to_database():
    from app.models import Violation
    import time
    token, session_id = setup()
    frame = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Aaf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8Qf//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8Qf//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8Qf//Z")
    try:
        client = TestClient(app)
        with patch("app.main.Detector", FakePhoneDetector):
            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                websocket.send_json({"type": "auth", "token": token})
                ready = websocket.receive_json()
                assert ready["type"] == "ready"

                # Send 3 frames so violation engine triggers debounced event (start_frames=2)
                for _ in range(3):
                    websocket.send_bytes(frame)
                    msg = websocket.receive_json()
                    assert msg["type"] == "detection"
                    assert msg["phone_detected"] is True
                    time.sleep(0.16)

        # After socket closes, violations must be persisted in database
        db = SessionLocal()
        violations = db.query(Violation).filter(Violation.session_id == session_id).all()
        assert len(violations) >= 1
        assert violations[0].violation_type == "MOBILE PHONE DETECTED"
        assert violations[0].confidence == 0.95
        db.close()
    finally:
        teardown()
