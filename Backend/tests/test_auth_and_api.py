import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-change-me")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_proctor.db")

import pytest
from fastapi.testclient import TestClient

from app.auth import hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import User


@pytest.fixture(autouse=True)
def clean_db():
    app.state.limiter.enabled = False
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    db.add(User(name="Student", email="student@example.com", role="student", password_hash=hash_password("password123"), is_active=1))
    db.add(User(name="Admin", email="admin@example.com", role="admin", password_hash=hash_password("admin1234"), is_active=1))
    db.commit()
    db.close()
    yield
    Base.metadata.drop_all(bind=engine)


def login(client, email, password):
    response = client.post("/login", data={"username": email, "password": password})
    assert response.status_code == 200
    return response.json()


def test_password_hashing_and_long_password_rejection():
    hashed = hash_password("correct horse battery staple")
    assert hashed.startswith("$2b$")
    assert len(hashed) > 50

    too_long = "x" * 73
    with pytest.raises(ValueError):
        hash_password(too_long)


def test_login_refresh_rotation_and_replay_rejection():
    client = TestClient(app)
    tokens = login(client, "student@example.com", "password123")
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] > 0

    rotated = client.post("/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert rotated.status_code == 200
    assert rotated.json()["refresh_token"] != tokens["refresh_token"]

    replay = client.post("/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert replay.status_code == 401


def test_invalid_login_is_rejected():
    client = TestClient(app)
    response = client.post("/login", data={"username": "student@example.com", "password": "wrong"})
    assert response.status_code == 401


def test_student_cannot_access_admin_sessions():
    client = TestClient(app)
    tokens = login(client, "student@example.com", "password123")
    response = client.get("/sessions", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert response.status_code == 403


def test_admin_can_access_admin_sessions():
    client = TestClient(app)
    tokens = login(client, "admin@example.com", "admin1234")
    response = client.get("/sessions", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert response.status_code == 200


def test_student_can_create_only_one_active_session_and_end_it():
    client = TestClient(app)
    tokens = login(client, "student@example.com", "password123")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    first = client.post("/api/v1/sessions", headers=headers)
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    second = client.post("/api/v1/sessions", headers=headers)
    assert second.status_code == 409

    ended = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
    assert ended.status_code == 200
    assert ended.json()["status"] == "COMPLETED"


def test_student_cannot_read_another_students_session():
    client = TestClient(app)
    first = login(client, "student@example.com", "password123")
    headers = {"Authorization": f"Bearer {first['access_token']}"}
    session = client.post("/api/v1/sessions", headers=headers).json()["session_id"]

    db = SessionLocal()
    db.add(User(name="Other", email="other@example.com", role="student", password_hash=hash_password("password123"), is_active=1))
    db.commit()
    db.close()

    other = login(client, "other@example.com", "password123")
    other_headers = {"Authorization": f"Bearer {other['access_token']}"}
    response = client.get(f"/api/v1/sessions/{session}", headers=other_headers)
    assert response.status_code == 403


def test_inactive_account_cannot_login():
    db = SessionLocal()
    db.add(User(name="Disabled", email="disabled@example.com", role="student", password_hash=hash_password("password123"), is_active=0))
    db.commit()
    db.close()

    client = TestClient(app)
    response = client.post("/login", data={"username": "disabled@example.com", "password": "password123"})
    assert response.status_code == 401


def test_malformed_and_expired_tokens_rejected():
    client = TestClient(app)
    # Malformed token
    res = client.get("/api/v1/sessions/1", headers={"Authorization": "Bearer not-a-valid-token"})
    assert res.status_code == 401

    # Expired token
    from datetime import datetime, UTC, timedelta
    from jose import jwt
    from app.config import SECRET_KEY
    expired_payload = {
        "sub": "1",
        "role": "student",
        "type": "access",
        "exp": datetime.now(UTC) - timedelta(hours=1),
    }
    expired_jwt = jwt.encode(expired_payload, SECRET_KEY, algorithm="HS256")
    res2 = client.get("/api/v1/sessions/1", headers={"Authorization": f"Bearer {expired_jwt}"})
    assert res2.status_code == 401


def test_logout_revokes_refresh_token():
    client = TestClient(app)
    tokens = login(client, "student@example.com", "password123")
    rt = tokens["refresh_token"]

    # Logout
    logout_res = client.post("/logout", json={"refresh_token": rt})
    assert logout_res.status_code == 200

    # Refresh with revoked token must fail
    refresh_res = client.post("/refresh", json={"refresh_token": rt})
    assert refresh_res.status_code == 401


def test_unauthenticated_request_rejected():
    client = TestClient(app)
    res = client.get("/sessions")
    assert res.status_code == 401


def test_student_cannot_read_another_students_report_or_events():
    client = TestClient(app)
    student1 = login(client, "student@example.com", "password123")
    s1_headers = {"Authorization": f"Bearer {student1['access_token']}"}
    session_id = client.post("/api/v1/sessions", headers=s1_headers).json()["session_id"]

    db = SessionLocal()
    db.add(User(name="StudentTwo", email="student2@example.com", role="student", password_hash=hash_password("password123"), is_active=1))
    db.commit()
    db.close()

    student2 = login(client, "student2@example.com", "password123")
    s2_headers = {"Authorization": f"Bearer {student2['access_token']}"}

    # Student 2 cannot access student 1 events
    assert client.get(f"/api/v1/events/{session_id}", headers=s2_headers).status_code == 403
    # Student 2 cannot access student 1 report
    assert client.get(f"/api/v1/reports/{session_id}", headers=s2_headers).status_code == 403


def test_admin_can_access_report_events_and_statistics():
    client = TestClient(app)
    student = login(client, "student@example.com", "password123")
    s_headers = {"Authorization": f"Bearer {student['access_token']}"}
    session_id = client.post("/api/v1/sessions", headers=s_headers).json()["session_id"]

    admin = login(client, "admin@example.com", "admin1234")
    a_headers = {"Authorization": f"Bearer {admin['access_token']}"}

    # Admin accesses report
    report_res = client.get(f"/api/v1/reports/{session_id}", headers=a_headers)
    assert report_res.status_code == 200
    assert report_res.json()["session_id"] == session_id

    # Admin accesses statistics
    stats_res = client.get("/statistics", headers=a_headers)
    assert stats_res.status_code == 200
    assert stats_res.json()["total_sessions"] >= 1


def test_session_idempotent_end_and_not_found():
    client = TestClient(app)
    student = login(client, "student@example.com", "password123")
    headers = {"Authorization": f"Bearer {student['access_token']}"}
    session_id = client.post("/api/v1/sessions", headers=headers).json()["session_id"]

    # End first time
    res1 = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
    assert res1.status_code == 200
    assert res1.json()["status"] == "COMPLETED"

    # End second time (idempotent)
    res2 = client.delete(f"/api/v1/sessions/{session_id}", headers=headers)
    assert res2.status_code == 200
    assert res2.json()["status"] == "COMPLETED"

    # Nonexistent session
    assert client.get("/api/v1/sessions/999999", headers=headers).status_code == 404
    assert client.delete("/api/v1/sessions/999999", headers=headers).status_code == 404
