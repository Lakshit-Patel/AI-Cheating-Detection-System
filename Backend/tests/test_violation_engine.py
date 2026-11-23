from datetime import UTC, datetime, timedelta

from app.violation_engine import ViolationEngine


def test_transient_flag_does_not_create_event():
    e = ViolationEngine(start_frames=2, clear_frames=2)
    now = datetime.now(UTC)
    started, ended = e.update({"looking_down": True}, now)
    assert started == []
    assert ended == []
    started, ended = e.update({"looking_down": False}, now + timedelta(seconds=1))
    assert started == []
    assert ended == []


def test_sustained_flag_creates_and_closes_event():
    e = ViolationEngine(start_frames=2, clear_frames=2)
    now = datetime.now(UTC)
    e.update({"looking_down": True}, now)
    started, _ = e.update({"looking_down": True}, now + timedelta(seconds=1))
    assert len(started) == 1
    assert started[0].violation_type == "LOOKING DOWN"

    _, ended = e.update({"looking_down": False}, now + timedelta(seconds=2))
    assert ended == []
    _, ended = e.update({"looking_down": False}, now + timedelta(seconds=3))
    assert len(ended) == 1
    assert ended[0].violation_type == "LOOKING DOWN"


def test_phone_confidence_is_preserved():
    e = ViolationEngine(start_frames=1, clear_frames=1)
    started, _ = e.update({"phone_detected": True, "phone_confidence": 0.91})
    assert started[0].confidence == 0.91


def test_multiple_simultaneous_violations():
    e = ViolationEngine(start_frames=2, clear_frames=2)
    now = datetime.now(UTC)
    e.update({"phone_detected": True, "phone_confidence": 0.88, "multiple_faces": True}, now)
    started, _ = e.update({"phone_detected": True, "phone_confidence": 0.88, "multiple_faces": True}, now + timedelta(seconds=1))
    assert len(started) == 2
    types = {s.violation_type for s in started}
    assert "MOBILE PHONE DETECTED" in types
    assert "MULTIPLE FACES" in types


def test_finalize_closes_all_active_violations():
    e = ViolationEngine(start_frames=1, clear_frames=2)
    start_time = datetime.now(UTC)
    e.update({"no_face": True}, start_time)
    assert len(e.active) == 1

    final_time = start_time + timedelta(seconds=5)
    ended = e.finalize(final_time)
    assert len(ended) == 1
    assert ended[0].violation_type == "NO FACE DETECTED"
    assert "duration=5.00s" in ended[0].details
    assert len(e.active) == 0
