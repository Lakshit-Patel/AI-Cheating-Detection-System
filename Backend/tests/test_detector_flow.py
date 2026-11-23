"""End-to-end Detector.process_frame flow with a FAKE FaceMesh fed synthetic landmarks.

Proves: calibration completes, then left/right/up/down each fire (and only their own
flag), and recover. MediaPipe/YOLO themselves are NOT exercised here.
"""
import numpy as np

from app import detector as det_mod
from tests.test_pose import synth


class _Result:
    def __init__(self, faces):
        self.multi_face_landmarks = faces


class _Face:
    def __init__(self, lms):
        self.landmark = lms


class FakeMesh:
    def __init__(self, **kw):
        self.faces = 1
        self.angles = dict(yaw=0.0, pitch=0.0)

    def process(self, rgb):
        if self.faces == 0:
            return _Result(None)
        lms = synth(**self.angles)
        return _Result([_Face(lms)] * self.faces)

    def close(self):
        pass


class FakeSolutions:
    class face_mesh:
        FaceMesh = FakeMesh


def make_detector(monkeypatch=None):
    det_mod._require_face_mesh = lambda: FakeSolutions.face_mesh
    d = det_mod.Detector()
    d.last_phones = []
    d.detect_phone = lambda frame: []          # no YOLO in this test
    return d


FRAME = np.zeros((360, 480, 3), dtype=np.uint8)


def run(d, n, **angles):
    d.face_mesh.angles = angles
    out = None
    for _ in range(n):
        _, out = d.process_frame(FRAME)
    return out


def calibrate(d):
    out = run(d, 30)
    assert out["calibrated"], out
    return out


def test_calibration_completes_without_getting_stuck():
    d = make_detector()
    out = run(d, 5)
    assert not out["calibrated"] and 0 < out["calibration_progress"] < 1
    out = run(d, 25)
    assert out["calibrated"] and out["calibration_progress"] == 1.0
    assert out["status"] == "STATUS : SECURE"


def test_each_direction_fires_alone_then_recovers():
    d = make_detector()
    calibrate(d)
    cases = {
        "looking_right": dict(yaw=35), "looking_left": dict(yaw=-35),
        "looking_up": dict(pitch=30), "looking_down": dict(pitch=-30),
    }
    for flag, ang in cases.items():
        out = run(d, 12, **ang)
        assert out[flag], (flag, out["relative_yaw"], out["relative_pitch"])
        assert sum(out[k] for k in cases) == 1, f"{flag}: other flags fired too"
        out = run(d, 12)                       # back to neutral
        assert not any(out[k] for k in cases), f"{flag} did not clear"


def test_small_movement_is_not_a_violation():
    d = make_detector()
    calibrate(d)
    out = run(d, 20, yaw=8, pitch=5)
    assert not any(out[k] for k in ("looking_left", "looking_right", "looking_up", "looking_down"))


def test_no_face_never_calibrates_and_flags_only_no_face():
    d = make_detector()
    d.face_mesh.faces = 0
    out = run(d, 30)
    assert out["no_face"] and not out["calibrated"]
    assert not any(out[k] for k in ("looking_left", "looking_right", "looking_up", "looking_down"))


def test_multiple_faces_do_not_calibrate():
    d = make_detector()
    d.face_mesh.faces = 2
    out = run(d, 30)
    assert out["multiple_faces"] and not out["calibrated"]


def test_reset_calibration_for_new_session():
    d = make_detector()
    calibrate(d)
    d.reset_calibration()
    out = run(d, 3)
    assert not out["calibrated"]


def test_turned_head_never_calibrates():
    d = make_detector()
    out = run(d, 40, yaw=35)
    assert not out["calibrated"] and out["calibration_progress"] == 0.0


def test_looking_away_then_straight_completes():
    d = make_detector()
    run(d, 10, yaw=35)
    out = run(d, 30)
    assert out["calibrated"] and out["calibration_progress"] == 1.0


def test_timeout_is_reported_not_stuck(monkeypatch):
    import app.pose as P
    t = [100.0]
    monkeypatch.setattr(P.time, "monotonic", lambda: t[0])
    d = make_detector()
    run(d, 5, yaw=35)
    t[0] += 25.0                                  # past the 20 s calibration timeout
    out = run(d, 1, yaw=35)
    assert not out["calibrated"]
    assert out["calibration_timed_out"] and out["calibration_attempts"] == 1
