"""Pose-math tests: synthetic landmarks with KNOWN head angles.

These prove sign conventions and the absence of +/-180 wraparound. They do NOT
prove accuracy on a real webcam; see README manual checklist for that.
"""
import numpy as np
import pytest

from app.pose import (
    Calibrator, CalibrationConfig, DirectionClassifier, DirectionConfig,
    FACE_IDS, pose_from_landmarks,
)

# Plausible face in a head frame: x = image-right in the MIRRORED frame, y = down, z = forward (toward camera).
# FaceMesh puts landmark 33 on the image-LEFT of 263 (verified on real captured landmarks), so 33/61 are at -x.
# Deliberately NOT identical to app.pose.MODEL_POINTS (wider eyes, deeper nose, longer face),
# so the tests exercise robustness to individual face-shape variation instead of being circular.
HEAD = {33: (-52, -38, -38), 263: (52, -38, -38), 1: (0, 0, 0),
        61: (-32, 45, -33), 291: (32, 45, -33), 199: (0, 88, -45)}


class LM:
    def __init__(self, x, y):
        self.x, self.y = x, y


def synth(yaw=0.0, pitch=0.0, roll=0.0, W=480, H=360, dist=600.0, noise=0.0, seed=0):
    """Mirror-view landmarks (frame flipped before FaceMesh). yaw>0 = turn to subject's right."""
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])  # +p: nose (+z) rises toward -y
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    R = Ry @ Rx @ Rz
    rng = np.random.default_rng(seed)
    lms = [LM(0.0, 0.0) for _ in range(300)]
    for idx, pt in HEAD.items():
        q = R @ np.array(pt, float)
        zc = dist - q[2]
        u = W * q[0] / zc + W / 2 + rng.normal(0, noise)
        v = W * q[1] / zc + H / 2 + rng.normal(0, noise)
        lms[idx] = LM(u / W, v / H)
    return lms


def pose(**kw):
    return pose_from_landmarks(synth(**kw), 480, 360)


def test_neutral_is_near_zero_no_wraparound():
    pitch, yaw, roll = pose()
    assert abs(pitch) < 3 and abs(yaw) < 3 and abs(roll) < 3


@pytest.mark.parametrize("deg", [10, 20, 30])
def test_yaw_sign_and_gain(deg):
    _, yaw_r, _ = pose(yaw=deg)
    _, yaw_l, _ = pose(yaw=-deg)
    assert yaw_r > 0 > yaw_l                       # right = +, left = -
    assert abs(yaw_r - deg) < 0.35 * deg           # gain ~1, not ~0.5 as before


@pytest.mark.parametrize("deg", [10, 20])
def test_pitch_sign_and_gain(deg):
    # A generic model leaves a small per-face pitch offset; calibration removes it,
    # so judge the response RELATIVE to the neutral pose, exactly as the detector does.
    base, _, _ = pose()
    up = pose(pitch=deg)[0] - base
    dn = pose(pitch=-deg)[0] - base
    assert up > 0 > dn                             # up = +, down = -
    assert abs(up - deg) < 0.4 * deg and abs(dn + deg) < 0.4 * deg


def test_yaw_and_pitch_are_decoupled():
    pitch, yaw, _ = pose(yaw=25)
    assert abs(pitch) < 8
    pitch, yaw, _ = pose(pitch=20)
    assert abs(yaw) < 8


def test_roll_sign_is_bounded():
    _, _, roll = pose(roll=15)
    assert abs(abs(roll) - 15) < 6


def test_degenerate_points_return_none():
    lms = [LM(0.5, 0.5) for _ in range(300)]
    pose_from_landmarks(lms, 480, 360)             # must not raise (all points identical)
    bad = [LM(float("nan"), 0.5) for _ in range(300)]
    assert pose_from_landmarks(bad, 480, 360) is None


def test_direction_classifier_labels_are_separate():
    c = DirectionClassifier(DirectionConfig(enter_frames=3, clear_frames=2))
    for _ in range(2):
        assert not c.update(0, -30)["left"]       # not yet: needs 3 frames
    assert c.update(0, -30)["left"]
    s = c.state
    assert not s["right"] and not s["up"] and not s["down"]
    # hysteresis: 15 deg is below the 20 deg threshold but above 0.7*20=14 -> stays raised
    for _ in range(5):
        c.update(0, -15)
    assert c.state["left"]
    for _ in range(2):
        c.update(0, 0)
    assert not c.state["left"]


def test_up_and_down_independent_of_yaw():
    c = DirectionClassifier(DirectionConfig(enter_frames=1))
    st = c.update(25, 0)
    assert st["up"] and not st["down"] and not st["left"] and not st["right"]
    c.reset()
    st = c.update(-25, 0)
    assert st["down"] and not st["up"]
    c.reset()
    st = c.update(25, 30)
    assert st["up"] and st["right"]               # both axes may fire together


def test_hold_never_raises_violation():
    c = DirectionClassifier(DirectionConfig(enter_frames=2))
    c.update(0, 30)
    c.hold()
    assert not c.update(0, 30)["right"]           # hold() reset the enter counter


def test_calibration_completes_and_rejects_off_center():
    cal = Calibrator(CalibrationConfig(samples_needed=5))
    for _ in range(10):
        cal.add(0, -40, 0)                         # way off-center (yaw)
    assert not cal.done and cal.rejected == 10
    for _ in range(5):
        cal.add(3.0, -2.0, 0)
    assert cal.done and cal.progress == 1.0
    assert cal.baseline_pitch == pytest.approx(3.0)
    assert cal.baseline_yaw == pytest.approx(-2.0)
    cal.reset()
    assert not cal.done and cal.progress == 0.0


def test_noisy_neutral_calibrates():
    cal = Calibrator(CalibrationConfig(samples_needed=20))
    for i in range(40):
        r = pose_from_landmarks(synth(noise=0.6, seed=i), 480, 360)
        assert r is not None
        cal.add(*r)
    assert cal.done


# ---- calibration regression tests (real-landmark failure: yaw~180/roll~-170 rejected everything) ----

# Landmarks captured from a real webcam (480x360 mirrored frame), order = FACE_IDS.
REAL_FRAME = [(141.55, 133.18), (228.66, 131.77), (163.53, 155.32),
              (144.63, 176.06), (201.51, 179.99), (163.53, 211.59)]


def test_real_captured_landmarks_give_a_non_flipped_pose():
    from app.pose import pose_from_image_points
    pitch, yaw, roll = pose_from_image_points(np.array(REAL_FRAME), 480, 360)
    assert abs(yaw) < 25 and abs(roll) < 30      # was yaw ~ -172, roll ~ -173 before the fix


def test_progress_follows_accepted_samples_not_time(monkeypatch):
    import app.pose as P
    t = [1000.0]
    monkeypatch.setattr(P.time, "monotonic", lambda: t[0])
    cal = Calibrator(CalibrationConfig(samples_needed=10))
    t[0] += 5.0                                   # time passing alone must not move the bar
    cal.tick()
    assert cal.progress == 0.0
    for i in range(5):
        cal.add(2.0, 1.0, 0.0)
    assert cal.progress == pytest.approx(0.5) and not cal.done


def test_rejected_samples_do_not_advance_progress():
    cal = Calibrator(CalibrationConfig(samples_needed=5))
    for _ in range(20):
        cal.add(0, 60, 0)                         # yaw far outside the limit
    assert cal.progress == 0.0 and not cal.done and cal.rejected == 20


def test_non_finite_and_insane_pose_rejected():
    cal = Calibrator(CalibrationConfig(samples_needed=3))
    cal.add(float("nan"), 0, 0)
    cal.add(0, float("inf"), 0)
    cal.add(170, 0, 0)                            # solver garbage (pitch beyond sanity bound)
    assert cal.rejected == 3 and cal.progress == 0.0


def test_large_constant_pitch_offset_still_calibrates():
    cal = Calibrator(CalibrationConfig(samples_needed=10))
    for i in range(10):
        cal.add(33.0 + 0.5 * (i % 2), -8.0, 1.0)  # generic-model offset like the captured face
    assert cal.done and cal.baseline_pitch == pytest.approx(33.25, abs=0.3)


def test_moving_head_restarts_calibration():
    cal = Calibrator(CalibrationConfig(samples_needed=10))
    for _ in range(6):
        cal.add(0, 0, 0)
    assert cal.progress == pytest.approx(0.6)
    cal.add(0, 20, 0)                             # head turned during calibration
    assert not cal.done and cal.progress == pytest.approx(0.1)
    assert cal.rejected == 1


def test_timeout_is_explicit_and_never_marks_calibrated(monkeypatch):
    import app.pose as P
    t = [0.0]
    monkeypatch.setattr(P.time, "monotonic", lambda: t[0])
    cal = Calibrator(CalibrationConfig(samples_needed=10, timeout_seconds=20.0))
    cal.tick()
    for _ in range(4):
        cal.add(0, 0, 0)
    t[0] = 21.0
    cal.tick()
    assert not cal.done and cal.timed_out and cal.attempts == 1
    assert cal.progress == 0.0 and "timed out" in cal.message.lower()
    for _ in range(10):                           # a fresh attempt can still succeed
        cal.add(0, 0, 0)
    assert cal.done and not cal.timed_out
