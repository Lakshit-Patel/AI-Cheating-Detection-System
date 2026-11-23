"""Pose estimation compatible with the current ProctorAI detector.py API.

Restores the face landmark geometry and angle convention from the previously
working standalone detector. The public names/classes are retained for the
current detector.py imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np

# Same six MediaPipe FaceMesh points used by the uploaded working detector.
FACE_IDS = (33, 263, 1, 61, 291, 199)

# Same 3D face map used by the uploaded working detector. Do not silently
# replace these with a differently oriented face model: the Euler-angle
# thresholds below depend on this exact geometry/convention.
MODEL_POINTS = np.array([
    (-30.0, -30.0, -30.0),  # 33: outer eye corner
    (30.0, -30.0, -30.0),   # 263: outer eye corner
    (0.0, 0.0, 0.0),        # 1: nose tip
    (-25.0, 30.0, -20.0),   # 61: mouth corner
    (25.0, 30.0, -20.0),    # 291: mouth corner
    (0.0, 65.0, -5.0),      # 199: chin
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def camera_matrix(width: int, height: int) -> np.ndarray:
    focal_length = float(width)
    return np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def pose_from_image_points(image_points: np.ndarray, width: int, height: int):
    """Return OpenCV decomposed (pitch, yaw, roll), or None on failure.

    This intentionally uses the same solvePnP + decomposeProjectionMatrix
    calculation as the uploaded detector, so the old thresholds retain their
    original meaning. Input points must be in the mirrored frame.
    """
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if pts.shape != (len(FACE_IDS), 2) or not np.all(np.isfinite(pts)):
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, pts, camera_matrix(width, height), DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        projection = np.hstack((rmat, tvec))
        result = cv2.decomposeProjectionMatrix(projection)
        euler = np.asarray(result[6], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(euler)):
            return None
        return float(euler[0]), float(euler[1]), float(euler[2])
    except (cv2.error, ValueError, TypeError):
        return None


def pose_from_landmarks(landmarks, width: int, height: int):
    try:
        points = np.array(
            [[landmarks[index].x * width, landmarks[index].y * height]
             for index in FACE_IDS],
            dtype=np.float64,
        )
    except (IndexError, AttributeError, TypeError):
        return None
    return pose_from_image_points(points, width, height)


@dataclass
class DirectionConfig:
    # Restored thresholds from the uploaded detector.
    yaw_threshold: float = 18.0
    up_threshold: float = 25.0
    down_threshold: float = 42.0
    # Keep short persistence to avoid single-frame noise without making
    # violations feel delayed.
    hysteresis_ratio: float = 0.85
    enter_frames: int = 2
    clear_frames: int = 2


class DirectionClassifier:
    """Classifies the legacy Euler angles after neutral-pose calibration."""
    def __init__(self, cfg: DirectionConfig | None = None):
        self.cfg = cfg or DirectionConfig()
        self.reset()

    def reset(self):
        self.state = {"left": False, "right": False, "up": False, "down": False}
        self._enter = {key: 0 for key in self.state}
        self._clear = {key: 0 for key in self.state}

    def _step(self, name: str, entering: bool, staying: bool):
        if not self.state[name]:
            self._enter[name] = self._enter[name] + 1 if entering else 0
            if self._enter[name] >= self.cfg.enter_frames:
                self.state[name] = True
                self._clear[name] = 0
        else:
            self._clear[name] = 0 if staying else self._clear[name] + 1
            if self._clear[name] >= self.cfg.clear_frames:
                self.state[name] = False
                self._enter[name] = 0

    def update(self, rel_pitch: float, rel_yaw: float) -> dict:
        c = self.cfg
        h = c.hysteresis_ratio
        self._step("left", rel_yaw <= -c.yaw_threshold,
                   rel_yaw <= -c.yaw_threshold * h)
        self._step("right", rel_yaw >= c.yaw_threshold,
                   rel_yaw >= c.yaw_threshold * h)
        self._step("up", rel_pitch >= c.up_threshold,
                   rel_pitch >= c.up_threshold * h)
        self._step("down", rel_pitch <= -c.down_threshold,
                   rel_pitch <= -c.down_threshold * h)
        return dict(self.state)

    def hold(self) -> dict:
        # Invalid pose must not create a new directional violation.
        self._enter = {key: 0 for key in self._enter}
        return dict(self.state)


@dataclass
class CalibrationConfig:
    samples_needed: int = 20
    max_abs_pitch: float = 20.0
    max_abs_yaw: float = 20.0
    max_abs_roll: float = 25.0
    timeout_seconds: float = 20.0


@dataclass
class Calibrator:
    cfg: CalibrationConfig = field(default_factory=CalibrationConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.pitch: list[float] = []
        self.yaw: list[float] = []
        self.done = False
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.message = "Look straight at the camera"
        self.rejected = 0

    @property
    def progress(self) -> float:
        return 1.0 if self.done else min(
            1.0, len(self.pitch) / max(1, self.cfg.samples_needed)
        )

    def add(self, pitch: float, yaw: float, roll: float) -> None:
        if self.done:
            return
        c = self.cfg
        if (abs(pitch) <= c.max_abs_pitch and abs(yaw) <= c.max_abs_yaw
                and abs(roll) <= c.max_abs_roll):
            self.pitch.append(float(pitch))
            self.yaw.append(float(yaw))
            self.message = "Hold still, calibrating"
        else:
            self.rejected += 1
            self.message = "Face the camera directly and level your head"
        if len(self.pitch) >= c.samples_needed:
            self.baseline_pitch = float(np.median(self.pitch))
            self.baseline_yaw = float(np.median(self.yaw))
            self.done = True
            self.message = "Calibrated"

    def note_invalid(self, reason: str) -> None:
        if not self.done:
            self.message = reason
