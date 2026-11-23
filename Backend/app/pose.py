
"""Head-pose math and direction classification using NumPy and OpenCV."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# MediaPipe FaceMesh landmark indices.
FACE_IDS = (33, 263, 1, 61, 291, 199)

# Generic 3D face model points, in the camera frame at zero rotation
# (x right, y down, z away from camera; nose tip closest to the camera).
#
# HANDEDNESS MATTERS: the detector mirrors the frame before FaceMesh, and
# FaceMesh then reports landmark 33 on the image-LEFT of 263 (verified from real
# captured landmarks: 33 at x=141, 263 at x=228). So 33/61 must sit at NEGATIVE x.
# With the opposite signs solvePnP converges to a ~180 degree flipped pose and
# every neutral face looks like yaw ~180 / roll ~-170.
MODEL_POINTS = np.array(
    [
        (-45.0, -35.0, 30.0),   # 33  eye outer corner (image-left)
        (45.0, -35.0, 30.0),    # 263 eye outer corner (image-right)
        (0.0, 0.0, 0.0),        # 1   nose tip
        (-28.0, 40.0, 25.0),    # 61  mouth corner (image-left)
        (28.0, 40.0, 25.0),     # 291 mouth corner (image-right)
        (0.0, 75.0, 35.0),      # 199 chin
    ],
    dtype=np.float64,
)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Build the camera intrinsic matrix."""
    focal_length = float(width)

    return np.array(
        [
            [focal_length, 0, width / 2.0],
            [0, focal_length, height / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def pose_from_image_points(
    image_points: np.ndarray,
    width: int,
    height: int,
):
    """Return (pitch, yaw, roll) in degrees, or None if estimation fails."""
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    if pts.shape[0] != len(MODEL_POINTS):
        return None

    if not np.all(np.isfinite(pts)):
        return None

    try:
        rvec0 = np.zeros((3, 1), dtype=np.float64)
        tvec0 = np.array(
            [[0.0], [0.0], [float(width) * 1.2]],
            dtype=np.float64,
        )

        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS,
            pts,
            camera_matrix(width, height),
            DIST_COEFFS,
            rvec0,
            tvec0,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not ok:
        return None

    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None

    # Reject solutions behind the camera, but do not reject a face
    # merely because its estimated forward vector has an unexpected sign.
    if float(tvec.ravel()[2]) <= 0:
        return None

    rmat, _ = cv2.Rodrigues(rvec)

    forward = rmat @ np.array([0.0, 0.0, -1.0])
    right = rmat @ np.array([1.0, 0.0, 0.0])

    yaw = float(
        np.degrees(np.arctan2(forward[0], -forward[2]))
    )

    pitch = float(
        np.degrees(
            np.arcsin(np.clip(-forward[1], -1.0, 1.0))
        )
    )

    roll = float(
        np.degrees(np.arctan2(right[1], right[0]))
    )

    return pitch, yaw, roll


def pose_from_landmarks(landmarks, width: int, height: int):
    """Estimate head pose from FaceMesh landmarks."""
    try:
        pts = [
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in FACE_IDS
        ]
    except (IndexError, AttributeError, TypeError):
        return None

    return pose_from_image_points(
        np.asarray(pts, dtype=np.float64),
        width,
        height,
    )


@dataclass
class DirectionConfig:
    yaw_threshold: float = 20.0
    up_threshold: float = 18.0
    down_threshold: float = 15.0
    hysteresis_ratio: float = 0.7
    enter_frames: int = 3
    clear_frames: int = 3


class DirectionClassifier:
    """Independent left/right/up/down state machines with hysteresis."""

    def __init__(self, cfg: DirectionConfig | None = None):
        self.cfg = cfg or DirectionConfig()
        self.reset()

    def reset(self):
        self.state = {
            "left": False,
            "right": False,
            "up": False,
            "down": False,
        }
        self._enter = {key: 0 for key in self.state}
        self._clear = {key: 0 for key in self.state}

    def _step(self, name: str, entering: bool, staying: bool):
        if not self.state[name]:
            self._enter[name] = (
                self._enter[name] + 1 if entering else 0
            )

            if self._enter[name] >= self.cfg.enter_frames:
                self.state[name] = True
                self._clear[name] = 0
        else:
            self._clear[name] = (
                0 if staying else self._clear[name] + 1
            )

            if self._clear[name] >= self.cfg.clear_frames:
                self.state[name] = False
                self._enter[name] = 0

    def update(self, rel_pitch: float, rel_yaw: float) -> dict:
        c = self.cfg
        exit_threshold = c.hysteresis_ratio

        self._step(
            "left",
            rel_yaw <= -c.yaw_threshold,
            rel_yaw <= -c.yaw_threshold * exit_threshold,
        )
        self._step(
            "right",
            rel_yaw >= c.yaw_threshold,
            rel_yaw >= c.yaw_threshold * exit_threshold,
        )
        self._step(
            "up",
            rel_pitch >= c.up_threshold,
            rel_pitch >= c.up_threshold * exit_threshold,
        )
        self._step(
            "down",
            rel_pitch <= -c.down_threshold,
            rel_pitch <= -c.down_threshold * exit_threshold,
        )

        return dict(self.state)

    def hold(self) -> dict:
        """Keep current state when pose is unavailable; do not raise new flags."""
        self._enter = {key: 0 for key in self._enter}
        return dict(self.state)


@dataclass
class CalibrationConfig:
    samples_needed: int = 20
    # Sanity bounds on the raw solver output. Pitch is deliberately wide: a
    # generic face model has a per-person constant pitch offset (about 30 degrees
    # on the captured test face), which the baseline subtraction removes. Neutrality
    # is enforced by STABILITY (max_spread_deg), not by absolute pitch.
    max_abs_pitch: float = 50.0
    max_abs_yaw: float = 25.0
    max_abs_roll: float = 30.0
    # A sample further than this from the median of the samples collected so far
    # means the head moved; the attempt restarts. Frame-to-frame solver jitter on
    # the captured face was ~1.8 deg (pitch) from 0.7px landmark noise.
    max_spread_deg: float = 10.0
    timeout_seconds: float = 20.0


@dataclass
class Calibrator:
    cfg: CalibrationConfig = field(
        default_factory=CalibrationConfig
    )

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.pitch = []
        self.yaw = []
        self.done = False
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        self.message = "Look straight at the camera"
        self.rejected = 0
        self.started_at = None
        self.timed_out = False
        self.attempts = 0

    @property
    def progress(self):
        """Fraction of the required stable samples collected (never time-based)."""
        if self.done:
            return 1.0

        return min(
            1.0,
            len(self.pitch) / max(1, self.cfg.samples_needed),
        )

    def tick(self):
        """Call once per frame while calibrating; enforces the timeout."""
        if self.done:
            return

        now = time.monotonic()

        if self.started_at is None:
            self.started_at = now
            return

        if now - self.started_at >= self.cfg.timeout_seconds:
            # Explicit failure of this attempt: never mark calibrated, start over.
            self.attempts += 1
            self.timed_out = True
            self.pitch = []
            self.yaw = []
            self.started_at = now
            self.message = (
                "Calibration timed out - face the camera, "
                "keep your head still and retry"
            )

    def add(self, pitch, yaw, roll):
        if self.done:
            return

        self.tick()
        c = self.cfg

        if not all(np.isfinite([pitch, yaw, roll])):
            self.rejected += 1
            self.message = "Cannot estimate head pose - improve lighting"
            return

        if (
            abs(pitch) > c.max_abs_pitch
            or abs(yaw) > c.max_abs_yaw
            or abs(roll) > c.max_abs_roll
        ):
            self.rejected += 1
            self.message = (
                "Face the camera directly and level your head"
            )
            return

        if len(self.pitch) >= 3:
            moved = (
                abs(pitch - float(np.median(self.pitch)))
                > c.max_spread_deg
                or abs(yaw - float(np.median(self.yaw)))
                > c.max_spread_deg
            )

            if moved:
                # Head moved during calibration: discard the window, keep this sample.
                self.rejected += 1
                self.pitch = [pitch]
                self.yaw = [yaw]
                self.message = "Hold still, restarting calibration"
                return

        self.pitch.append(pitch)
        self.yaw.append(yaw)

        if len(self.pitch) >= c.samples_needed:
            self._finish()
        else:
            self.message = "Hold still, calibrating"

    def _finish(self):
        self.baseline_pitch = float(np.median(self.pitch))
        self.baseline_yaw = float(np.median(self.yaw))
        self.done = True
        self.timed_out = False
        self.message = "Calibrated"

    def note_invalid(self, reason):
        if not self.done:
            self.message = reason
