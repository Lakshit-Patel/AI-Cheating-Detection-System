
from __future__ import annotations

import logging
from collections import deque
from threading import RLock

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .model_manager import model_manager
from .pose import (
    CalibrationConfig,
    Calibrator,
    DirectionClassifier,
    DirectionConfig,
    pose_from_landmarks,
)

logger = logging.getLogger("proctor.detector")


def _require_face_mesh():
    solutions = getattr(mp, "solutions", None)

    if solutions is None or not hasattr(solutions, "face_mesh"):
        raise RuntimeError(
            f"Installed mediapipe {getattr(mp, '__version__', '?')} "
            "does not expose solutions.face_mesh. Use "
            "mediapipe==0.10.14 with numpy<2."
        )

    return solutions.face_mesh


def _cfg(name, default):
    return getattr(config, name, default)


class Detector:
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)

    SMOOTHING = 3
    PHONE_EVERY_N_FRAMES = 3

    def __init__(self):
        self.face_mesh = _require_face_mesh().FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.frame_counter = 0
        self.last_phones = []

        self.pitch_history = deque(maxlen=self.SMOOTHING)
        self.yaw_history = deque(maxlen=self.SMOOTHING)
        self.roll_history = deque(maxlen=self.SMOOTHING)

        self.calibrator = Calibrator(
            CalibrationConfig(
                samples_needed=int(
                    _cfg("CALIBRATION_SAMPLES", 20)
                ),
                max_abs_pitch=50.0,
                max_abs_yaw=20.0,
                max_abs_roll=25.0,
                timeout_seconds=20.0,
            )
        )

        self.directions = DirectionClassifier(
            DirectionConfig(
                yaw_threshold=float(
                    _cfg("YAW_THRESHOLD_DEG", 20.0)
                ),
                up_threshold=float(
                    _cfg("LOOK_UP_THRESHOLD_DEG", 18.0)
                ),
                down_threshold=float(
                    _cfg("LOOK_DOWN_THRESHOLD_DEG", 15.0)
                ),
                hysteresis_ratio=0.7,
                enter_frames=int(
                    _cfg("DIRECTION_ENTER_FRAMES", 3)
                ),
                clear_frames=int(
                    _cfg("DIRECTION_CLEAR_FRAMES", 3)
                ),
            )
        )

        self.calibration_started_at = None
        self._closed = False
        self._lock = RLock()

    @property
    def calibrated(self):
        return self.calibrator.done

    @property
    def baseline_pitch(self):
        return self.calibrator.baseline_pitch

    @property
    def baseline_yaw(self):
        return self.calibrator.baseline_yaw

    def reset_calibration(self):
        with self._lock:
            self.calibrator.reset()
            self.directions.reset()

            self.pitch_history.clear()
            self.yaw_history.clear()
            self.roll_history.clear()

            self.calibration_started_at = None

    def estimate_head_pose(self, landmarks, frame_shape):
        height, width = frame_shape[:2]

        return pose_from_landmarks(
            landmarks,
            width,
            height,
        )

    @staticmethod
    def _draw_text(
        frame,
        text,
        x,
        y,
        color,
        scale=0.65,
    ):
        cv2.putText(
            frame,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

    def detect_phone(self, frame):
        try:
            yolo = model_manager.get_yolo()

        except Exception:
            logger.exception("Could not load YOLO model")
            return []

        if yolo is None:
            return []

        try:
            with model_manager.inference_lock:
                results = yolo(
                    frame,
                    imgsz=320,
                    verbose=False,
                )

        except Exception:
            logger.exception("YOLO phone detection failed")
            return []

        phones = []
        names = getattr(yolo, "names", {}) or {}

        threshold = float(
            _cfg("PHONE_CONF_THRESHOLD", 0.45)
        )

        for result in results or []:
            boxes = getattr(result, "boxes", None)

            if boxes is None:
                continue

            for box in boxes:
                try:
                    class_id = int(box.cls[0])

                    if isinstance(names, dict):
                        class_name = names.get(class_id)
                    else:
                        class_name = names[class_id]

                    confidence = float(box.conf[0])

                    if class_name != "cell phone":
                        continue

                    if confidence < threshold:
                        continue

                    coords = (
                        np.asarray(box.xyxy[0])
                        .reshape(-1)[:4]
                    )

                    x1, y1, x2, y2 = map(int, coords)

                    phones.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "confidence": confidence,
                        }
                    )

                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    logger.debug(
                        "Skipping malformed YOLO detection"
                    )

        return phones

    def process_frame(self, frame):
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            raise ValueError(
                "Expected a non-empty OpenCV BGR frame"
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("Detector is closed")

            # Mirror the camera frame.
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Run YOLO every third frame.
            self.frame_counter += 1

            if (
                self.frame_counter % self.PHONE_EVERY_N_FRAMES
                == 0
            ):
                self.last_phones = self.detect_phone(frame)

            phones = list(self.last_phones)
            phone_detected = bool(phones)

            for phone in phones:
                x1, y1, x2, y2 = phone["bbox"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    self.RED,
                    2,
                )

                self._draw_text(
                    frame,
                    "PHONE",
                    x1,
                    max(20, y1 - 8),
                    self.RED,
                )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = self.face_mesh.process(rgb)

            status = "STATUS : SECURE"
            color = self.GREEN

            pitch = yaw = roll = 0.0
            relative_pitch = relative_yaw = 0.0

            face_count = 0
            no_face = False
            multiple_faces = False
            pose_valid = False
            usable_pose = False

            directions = dict(self.directions.state)

            if not self.calibrated:
                self.calibrator.tick()

            faces = (
                getattr(results, "multi_face_landmarks", None)
                or []
            )

            if not faces:
                no_face = True
                status = "FLAGGED : NO FACE DETECTED"
                color = self.RED

                self.calibrator.note_invalid(
                    "No face detected - move into view"
                )

            else:
                face_count = len(faces)
                multiple_faces = face_count > 1

                if multiple_faces:
                    status = "FLAGGED : MULTIPLE FACES"
                    color = self.RED

                    self.calibrator.note_invalid(
                        "Only one person may be in view"
                    )

                landmarks = faces[0].landmark

                # Draw face bounding rectangle.
                points = np.asarray(
                    [
                        (lm.x * width, lm.y * height)
                        for lm in landmarks
                    ],
                    dtype=np.float32,
                )

                if points.size:
                    x1, y1 = (
                        np.min(points, axis=0).astype(int)
                    )

                    x2, y2 = (
                        np.max(points, axis=0).astype(int)
                    )

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width - 1, x2)
                    y2 = min(height - 1, y2)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        self.GREEN,
                        2,
                    )

                # Estimate head pose on EVERY frame.
                pose = self.estimate_head_pose(
                    landmarks,
                    frame.shape,
                )

                if (
                    pose is None
                    or len(pose) != 3
                    or not np.all(np.isfinite(pose))
                ):
                    self.calibrator.note_invalid(
                        "Cannot estimate head pose - improve lighting"
                    )

                else:
                    pose_valid = True

                    raw_pitch, raw_yaw, raw_roll = map(
                        float,
                        pose,
                    )

                    self.pitch_history.append(raw_pitch)
                    self.yaw_history.append(raw_yaw)
                    self.roll_history.append(raw_roll)

                    pitch = float(
                        np.mean(self.pitch_history)
                    )

                    yaw = float(
                        np.mean(self.yaw_history)
                    )

                    roll = float(
                        np.mean(self.roll_history)
                    )

                    # Calibrate only with exactly one detected face.
                    if (
                        not self.calibrated
                        and not multiple_faces
                    ):
                        if self.calibration_started_at is None:
                            import time

                            self.calibration_started_at = (
                                time.monotonic()
                            )

                        self.calibrator.add(
                            pitch,
                            yaw,
                            roll,
                        )

                    if not self.calibrated:
                        if not multiple_faces:
                            status = "CALIBRATING..."
                            color = self.YELLOW

                    else:
                        relative_pitch = (
                            pitch
                            - self.calibrator.baseline_pitch
                        )

                        relative_yaw = (
                            yaw
                            - self.calibrator.baseline_yaw
                        )

                        if not multiple_faces:
                            usable_pose = True

                            directions = (
                                self.directions.update(
                                    relative_pitch,
                                    relative_yaw,
                                )
                            )

                        if (
                            any(directions.values())
                            and not multiple_faces
                        ):
                            label = (
                                "LEFT"
                                if directions["left"]
                                else "RIGHT"
                                if directions["right"]
                                else "UP"
                                if directions["up"]
                                else "DOWN"
                            )

                            status = (
                                f"FLAGGED : LOOKING {label}"
                            )

                            color = self.RED

                # Draw the six landmarks used by pose.py.
                for index in (
                    33,
                    263,
                    1,
                    61,
                    291,
                    199,
                ):
                    if index < len(landmarks):
                        lm = landmarks[index]

                        px = int(lm.x * width)
                        py = int(lm.y * height)

                        cv2.circle(
                            frame,
                            (px, py),
                            3,
                            self.BLUE,
                            -1,
                        )

            # Invalid frames cannot create new violations.
            if not usable_pose:
                directions = self.directions.hold()

            looking_left = bool(directions["left"])
            looking_right = bool(directions["right"])
            looking_up = bool(directions["up"])
            looking_down = bool(directions["down"])

            no_valid_pose = (
                face_count >= 1 and not pose_valid
            )

            if no_valid_pose and not multiple_faces:
                status = "NO VALID POSE (diagnostic)"
                color = self.YELLOW

            # Phone detection takes priority in displayed status.
            if phone_detected:
                status = "FLAGGED : MOBILE PHONE DETECTED"
                color = self.RED

            self._draw_text(
                frame,
                status,
                20,
                32,
                color,
            )

            self._draw_text(
                frame,
                f"Pitch: {pitch:.1f}",
                20,
                65,
                self.GREEN,
            )

            self._draw_text(
                frame,
                f"Yaw: {yaw:.1f}",
                20,
                95,
                self.GREEN,
            )

            self._draw_text(
                frame,
                f"Roll: {roll:.1f}",
                20,
                125,
                self.GREEN,
            )

            self._draw_text(
                frame,
                (
                    f"Relative P/Y: "
                    f"{relative_pitch:+.1f} / "
                    f"{relative_yaw:+.1f}"
                ),
                20,
                155,
                (
                    self.RED
                    if any(
                        (
                            looking_left,
                            looking_right,
                            looking_up,
                            looking_down,
                        )
                    )
                    else self.BLUE
                ),
                scale=0.55,
            )

            phone_confidence = max(
                (
                    p["confidence"]
                    for p in phones
                ),
                default=None,
            )

            data = {
                "status": status,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "relative_pitch": relative_pitch,
                "relative_yaw": relative_yaw,
                "face_count": face_count,
                "looking_left": looking_left,
                "looking_right": looking_right,
                "looking_down": looking_down,
                "looking_up": looking_up,
                "no_face": no_face,
                "no_valid_pose": no_valid_pose,
                "multiple_faces": multiple_faces,
                "phone_detected": phone_detected,
                "phone_count": len(phones),
                "phone_confidence": phone_confidence,
                "pose_valid": pose_valid,
                "calibrated": self.calibrated,
                "calibration_progress": (
                    self.calibrator.progress
                ),
                "calibration_message": (
                    self.calibrator.message
                ),
                "calibration_timed_out": (
                    self.calibrator.timed_out
                ),
                "calibration_attempts": (
                    self.calibrator.attempts
                ),
                "baseline_pitch": (
                    self.calibrator.baseline_pitch
                    if self.calibrated
                    else None
                ),
                "baseline_yaw": (
                    self.calibrator.baseline_yaw
                    if self.calibrated
                    else None
                ),
            }

            return frame, data

    def close(self):
        with self._lock:
            if self._closed:
                return

            self._closed = True

            try:
                self.face_mesh.close()

            except Exception:
                logger.exception(
                    "Error closing MediaPipe FaceMesh"
                )
