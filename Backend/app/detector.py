import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from ultralytics import YOLO


class Detector:

    # Class-level constants (shared, not rebuilt per instance)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)

    FACE_IDS = (33, 263, 1, 61, 291, 199)

    MODEL_POINTS = np.array([
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (0.0, 0.0, 0.0),
        (-25.0, 30.0, -20.0),
        (25.0, 30.0, -20.0),
        (0.0, 65.0, -5.0)
    ], dtype=np.float64)

    DIST_COEFFS = np.zeros((4, 1))

    SMOOTHING = 5
    YAW_THRESHOLD = 18
    LOOK_DOWN_THRESHOLD =42 # Increase this if it flags too easily
    LOOK_UP_THRESHOLD = 25
    CALIBRATION_SECONDS = 3.0

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.yolo = YOLO("yolov8n.pt")
        
        self.frame_counter = 0
        self.last_phones = []

        # Fixed-size ring buffers: O(1) append/evict instead of list + pop(0)
        self.pitch_history = deque(maxlen=self.SMOOTHING)
        self.yaw_history = deque(maxlen=self.SMOOTHING)
        self.roll_history = deque(maxlen=self.SMOOTHING)

        # Camera matrix depends only on frame size — cache it instead of
        # rebuilding a fresh matrix every single frame
        self._cached_dims = None
        self._cached_camera_matrix = None

        # -------------------------
        # CALIBRATION
        # -------------------------
        # Different webcams/desk heights/monitor angles all produce a
        # different "neutral" pitch and yaw for the same person looking
        # straight at the screen. A fixed threshold can't account for
        # that, so instead we sample the first few seconds of a session,
        # average them into a baseline, and compare everything after
        # that relative to the baseline instead of to zero.
        self.calibrated = False
        self.calibration_start = None
        self.calibration_pitch_samples = []
        self.calibration_yaw_samples = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0

    # -------------------------
    # CAMERA MATRIX
    # -------------------------

    def get_camera_matrix(self, width, height):
        if self._cached_dims != (width, height):
            focal_length = width
            center = (width / 2, height / 2)

            self._cached_camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            self._cached_dims = (width, height)

        return self._cached_camera_matrix, self.DIST_COEFFS

    # -------------------------
    # HEAD POSE
    # -------------------------

    def estimate_head_pose(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        image_points = np.array(
            [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.FACE_IDS],
            dtype=np.float64
        )

        camera_matrix, dist = self.get_camera_matrix(w, h)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, translation_vector))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

        pitch = float(euler_angles[0])
        yaw = float(euler_angles[1])
        roll = float(euler_angles[2])

        return pitch, yaw, roll

    # -------------------------
    # FACE BOX
    # -------------------------

    def draw_face_box(self, frame, landmarks):
        h, w = frame.shape[:2]

        pts = np.array([[lm.x, lm.y] for lm in landmarks])
        x1, y1 = (pts.min(axis=0) * [w, h]).astype(int)
        x2, y2 = (pts.max(axis=0) * [w, h]).astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.GREEN, 2)

    # -------------------------
    # DRAW TEXT
    # -------------------------

    def draw_text(self, frame, text, x, y, color):
        cv2.putText(
            frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

    # -------------------------
    # PROCESS FRAME
    # -------------------------
    
    def detect_phone(self, frame):
        results = self.yolo(frame,
                            imgsz=320,
                            verbose=False
                        )
        phones = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                # COCO class 67 = cell phone
                if cls == 67:
                    conf = float(box.conf[0])
                    if conf > 0.45:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        phones.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf
                        })
                        
        return phones

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            self.last_phones = self.detect_phone(frame)
        phones = self.last_phones
        phone_detected = len(phones) > 0
        for phone in phones:
            x1, y1, x2, y2 = phone["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.RED, 2)
            self.draw_text(
                frame,
                "PHONE",
                x1,
                max(20, y1 - 10),
                self.RED
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        status = "STATUS : SECURE"
        color = self.GREEN

        pitch = yaw = roll = 0.0
        face_count = 0

        looking_left = looking_right = looking_down = looking_up = False
        no_face = multiple_faces = False

        if results.multi_face_landmarks is None:
            no_face = True
            status = "FLAGGED : NO FACE DETECTED"
            color = self.RED

        else:
            face_count = len(results.multi_face_landmarks)

            if face_count > 1:
                multiple_faces = True
                status = "FLAGGED : MULTIPLE FACES"
                color = self.RED

            landmarks = results.multi_face_landmarks[0].landmark
            self.draw_face_box(frame, landmarks)

            pose = self.estimate_head_pose(landmarks, frame.shape)

            if pose is not None:
                pitch, yaw, roll = pose

                # -------------------------
                # SMOOTHING (deque handles maxlen eviction automatically)
                # -------------------------
                self.pitch_history.append(pitch)
                self.yaw_history.append(yaw)
                self.roll_history.append(roll)

                pitch = sum(self.pitch_history) / len(self.pitch_history)
                yaw = sum(self.yaw_history) / len(self.yaw_history)
                roll = sum(self.roll_history) / len(self.roll_history)

                # -------------------------
                # CALIBRATION
                # -------------------------
                if not self.calibrated:
                    if self.calibration_start is None:
                        self.calibration_start = time.time()

                    self.calibration_pitch_samples.append(pitch)
                    self.calibration_yaw_samples.append(yaw)

                    elapsed = time.time() - self.calibration_start

                    if elapsed >= self.CALIBRATION_SECONDS:
                        self.baseline_pitch = (
                            sum(self.calibration_pitch_samples)
                            / len(self.calibration_pitch_samples)
                        )
                        self.baseline_yaw = (
                            sum(self.calibration_yaw_samples)
                            / len(self.calibration_yaw_samples)
                        )
                        self.calibrated = True

                    status = "CALIBRATING..."
                    color = self.YELLOW

                else:
                    # Compare against this session's own neutral pose,
                    # not against zero — this is what makes the
                    # thresholds work across different webcams/desks.
                    relative_pitch = pitch - self.baseline_pitch
                    relative_yaw = yaw - self.baseline_yaw

                    # -------------------------
                    # HEAD DIRECTION
                    # -------------------------
                    if relative_yaw < -self.YAW_THRESHOLD:
                        looking_left = True
                        status = "FLAGGED : LOOKING LEFT"
                        color = self.RED

                    elif relative_yaw > self.YAW_THRESHOLD:
                        looking_right = True
                        status = "FLAGGED : LOOKING RIGHT"
                        color = self.RED

                    elif relative_pitch < -self.LOOK_DOWN_THRESHOLD:
                        looking_down = True
                        status = "FLAGGED : LOOKING DOWN"
                        color = self.RED
                        # Looking UP (optional)
                    elif relative_pitch > self.LOOK_UP_THRESHOLD:
                        looking_up = True
                        status = "FLAGGED : LOOKING UP"
                        color = self.RED

                    else:
                        status = "STATUS : SECURE"
                        color = self.GREEN

                # -------------------------
                # DRAW LANDMARKS
                # -------------------------
                h, w = frame.shape[:2]

                for idx in self.FACE_IDS:
                    lm = landmarks[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, self.BLUE, -1)
        if phone_detected:
            status = "FLAGGED : MOBILE PHONE DETECTED"
            color = self.RED

        # -------------------------
        # DRAW STATUS
        # -------------------------
        self.draw_text(frame, status, 20, 35, color)
        self.draw_text(frame, f"Pitch : {pitch:.1f}", 20, 70, self.GREEN)
        self.draw_text(frame, f"Yaw : {yaw:.1f}", 20, 105, self.GREEN)
        self.draw_text(frame, f"Roll : {roll:.1f}", 20, 140, self.GREEN)

        data = {
            "status": status,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "face_count": face_count,
            "looking_left": looking_left,
            "looking_right": looking_right,
            "looking_down": looking_down,
            "looking_up": looking_up,
            "no_face": no_face,
            "multiple_faces": multiple_faces,
            "phone_detected": phone_detected,
            "phone_count": len(phones),
            "calibrated": self.calibrated
        }

        return frame, data

    # -------------------------
    # CLEANUP
    # -------------------------

    def close(self):
        self.face_mesh.close()