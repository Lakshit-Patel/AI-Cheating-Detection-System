import numpy as np
import pytest
from app.detector import Detector

def test_detector_blank_frame_reports_no_face():
    detector = Detector()
    try:
        # Create a 480x640 black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed, data = detector.process_frame(frame)
        assert data['no_face'] is True
        assert data['multiple_faces'] is False
        assert data['pose_valid'] is False
        assert data['status'] == 'FLAGGED : NO FACE DETECTED'
        assert processed.shape == (480, 640, 3)
    finally:
        detector.close()

# Calibration behaviour is covered without MediaPipe in tests/test_pose.py and
# tests/test_detector_flow.py (the old _calibrate()/MIN_CALIBRATION_SAMPLES API was replaced
# by app.pose.Calibrator).
