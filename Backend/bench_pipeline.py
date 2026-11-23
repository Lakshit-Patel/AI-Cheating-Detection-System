"""Benchmark the real ProctorAI Detector pipeline on live webcam frames.

Run from the Backend directory (venv active, same OPENBLAS/OMP/MKL env vars as the server):
    python bench_pipeline.py            # 60 s run
    python bench_pipeline.py 30         # custom duration in seconds

Measures Detector.process_frame only (FaceMesh + head pose + YOLO every Nth frame +
drawing). It does NOT include webcam capture, JPEG encode/decode, or the WebSocket.
Frames are 480x360, the size the frontend sends. Sit in front of the camera while it runs.
"""
import platform
import sys
import time

import cv2
import numpy as np

from app.detector import Detector

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
WARMUP_FRAMES = 30

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Could not open webcam (is the browser tab / backend still using it?)")

det = Detector()
all_ms, yolo_ms, no_yolo_ms = [], [], []
calibrated_at = None

print("Warming up...")
n = 0
start = None
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.resize(frame, (480, 360))

    t0 = time.perf_counter()
    _, out = det.process_frame(frame)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    n += 1

    if n <= WARMUP_FRAMES:
        if n == WARMUP_FRAMES:
            print(f"Measuring for {DURATION:.0f} s...")
            start = time.perf_counter()
        continue

    all_ms.append(dt_ms)
    # YOLO runs every PHONE_EVERY_N_FRAMES-th frame; split the two populations.
    (yolo_ms if det.frame_counter % det.PHONE_EVERY_N_FRAMES == 0 else no_yolo_ms).append(dt_ms)
    if calibrated_at is None and out.get("calibrated"):
        calibrated_at = len(all_ms)

    if time.perf_counter() - start >= DURATION:
        break

elapsed = time.perf_counter() - start
cap.release()
if hasattr(det, "close"):
    det.close()


def stats(label, xs):
    if not xs:
        print(f"{label:<22} (no samples)")
        return
    a = np.array(xs)
    print(f"{label:<22} n={len(a):<6} mean={a.mean():6.1f}  p50={np.percentile(a, 50):6.1f}  "
          f"p95={np.percentile(a, 95):6.1f}  p99={np.percentile(a, 99):6.1f}  max={a.max():6.1f}  (ms)")


print("\n=== ProctorAI pipeline benchmark ===")
print(f"Machine: {platform.processor() or platform.machine()} | Python {platform.python_version()}")
print(f"Frame size: 480x360 | frames measured: {len(all_ms)} in {elapsed:.1f} s")
print(f"Throughput (back-to-back): {len(all_ms) / elapsed:.1f} FPS")
stats("All frames", all_ms)
stats(f"With YOLO (1 in {det.PHONE_EVERY_N_FRAMES})", yolo_ms)
stats("Without YOLO", no_yolo_ms)
print(f"Calibrated after {calibrated_at} measured frames" if calibrated_at else
      "Did not finish calibrating during the run")
