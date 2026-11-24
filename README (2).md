# ProctorAI: Real-Time Exam Proctoring

A real-time exam proctoring system. The browser streams webcam frames to a FastAPI backend over an authenticated WebSocket. The backend estimates head pose with MediaPipe FaceMesh and OpenCV `solvePnP`, detects phones with YOLOv8n, turns frame-level signals into debounced violation events, and stores them for an admin dashboard.

Detections are **telemetry for a human reviewer, not proof of cheating** (see [Limitations](#limitations)).

## Measured results

All numbers come from runs on one CPU-only Windows laptop (Intel Core i3-1215U, 8 GB RAM), Python 3.12. They are single-machine, single-person measurements, not production benchmarks.

| What | Result | How it was measured |
|---|---|---|
| Pipeline speed | **21 ms mean, 55 ms p95, 77 ms p99** per frame (about 47 FPS of headroom) | `Backend/bench_pipeline.py`: 1,769 frames over 60 s of live webcam at 480x360, timing `Detector.process_frame` only |
| Frames without YOLO | 8.7 ms mean, 12.5 ms p95 | same run; YOLO runs on every 3rd frame |
| Frames with YOLO | 46 ms mean, 69 ms p95 | same run |
| False alarms in normal use | **0 violations in an 8.9-minute session** of ordinary on-screen work (solving a coding problem, watching a video) | one recorded session, one person, mostly still |
| Detector coverage | All 7 violation types fired on a live webcam: looking left / right / up / down, phone, no face, multiple faces | one deliberate-trigger session |
| Automated tests | **60 passing** (pytest, Windows, Python 3.12) | `python -m pytest -v` |

**What these do not show:** a catch rate. The deliberate-trigger session shows each detector can fire, but I did not count attempts, so no detection accuracy figure is claimed. Concurrent-session capacity was not tested.

The live server deliberately processes at most about **6.7 frames per second per session** (`min_interval = 0.15 s`). The pipeline benchmark above is headroom, not the served frame rate.

## Architecture

```text
  Browser (Student page)                       Browser (Admin dashboard)
  - webcam capture, 480x360 JPEG frames        - statistics, session list
  - WebSocket stream + live overlay            - per-session violation viewer
  - JWT access + refresh tokens                - role-restricted REST calls
            |                                              |
            | WebSocket (first message: auth token)        | HTTPS REST (Bearer JWT)
            v                                              v
  +--------------------------------------------------------------------+
  |                          FastAPI backend                           |
  |  Auth: bcrypt, JWT access tokens, rotating refresh tokens          |
  |  SlowAPI rate limits on /register /login /refresh /logout          |
  |                                                                    |
  |  WebSocket worker                                                  |
  |   - latest-frame queue (size 1: stale frames are dropped)          |
  |   - throttle: <= ~6.7 processed frames/s per session               |
  |   - Detector (runs in a thread pool)                               |
  |       MediaPipe FaceMesh -> solvePnP head pose -> calibration      |
  |       YOLOv8n phone detection (every 3rd frame, imgsz=320)         |
  |   - Violation engine: debounce, durations, confidence              |
  |                                                                    |
  |  SQLAlchemy 2.0 + Alembic migrations (SQLite locally,              |
  |  PostgreSQL config included for Docker)                            |
  +--------------------------------------------------------------------+
```

## How the vision pipeline works

**Head pose.** Six FaceMesh landmarks (IDs 33, 263, 1, 61, 291, 199) are matched to a generic 3D face model with `cv2.solvePnP`. Pitch, yaw and roll are read from the rotation's forward vector, which avoids the +/-180 degree wraparound that Euler decomposition produces.

**Calibration.** Each user's neutral pose is learned per session. A sample is accepted only if the pose is sane (|yaw| <= 25, |roll| <= 30, |pitch| <= 50 degrees) and stable (within 10 degrees of the median of the samples collected so far). Calibration completes after 20 stable samples, and the baseline is the median. Progress is the fraction of stable samples collected, never elapsed time. If the head moves during calibration the window restarts. After 20 seconds without success the attempt is reported as timed out and restarted. A generic face model has a constant per-person pitch offset (about 30 degrees on the test face), which is why neutrality is judged by stability and the offset is removed by the baseline.

**Direction classifier.** Yaw, up and down are measured relative to the baseline, with separate thresholds and hysteresis, and a direction must persist for 3 frames to enter and 3 to clear. Thresholds are set in `.env` (`YAW_THRESHOLD_DEG`, `LOOK_UP_THRESHOLD_DEG`, `LOOK_DOWN_THRESHOLD_DEG`) and are starting values, not tuned values.

**Phone detection.** YOLOv8n (COCO class "cell phone", confidence >= 0.45, `imgsz=320`) runs on every 3rd frame with cached boxes in between. Weights are loaded with an explicit PyTorch safe-globals allowlist rather than `weights_only=False`.

**Violation engine.** Frame flags become events only after 2 consecutive frames and close after 3 clear frames. Each event stores type, confidence and duration. Handled states: no face, multiple faces (up to 2 faces tracked), invalid pose (never raises a direction violation and never counts toward calibration).

## Engineering note: the calibration bug

Calibration once stayed at 100% and never completed. Tracing it showed three separate causes:

1. The face model was mirrored relative to the landmarks. The detector flips the frame before FaceMesh, which puts landmark 33 on the image-left (captured values: x=141 for 33, x=228 for 263). With the model's left/right reversed, `solvePnP` converged to a flipped pose, and real neutral landmarks decomposed to roughly yaw -172 and roll -173 degrees. Every sample failed the pose gates.
2. The progress bar was computed from elapsed time, so it hit 100% while zero samples had been accepted.
3. The configured calibration timeout was never read, so a failed calibration could never fail explicitly.

The earlier test suite passed because its synthetic faces had the same mirrored geometry as the bug. The fix corrected the model, switched progress to accepted samples, enforced the timeout, and replaced an absolute pitch gate with a stability check. The regression tests include a case built from the real captured landmarks, and the new tests fail on the old code.

## Security

- bcrypt password hashing (72-byte limit enforced).
- JWT access tokens (30 min) and rotating refresh tokens (7 days). Reusing a revoked refresh token revokes the whole token family.
- WebSocket authentication happens in the first message, so tokens never appear in URLs.
- Role-based access control: students access only their own sessions, and admin endpoints return 403 for student tokens.
- SlowAPI rate limits: `/register` 5/min, `/login` 10/min, `/refresh` 20/min, `/logout` 20/min (in-memory, single process; Redis storage would be needed for multiple replicas).
- `SECRET_KEY` is required when `APP_ENV=production`.

## API

| Method | Path | Access |
|---|---|---|
| POST | `/register`, `/login`, `/refresh`, `/logout` | public / token |
| GET | `/health`, `/ready` | public |
| POST | `/api/v1/sessions` | authenticated |
| GET | `/api/v1/sessions/{id}`, `/api/v1/events/{id}`, `/api/v1/reports/{id}` | owner or admin |
| DELETE | `/api/v1/sessions/{id}` | owner or admin |
| WebSocket | `/ws/{session_id}` | authenticated (first message) |
| GET | `/sessions`, `/statistics`, `/api/v1/sessions/{id}/violations` | admin |

## Quickstart (Windows, PowerShell)

Requires Python 3.12 and a webcam (served from `localhost`).

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# set SECRET_KEY: python -c "import secrets; print(secrets.token_hex(32))"

alembic upgrade head
python seed.py --email admin@example.com --password "<choose-a-password>" --name "Administrator"

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd Frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500/login.html`. Create a student with `POST /register` (`{"name": "...", "email": "...", "password": "..."}`), log in, click **Start monitoring**, and look straight at the camera until the status changes from `CALIBRATING` to `SECURE`. Log in as the admin account to see sessions and violations.

Run the backend from the `Backend` directory. Pinned dependencies matter: MediaPipe 0.10.14 requires NumPy below 2.

Docker: `docker compose up --build` from `Backend` starts the API and PostgreSQL (set `SECRET_KEY` and `POSTGRES_PASSWORD` in `Backend/.env`). The PostgreSQL deployment has not been load-tested.

## Tests and benchmark

```powershell
cd Backend
python -m pytest -v          # 60 tests
python bench_pipeline.py     # pipeline timing on your webcam (close other camera users first)
```

Coverage: auth and token rotation, RBAC, session lifecycle, pose estimation and calibration (including a regression test on real captured landmarks), detector flow with a fake FaceMesh, violation debouncing, and live WebSocket streaming.

## Limitations

- **Not proof of cheating.** Glancing at notes, thinking postures, disabilities, lighting and camera placement all change pose. Output is for human review, and no composite "cheating score" is produced on purpose.
- **Accuracy is not established.** Detection accuracy and catch rate were not measured. The false-alarm result is one session of one mostly-still person.
- **Thresholds are untuned** starting values.
- **Generic face model.** Pitch carries a per-person offset that calibration removes, but a head held steadily tilted during calibration becomes the baseline.
- **Single-user testing.** Concurrent-session behavior and PostgreSQL deployment were not load-tested.
- **Refresh token stored in `localStorage`**, which is exposed to XSS; a production deployment should use httpOnly cookies.
- **Rate limiter is in-memory**, so it is per-process.

## Project structure

```text
Backend/
  app/          main.py, detector.py, pose.py, violation_engine.py, auth*.py, models.py, ...
  alembic/      versioned migrations (0001_initial, 0002_refresh_tokens)
  tests/        pose, detector flow, violation engine, auth/API, WebSocket, session lifecycle
  bench_pipeline.py, seed.py, Dockerfile, docker-compose.yml, yolov8n.pt
Frontend/       login.html, index.html (student), admin.html (dashboard)
```
