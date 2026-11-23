# ProctorAI — Online Cheating Preventer

An intelligent, real-time AI-powered online exam proctoring and integrity monitoring system. ProctorAI combines computer vision (MediaPipe FaceMesh for 3D head-pose estimation and YOLOv8 for cell phone detection) with a high-performance asynchronous FastAPI backend, WebSockets, SQLAlchemy ORM, Alembic migrations, and role-based security.

---

## 1. System Architecture

```text
+------------------------------------------------------------------------+
¦                        CLIENT / BROWSER TIERS                          ¦
¦                                                                        ¦
¦   [Student Monitoring Page]                     [Admin Dashboard]      ¦
¦     - Camera Capture (640x480)                    - Real-time Stats    ¦
¦     - Authenticated WS Streaming                  - Exam Session List  ¦
¦     - Live Canvas Overlay                         - Violation Viewer   ¦
¦     - LocalStorage JWT (Access/Refresh)           - Swagger OpenAPI    ¦
+------------------------------------------------------------------------+
                   ¦ WSS Binary Frames                    ¦ HTTPS REST
                   ¦ {"type":"auth","token":...}          ¦ Bearer JWT
                   ?                                      ?
+------------------------------------------------------------------------+
¦                       FASTAPI BACKEND SERVICE                          ¦
¦                                                                        ¦
¦   +-- SlowAPI Rate Limiting (In-Memory / Redis pluggable)              ¦
¦   +-- Security Middleware (CORS, Request ID, Exception Handler)        ¦
¦   +-- Auth Engine (bcrypt 72-byte max, JWT HS256, Refresh Rotation)    ¦
¦   ¦                                                                    ¦
¦   +-- Asynchronous WebSocket Worker Pool                               ¦
¦   ¦     +-- Latest-Frame Bounded Queue (O(1) backpressure buffer)      ¦
¦   ¦     +-- Rate Throttling (~6.7 FPS inference target)               ¦
¦   ¦     ¦                                                              ¦
¦   ¦     +-- Vision Inference Engine (Threadpool / Process-Safe)        ¦
¦   ¦     ¦     +-- MediaPipe FaceMesh (468 landmarks, solvePnP 3D pose) ¦
¦   ¦     ¦     ¦     +-- Adaptive Baseline Calibration (pitch/yaw/roll) ¦
¦   ¦     ¦     +-- YOLOv8n Phone Detector (COCO Class 67, conf >= 0.45) ¦
¦   ¦     ¦                                                              ¦
¦   ¦     +-- Debounced Violation Engine                                 ¦
¦   ¦           +-- N-frame persistence filter (false positive rejection)¦
¦   ¦           +-- Duration & Confidence Tracking                       ¦
¦   ¦                                                                    ¦
¦   +-- Database Access Layer (SQLAlchemy 2.0 / SQLite / PostgreSQL)      ¦
¦         +-- Alembic Versioned Migrations (0001_initial, 0002_refresh)  ¦
¦         +-- Cascade Constraints & Safe Rollback Handling               ¦
+------------------------------------------------------------------------+
```

---

## 2. Key Features

- **Real-Time Proctoring Pipeline**: Streams browser webcam frames over WebSocket, processes them asynchronously, and returns annotated overlays with telemetry metadata.
- **3D Head-Pose Estimation**: Uses MediaPipe FaceMesh 6 landmark points (`solvePnP` and `decomposeProjectionMatrix`) with moving average smoothing and an adaptive user calibration baseline.
- **YOLOv8 Phone Detection**: Runs YOLOv8n with thread-safe inference lock and process-shared model management, filtering specifically for class 67 (cell phone) with confidence thresholding.
- **Multi-Signal Debounced Violation Engine**: Prevents single-frame noise from polluting audit logs. Violations require sustained observations (e.g., 2+ frames) to open and clear intervals (3+ frames) to close, recording exact duration.
- **Enterprise-Grade Authentication & Security**:
  - `bcrypt` password hashing enforcing the 72-byte limit.
  - Short-lived JWT access tokens (30 minutes) + long-lived rotating refresh tokens (7 days).
  - Refresh-token replay detection (automatic revocation of entire token family upon reuse).
  - First-message WebSocket authentication handshake (no sensitive JWT tokens leaked in WebSocket URLs or query strings).
  - In-memory rate limiting on sensitive routes (`/login`, `/register`, `/refresh`, `/logout`).
  - Strict role-based access control (RBAC): students access only their own sessions; admin endpoints reject student tokens.
- **Dual Database Support**: SQLite for instantaneous local zero-config testing and PostgreSQL for production Dockerized deployments.
- **Alembic Versioned Migrations**: Clean declarative migrations without relying on runtime `create_all()`.
- **Comprehensive Automated Test Suite**: 27 unit and integration tests covering auth, token rotation, replay prevention, session lifecycle, detector calibration, violation debouncing, and live WebSocket streaming.

---

## 3. Computer Vision Pipeline & Limitations

### Face Mesh & Head Pose
1. **Landmarks**: Extracts 6 facial landmarks (nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner).
2. **Perspective-n-Point (`solvePnP`)**: Solves 3D-to-2D pose using an iterative camera projection matrix.
3. **Calibration**: Samples the first 5 seconds of centered, neutral head orientation to establish an individualized pitch/yaw baseline, compensating for varied webcam elevations, monitor heights, and desk angles.
4. **Debouncing**: Ring-buffer smoothing (`deque(maxlen=5)`) filters out micro-tremors and sensor noise.

### YOLOv8 Phone Detection
- Leverages `yolov8n.pt` quantized weights for ultra-low latency inference (~15ms on modern CPUs).
- Bounded to `imgsz=320` and throttled to run every 3rd processed frame with cached bounding boxes for optimal frame rate balance.

### Critical Limitations & Ethical Framing
> [!IMPORTANT]
> **Head-pose deviations and automated detections are NOT proof of cheating.**
> - Natural glances at scratch paper, thinking postures, physical handicaps, lighting variations, or webcam positions can alter pose angles.
> - Detections represent **observational telemetry for human proctor review**, not automated academic penalties.
> - Composite cheating risk scores are intentionally avoided in favor of transparent, defensible violation counts, timestamps, and durations.

---

## 4. Quickstart Guide (Local Development)

### Prerequisites
- Python 3.12+
- Web browser (Chrome / Edge / Firefox) with webcam access (localhost or HTTPS required)

### Step 1: Virtual Environment & Dependencies
```powershell
# Navigate to Backend
cd proctorai_final/Backend

# Create virtual environment
python -m venv --system-site-packages .venv

# Activate virtual environment
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Configure Environment
```powershell
# Copy the environment template
copy .env.example .env
```

### Step 3: Run Database Migrations & Seed Admin
```powershell
# Apply Alembic schema migrations
alembic upgrade head

# Seed administrator account
python seed.py --email admin@example.com --password "admin1234" --name "Administrator"
```

### Step 4: Start Backend Server
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API is now live at `http://127.0.0.1:8000`. Swagger documentation is available at `http://127.0.0.1:8000/docs` (admin authenticated) or `/openapi.json`.

### Step 5: Start Frontend Server
In a separate terminal:
```powershell
cd proctorai_final/Frontend
python -m http.server 5500
```
Open your browser to:
`http://127.0.0.1:5500/login.html`

---

## 5. Demonstration Walkthrough

### A. Student Monitoring Flow
1. Navigate to `http://127.0.0.1:5500/login.html`.
2. Register a student or login:
   - To register via API / Swagger: `POST /register` with `{"name": "Student", "email": "student@example.com", "password": "password123"}`.
   - Or log in with:
     - **Email**: `student@example.com`
     - **Password**: `password123`
3. You will be redirected to `index.html` (Exam Monitoring).
4. Click **"Start monitoring"**:
   - Browser prompts for webcam permission. Allow access.
   - Session status changes to `Connecting…` then `Live`.
   - The status box displays `CALIBRATING (XX%)`. Look directly at the screen for 5 seconds until calibration completes (`STATUS : SECURE`).
   - Test violations:
     - Look far left or right (`FLAGGED : LOOKING LEFT` / `RIGHT`).
     - Look down (`FLAGGED : LOOKING DOWN`).
     - Step away or cover camera (`FLAGGED : NO FACE DETECTED`).
     - Hold up a mobile phone to the camera (`FLAGGED : MOBILE PHONE DETECTED`).
5. Click **"Stop monitoring"**:
   - The WebSocket closes cleanly, webcam tracks release, and the session status updates to `COMPLETED`.

### B. Admin Dashboard Flow
1. Navigate to `http://127.0.0.1:5500/login.html`.
2. Log in with admin credentials:
   - **Email**: `admin@example.com`
   - **Password**: `admin1234`
3. You will be redirected to `admin.html`.
4. Inspect:
   - **Global Statistics Cards**: Total Sessions, Total Violations, Phone Detections.
   - **Exam Sessions List**: View active and completed student sessions.
   - Click on any session to load its detailed recorded violations with timestamps, durations, and confidence levels.
   - Click **"API Schema"** to inspect OpenAPI specifications.

---

## 6. Automated Testing

Run the test suite with:
```powershell
cd proctorai_final/Backend
pytest -v
```
All 27 automated tests pass deterministically without webcam or network dependencies:
- **Auth & Security**: Password hashing, 72-byte max check, JWT creation/decoding, refresh-token rotation, token family replay invalidation, logout revocation, inactive account blocking.
- **Authorization**: Student access controls, cross-student session isolation, admin role privileges.
- **Session Lifecycle**: Idempotent session start/stop, duplicate active session rejection (HTTP 409), 404 handling.
- **Computer Vision**: FaceMesh landmark extraction, pose estimation, calibration sample collection, extreme pose rejection.
- **Event Engine**: Multi-frame debounce, multi-signal tracking, duration calculation, finalization cleanup.
- **WebSocket Protocol**: Auth handshake validation, unauthorized user rejection, frame-size limit enforcement, corrupted image handling, database violation persistence.

---

## 7. Docker Deployment

Deploy using Docker Compose with PostgreSQL:
```powershell
cd proctorai_final/Backend
docker compose up --build
```
This starts:
1. `db`: PostgreSQL 16 Alpine container with healthchecks and persistent volume storage.
2. `api`: Python 3.12 container that applies Alembic migrations automatically on startup and serves Uvicorn on port 8000.

---

## 8. Security Architecture & Threat Model

| Threat / Concern | Mitigation Implemented |
|:---|:---|
| **Credential stuffing / Brute-force** | SlowAPI in-memory rate limits (5/min on register, 10/min on login). |
| **Password security** | Salted `bcrypt` hashing with UTF-8 byte truncation protection (72 bytes). |
| **Token interception in URLs** | WebSocket authenticates via initial JSON message (`auth`), NEVER via query params in URLs. |
| **Stolen refresh token** | Hashed token storage (`sha256`), rotation on every refresh, and family revocation upon replay. |
| **Session tampering** | Server-side user ID extraction from verified JWT `sub` claim; user ID from client is never trusted. |
| **Resource exhaustion / DoS** | Frame size capped (`MAX_FRAME_BYTES = 2MB`), 1-frame bounded queue dropping stale frames. |
| **Cross-Origin attacks** | Explicit CORS allowlist configured via `CORS_ORIGINS`. |
| **Frontend Token Storage Note** | LocalStorage is used for prototype simplicity; for high-security production deployments, HttpOnly, Secure, SameSite cookies with a Backend-for-Frontend (BFF) proxy are recommended. |
