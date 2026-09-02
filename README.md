# ProctorAI — AI-Powered Online Exam Cheating Detection System

Real-time exam proctoring that watches for the things a human invigilator would: no face in frame, multiple people, looking away from the screen for too long, and phones. Built as a FastAPI backend doing computer vision over a live WebSocket feed, with JWT-secured REST APIs, role-based access, and a lightweight browser client for students and admins.

## What It Does

ProctorAI simulates the core of a real online-exam proctoring product. A student starts an exam session in the browser; their webcam feed is streamed frame-by-frame over a WebSocket to a FastAPI backend, which runs it through a computer-vision pipeline in real time. Each frame is analyzed for face presence, head pose (are they looking at the screen?), the number of people in frame, and whether a mobile phone is visible. Flagged frames are logged as violations against that student's session, and an admin dashboard lets a proctor review every session, its violation history, and an auto-computed risk score.

## Technologies Used

| Layer | Stack |
|---|---|
| **Backend** | FastAPI, Uvicorn (ASGI server), WebSockets |
| **Computer Vision** | OpenCV, MediaPipe Face Mesh (478-point landmarks), Ultralytics YOLOv8 (object detection) |
| **Auth** | OAuth2 Password Flow + JWT (`python-jose`), password hashing with `passlib`/bcrypt |
| **Data** | SQLAlchemy ORM, SQLite |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript — `fetch` for REST, native `WebSocket` API for the live feed |
| **Config** | `python-dotenv` for environment-based secrets |

## Key Features

- **Real-time head pose estimation** — MediaPipe face landmarks feed into OpenCV's `solvePnP` to compute pitch/yaw/roll every frame, smoothed over a rolling window to cut down on jitter.
- **Per-session calibration** — the first ~3 seconds of a session are used to learn that student's own neutral head position (camera angle and desk setup differ per user), and all deviations are measured relative to that personal baseline rather than a fixed global threshold.
- **Multi-signal violation detection** — flags no face detected, multiple faces in frame, and sustained looking left/right/up/down.
- **Phone detection** — YOLOv8n scans for COCO class 67 (cell phone) above a confidence threshold, sampled every 3rd frame to keep the pipeline real-time.
- **Violation cooldown** — the same violation type is only logged once per 5-second window per session, so one continuous "looking away" doesn't flood the database with duplicate entries.
- **JWT auth with roles** — `student` and `admin` roles, enforced on both REST endpoints and the WebSocket handshake (a student can only see their own sessions; only admins can see everything).
- **Automatic risk scoring** — each session report computes a `LOW` / `MEDIUM` / `HIGH` risk level from total violation count.
- **Admin dashboard** — lists every exam session, per-session violation breakdown, and aggregate stats (total sessions, phone detections, no-face count, gaze violations by direction).
- **Live annotated video** — the backend draws the face box, landmarks, pose values, and status text directly onto the frame and streams the annotated JPEG back to the client alongside the structured detection data.
- **Locked-down API docs** — Swagger UI isn't public; `/docs` requires a valid admin JWT passed as a query token.

## Architecture

```
┌──────────────────┐        WebSocket (/ws/{session_id})        ┌───────────────────────┐
│   Browser (JS)    │ ───────────── webcam frames ─────────────▶ │      FastAPI App       │
│  index / login /  │                                            │                         │
│  admin.html        │ ◀──── annotated frame + violation JSON ─── │  ┌───────────────────┐ │
└──────────────────┘                                            │  │     Detector        │ │
        │  REST (fetch)                                          │  │ MediaPipe FaceMesh   │ │
        │  /register /login                                      │  │ + solvePnP head pose │ │
        │  /api/v1/sessions, /reports, /events...                │  │ + YOLOv8n phone det. │ │
        ▼                                                        │  └───────────────────┘ │
┌──────────────────┐        JWT verified on every call           │           │             │
│  Auth (JWT/OAuth2) │◀───────────────────────────────────────── │           ▼             │
└──────────────────┘                                            │  SQLAlchemy → SQLite    │
                                                                  │  (users, sessions,      │
                                                                  │   violations)            │
                                                                  └───────────────────────┘
```

**Flow:**
1. A student logs in (`/login`) and receives a JWT.
2. The client opens an exam session (`POST /api/v1/sessions`) and connects to `/ws/{session_id}?token=...`.
3. Each webcam frame is sent as raw bytes over the socket. The `Detector` class decodes it, runs face mesh + head pose + YOLO phone detection, draws the annotations, and returns the frame plus a structured status payload.
4. If a frame is flagged, a `Violation` row is written (subject to the cooldown), tied to the `ExamSession`.
5. Admins query `/sessions`, `/statistics`, and per-session `/reports` to review activity after the fact; the risk score is derived from violation counts on read.

## How to Run It

**Backend**

```bash
cd Backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# optional — otherwise a dev default is used
echo "SECRET_KEY=your-random-secret-key" > .env

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `yolov8n.pt` weights ship in the repo, so no separate model download is needed. On first run, the app auto-creates the SQLite database and seeds two demo accounts (see below).

**Frontend**

The frontend is static HTML/JS, but serve it over HTTP rather than opening the files directly (`file://`) so the browser's WebSocket/camera permissions behave correctly:

```bash
cd Frontend
python -m http.server 5500
```

Then visit `http://localhost:5500/login.html`. The frontend calls the backend at `http://127.0.0.1:8000` by default — update that base URL in the JS if your backend runs elsewhere.

**Quick CV test without the frontend**

```bash
cd Backend
python test_client.py
```

This streams your local webcam straight to the WebSocket endpoint and shows the annotated frames returned by the detector — useful for testing the detection pipeline in isolation.

## API & Authentication

Auth is JWT-based via OAuth2's password flow. Log in to get a bearer token, then send it as `Authorization: Bearer <token>` on REST calls, or as a `token` query param on the WebSocket connection.

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `POST` | `/register` | Public | Create a student account |
| `POST` | `/login` | Public | OAuth2 password login → returns JWT |
| `GET` | `/health` | Public | Health check |
| `POST` | `/api/v1/sessions` | Student | Start a new exam session |
| `GET` | `/api/v1/sessions/{id}` | Owner / Admin | Session status + violation count |
| `DELETE` | `/api/v1/sessions/{id}` | Owner / Admin | End a session |
| `POST` | `/api/v1/detect` | Auth | Run detection on a single uploaded frame |
| `GET` | `/api/v1/events/{session_id}` | Owner / Admin | Ordered violation log for a session |
| `GET` | `/api/v1/reports/{session_id}` | Owner / Admin | Violation summary + computed risk score |
| `GET` | `/api/v1/sessions/{id}/violations` | Admin | Raw violation records |
| `GET` | `/sessions` | Admin | All exam sessions |
| `GET` | `/statistics` | Admin | Aggregate violation stats across all sessions |
| `WS` | `/ws/{session_id}?token=` | Owner / Admin | Live video stream → real-time detection results |

**Demo accounts** (auto-seeded on startup, for local testing only):

| Role | Email | Password |
|---|---|---|
| Student | `student@test.com` | `password123` |
| Admin | `admin@test.com` | `admin123` |

> These are convenience seed accounts for local demo/testing — swap them out and rotate `SECRET_KEY` before any real deployment.


## Future Improvements

- Rewrite the frontend in React for a more maintainable, componentized UI
- Add audio-based cheating signals (detecting speech, multiple voices)
- Support a secondary camera angle (e.g., phone camera) for room coverage
- Store session recordings/clips of flagged moments, not just event logs
- Fine-tune the phone-detection model on exam-specific footage to cut false positives
- Containerize with Docker / docker-compose for one-command setup
- Move from SQLite to PostgreSQL for multi-user production use
- Add refresh tokens and token revocation instead of a single short-lived JWT
- Email/webhook alerts when a session crosses the HIGH risk threshold

---

**Author:** Lakshit Patel — B.Tech CSE, VIT-AP University
