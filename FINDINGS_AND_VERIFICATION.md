# ProctorAI - audit findings, fixes, and what is / is not verified

## Confirmed bugs (reproduced with code, not guessed)
| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Calibration can never complete.** Old MODEL_POINTS put the eyes *closer* to the camera than the nose and were left/right-mirrored relative to the `cv2.flip` the detector applies. Neutral pose decomposed to pitch ~ -174, roll ~ 180. Calibration only accepts |pitch| <= 25, so every frame was rejected. | Synthetic-landmark run of the original `estimate_head_pose`. Your DB: 15 sessions, 14 violations, **zero** left/right/up/down events. |
| 2 | Pitch wrapped through +/-180 (up 15 deg -> +153, down 15 deg -> -161). Yaw gain ~0.55 (20 deg true read 10.9). | Same run. |
| 3 | `float(euler_angles[i])` raises `TypeError` on NumPy >= 2.x (only a warning on 1.26). Any exception in the processor task killed the socket with 1011 and the UI showed nothing. | Reproduced on NumPy 2.4.4. **Suspected** cause of your "ready then black preview"; confirm on your machine (check backend log for a traceback after `ready`). |
| 4 | Session created over REST, WebSocket fails -> row stays `ACTIVE` forever -> every later start returns 409. | Code path; fixed. |
| 5 | "NO VALID POSE" status was prefixed `FLAGGED` so the UI showed a violation banner for an event the engine never persists. | Code. |
| 6 | Per-frame `logger.info` and a frontend `console.log` of every base64 frame. | Code. |
| 7 | `backend/.env` is byte-identical to `.env.example`: your JWT signing key is the repo template value. **Regenerate it**: `python -c "import secrets; print(secrets.token_hex(32))"`. | diff |
| 8 | Installed mediapipe 0.10.33 has no `mp.solutions`; the app needs the pinned 0.10.14 (needs numpy<2). Now fails with an explicit message. | Import check. |

## What changed
- NEW `app/pose.py` (correct model, bounded angles from the forward vector, PnP initial guess, independent left/right & up/down hysteresis, calibrator).
- `app/detector.py` uses it; thresholds from env; no-face / multi-face / bad pose never calibrate and never *raise* a direction violation; phone class resolved by name from the model.
- `app/model_manager.py` lazy-imports ultralytics; missing weights disables phone detection (visible in `/ready`).
- `app/main.py` orphan-session reaping (startup + on create), live-socket registry, text-frame handling, structured `{"type":"error"}` messages, per-frame failure tolerance, open-violation finalisation on stop.
- `Frontend/index.html` shows server errors, calibration guidance, camera-permission errors; releases camera and completes the session when the socket dies.
- Tests: `test_pose.py`, `test_detector_flow.py` (new), `test_session_lifecycle.py` (new), `test_detector.py` (obsolete test removed).

## Verification status - read this
**Actually executed (offline sandbox, 20 tests, all passed):** `tests/test_pose.py`, `tests/test_detector_flow.py` - pose sign/gain/no-wraparound on synthetic landmarks of a *different* face shape, noise robustness, calibration, hysteresis, each direction firing alone and recovering through `Detector.process_frame` with a **fake FaceMesh**.

**Written but NOT run** (sandbox lacks FastAPI/SQLAlchemy/pytest and has no network): `test_session_lifecycle.py`, and the existing API/WebSocket suites against my `main.py` changes. Run `python -m pytest -q` yourself.

**Not verified at all:** real webcam accuracy, MediaPipe landmark behaviour, YOLO phone detection, FPS, the browser black-preview fix, deployment. Default thresholds (yaw 20, up 18, down 15) are starting values, **not tuned**. Synthetic tests prove the geometry/sign logic, not real-world accuracy.

**Not done:** admin-page audit, refresh-token-in-localStorage risk (still there), full README rewrite, dependency compatibility run, deployment.

## Run it (PowerShell)
```powershell
cd "D:\Proctor_AI FINAL\proctorai_final\Backend"
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp,'solutions'))"   # expect 0.10.14 True
python -c "import numpy; print(numpy.__version__)"                                    # expect 1.26.4
python -m pytest -q
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000        # run from Backend\, NOT the parent folder
# separate terminal:
cd ..\Frontend; python -m http.server 5500
```
Stale-module check for the old `process_frame` error: `python -c "import app.detector as d; print(d.__file__, hasattr(d.Detector,'process_frame'))"` and delete every `__pycache__`.

## Manual hardware checklist
1. Face camera, level head: status goes CALIBRATING -> SECURE within ~5 s.
2. Turn head ~30 deg left / right / up / down, hold 1 s: only the matching banner appears; return to centre: it clears.
3. Small nods/turns (<10 deg): no flag.  4. Cover camera: NO FACE after ~2 frames; calibration never completes while covered.
5. Second person: MULTIPLE FACES.  6. Phone in view: MOBILE PHONE (check `/ready` shows `phone_model.loaded: true`).
7. Stop: camera light off, session COMPLETED, open violations have durations.  8. Close the tab mid-session, start again: no 409.
9. If a direction is reversed or too sensitive on your camera, adjust the `*_THRESHOLD_DEG` env values and tell me the printed relative yaw/pitch.
