"""
Quick end-to-end test for the Phase 1 API.

Captures your webcam locally, sends each frame to the FastAPI WebSocket
endpoint, and displays the annotated frame + data that comes back.
This stands in for a real frontend until Phase 6 / the React rewrite.

Run the server first:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then, in a second terminal:
    python test_client.py
"""

import asyncio
import base64
import json
import platform

import cv2
import numpy as np
import websockets

SERVER_URL = "ws://localhost:8000/ws/test-session-1"

# Windows' default Proactor event loop occasionally throws spurious
# WinError 64 ("network name no longer available") resets under rapid
# socket I/O like this. The Selector loop doesn't have this issue and
# is fine here since we aren't spawning subprocesses.
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def stream_webcam():
    cap = (
        cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if platform.system() == "Windows"
        else cv2.VideoCapture(0)
    )

    if not cap.isOpened():
        print("Unable to open webcam.")
        return

    async with websockets.connect(SERVER_URL, max_size=None) as ws:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Downscale before sending — smaller payloads mean less
                # chance of a slow/dropped send on a laggy connection,
                # and less work for the server per frame.
                frame = cv2.resize(frame, (640, 480))

                # Encode as JPEG before sending — keeps each message small
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                await ws.send(buffer.tobytes())

                response = await ws.recv()
                data = json.loads(response)

                if "error" in data:
                    print("Server error:", data["error"])
                    continue

                # Decode the annotated frame the server sent back
                frame_bytes = base64.b64decode(data["frame"])
                annotated = cv2.imdecode(
                    np.frombuffer(frame_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                cv2.imshow("Proctor Test Client", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(stream_webcam())