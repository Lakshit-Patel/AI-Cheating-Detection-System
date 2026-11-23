# ProctorAI frontend integration

The three supplied HTML pages have been updated to match the rebuilt backend API and WebSocket contract. They remain plain HTML/CSS/JavaScript, so no frontend build step is required.

## Local setup (Windows PowerShell)

1. Start the backend from `Backend/` after installing requirements, creating `.env`, running `alembic upgrade head`, and creating an admin account. Follow `Backend/README.md`.
2. From `Frontend/`, run:

   ```powershell
   python -m http.server 5500
   ```

3. Open `http://127.0.0.1:5500/login.html`. The pages use `http://127.0.0.1:8000` by default.
4. To change the API base URL for this browser, open the browser console and run `localStorage.setItem('apiBase', 'http://127.0.0.1:8000')`, then reload. For deployment, use HTTPS for the API and WSS for the WebSocket.

## Frontend changes

- Login stores both access and refresh tokens, handles errors as text, and routes by the JWT role claim.
- REST calls refresh the access token and rotate the refresh token when the access token is near expiry or the API returns 401.
- The student page creates a session, opens `/ws/{session_id}` without a token in the URL, sends the first-message auth handshake, waits for `ready`, and only then sends binary JPEG frames. It avoids overlapping frame conversions and stops camera tracks when monitoring stops.
- The admin page uses the backend's actual session field (`id`), displays violation event data, and uses authenticated API calls. The API Schema button opens `/openapi.json` without putting a token in the URL.

## Notes

- The current browser implementation stores tokens in `localStorage`, which is convenient for a student project but remains exposed to successful same-origin XSS. A production deployment should consider a backend-for-frontend with Secure, HttpOnly, SameSite cookies and a stricter CSP.
- Browser camera access requires localhost or HTTPS.
- Detection output is evidence for review, not proof of cheating.
