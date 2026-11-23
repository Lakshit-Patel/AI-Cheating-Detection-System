# Frontend integration contract

The backend WebSocket protocol is intentionally different from the old implementation.

## Login

`POST /login` remains form-urlencoded:

- `username`: email
- `password`: password

The response contains:

- `access_token`
- `refresh_token`
- `token_type`
- `expires_in`

## Create a session

Send the access token as a bearer token to:

`POST /api/v1/sessions`

The response contains `session_id`.

## WebSocket

Connect without putting the JWT in the URL:

`ws://127.0.0.1:8000/ws/{session_id}`

Immediately after `open`, send:

```json
{"type":"auth","token":"ACCESS_TOKEN"}
```

The server replies:

```json
{"type":"ready","session_id":123}
```

Then send binary JPEG frames. The server returns JSON detection messages containing the detection fields and a base64 JPEG in `frame`.

## Refresh

When the access token expires, call:

`POST /refresh`

with:

```json
{"refresh_token":"..."}
```

Replace both stored tokens with the returned pair. Do not reuse the old refresh token.
