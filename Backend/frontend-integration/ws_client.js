// Browser-side contract for the new proctoring WebSocket.
// Adapt the selectors/UI state to the existing project rather than replacing the UI.

export function openProctorSocket({ sessionId, accessToken, onDetection, onError }) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/${sessionId}`);

  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "auth", token: accessToken }));
  };

  ws.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    const message = JSON.parse(event.data);
    if (message.type === "detection") onDetection?.(message);
    else if (message.error) onError?.(message.error);
  };

  ws.onerror = () => onError?.("WebSocket connection failed");
  return ws;
}

export function sendJpegFrame(ws, blob) {
  if (ws.readyState === WebSocket.OPEN) ws.send(blob);
}
