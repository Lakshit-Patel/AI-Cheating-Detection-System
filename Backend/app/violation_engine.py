from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class ActiveViolation:
    violation_type: str
    started_at: datetime
    confidence: float | None = None
    details: str | None = None


class ViolationEngine:
    """Converts frame-level detector flags into debounced violation events."""

    FLAG_TO_TYPE = {
        "phone_detected": "MOBILE PHONE DETECTED",
        "no_face": "NO FACE DETECTED",
        "multiple_faces": "MULTIPLE FACES",
        "looking_left": "LOOKING LEFT",
        "looking_right": "LOOKING RIGHT",
        "looking_down": "LOOKING DOWN",
        "looking_up": "LOOKING UP",
    }

    def __init__(self, start_frames: int = 2, clear_frames: int = 3):
        self.start_frames = max(1, start_frames)
        self.clear_frames = max(1, clear_frames)
        self.candidates: dict[str, int] = {}
        self.absent_counts: dict[str, int] = {}
        self.active: dict[str, ActiveViolation] = {}

    def current_types(self, data: dict) -> list[str]:
        types = []
        for flag, violation_type in self.FLAG_TO_TYPE.items():
            if data.get(flag):
                types.append(violation_type)
        # Phone is independently observable; preserve it alongside face/pose signals.
        return types

    def update(self, data: dict, now: datetime | None = None):
        now = now or datetime.now(UTC)
        observed = set(self.current_types(data))
        started: list[ActiveViolation] = []
        ended: list[ActiveViolation] = []

        for violation_type in observed:
            self.candidates[violation_type] = self.candidates.get(violation_type, 0) + 1
            self.absent_counts[violation_type] = 0
            if violation_type not in self.active and self.candidates[violation_type] >= self.start_frames:
                confidence = data.get("phone_confidence") if violation_type == "MOBILE PHONE DETECTED" else None
                event = ActiveViolation(violation_type, now, confidence)
                self.active[violation_type] = event
                started.append(event)

        for violation_type in list(self.candidates):
            if violation_type not in observed:
                self.candidates[violation_type] = 0
                self.absent_counts[violation_type] = self.absent_counts.get(violation_type, 0) + 1

        for violation_type, event in list(self.active.items()):
            if violation_type not in observed:
                if self.absent_counts.get(violation_type, 0) >= self.clear_frames:
                    event.details = f"duration={max(0.0, (now - event.started_at).total_seconds()):.2f}s"
                    ended.append(event)
                    del self.active[violation_type]
                    self.candidates[violation_type] = 0
                    self.absent_counts[violation_type] = 0

        return started, ended

    def finalize(self, now: datetime | None = None):
        now = now or datetime.now(UTC)
        ended = []
        for event in list(self.active.values()):
            event.details = f"duration={max(0.0, (now - event.started_at).total_seconds()):.2f}s"
            ended.append(event)
        self.active.clear()
        self.candidates.clear()
        self.absent_counts.clear()
        return ended
