"""Episode log loading and replay helpers for Phase 3 recordings."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from flybrain.phase3 import (
    EVENTS_FILE,
    METADATA_FILE,
    POSE_FILE,
    REQUIRED_STREAM_FILES,
    SCHEMA_VERSION,
    TASTE_FILE,
    VISION_MANIFEST_FILE,
)


@dataclass(frozen=True)
class VisionFrameRef:
    step: int
    timestamp: float
    frame_index: int
    left_file: str
    right_file: str
    body_forward: tuple[float, float, float]
    left_eye_forward: tuple[float, float, float]
    right_eye_forward: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeEvent:
    step: int
    timestamp: float
    event: str
    payload: dict[str, Any]


class EpisodeReplay:
    """Load a Phase 3 episode directory and iterate synchronized streams."""

    def __init__(self, episode_dir: str | Path) -> None:
        self.root = Path(episode_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {self.root}")

        self.metadata: dict[str, Any] = self._load_json(METADATA_FILE)
        self.events: list[EpisodeEvent] = self._load_events()
        self.pose = self._load_csv(POSE_FILE)
        self.taste = self._load_csv(TASTE_FILE)
        self.vision_frames: list[VisionFrameRef] = self._load_vision_manifest()

    @classmethod
    def from_directory(cls, episode_dir: str | Path) -> EpisodeReplay:
        return cls(episode_dir)

    def validate_schema(self) -> None:
        missing = [name for name in REQUIRED_STREAM_FILES if not (self.root / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required episode files: {missing}")

        version = int(self.metadata.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version={SCHEMA_VERSION}, got {version}")

        for stream_name, rows in (("pose", self.pose), ("taste", self.taste)):
            if not rows:
                raise ValueError(f"{stream_name} stream is empty")
            if "sim_time" not in rows[0]:
                raise ValueError(f"{stream_name} rows missing sim_time")
            for row in rows:
                if row.get("sim_time") in (None, ""):
                    raise ValueError(f"{stream_name} has blank sim_time")

        for event in self.events:
            if event.timestamp is None:
                raise ValueError("events must include timestamp")

    def iter_pose(self) -> Iterator[dict[str, str]]:
        yield from self.pose

    def iter_taste(self) -> Iterator[dict[str, str]]:
        yield from self.taste

    def iter_vision(self) -> Iterator[VisionFrameRef]:
        yield from self.vision_frames

    def vision_matches_pose_orientation(self, min_dot: float = 0.95) -> bool:
        """True when each vision frame's eye forwards align with body forward."""
        if not self.vision_frames:
            return True

        for frame in self.vision_frames:
            body = _normalize(frame.body_forward)
            left = _normalize(frame.left_eye_forward)
            right = _normalize(frame.right_eye_forward)
            if _dot(body, left) < min_dot or _dot(body, right) < min_dot:
                return False
        return True

    def taste_nonzero_only_with_contact(self) -> bool:
        """Within taste.csv, non-zero proboscis taste requires proboscis_contact=1."""
        for row in self.taste:
            contact = row.get("proboscis_contact", "0")
            sugar = float(row.get("proboscis_sugar", 0) or 0)
            bitter = float(row.get("proboscis_bitter", 0) or 0)
            if (sugar > 0 or bitter > 0) and contact not in ("1", "1.0", "true", "True"):
                return False
        return True

    def _load_json(self, relative: str) -> dict[str, Any]:
        path = self.root / relative
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _load_csv(self, relative: str) -> list[dict[str, str]]:
        path = self.root / relative
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _load_events(self) -> list[EpisodeEvent]:
        path = self.root / EVENTS_FILE
        events: list[EpisodeEvent] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                events.append(
                    EpisodeEvent(
                        step=int(payload.get("step", 0)),
                        timestamp=float(payload.get("timestamp", 0.0)),
                        event=str(payload.get("event", "")),
                        payload=payload,
                    )
                )
        return events

    def _load_vision_manifest(self) -> list[VisionFrameRef]:
        path = self.root / VISION_MANIFEST_FILE
        if not path.exists():
            return []

        frames: list[VisionFrameRef] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                frames.append(
                    VisionFrameRef(
                        step=int(payload["step"]),
                        timestamp=float(payload["timestamp"]),
                        frame_index=int(payload["frame_index"]),
                        left_file=str(payload["left_file"]),
                        right_file=str(payload["right_file"]),
                        body_forward=_as_vec3(payload["body_forward"]),
                        left_eye_forward=_as_vec3(payload["left_eye_forward"]),
                        right_eye_forward=_as_vec3(payload["right_eye_forward"]),
                    )
                )
        return frames


def _as_vec3(value: Any) -> tuple[float, float, float]:
    return float(value[0]), float(value[1]), float(value[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = (_dot(v, v)) ** 0.5
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0
    return v[0] / norm, v[1] / norm, v[2] / norm
