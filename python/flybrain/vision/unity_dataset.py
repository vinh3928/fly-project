"""Unity vision dataset (Tools → Fly → Record Vision Dataset) → training batches.

Layout (written by ``VisionDatasetWriter.cs``):
  frames.u8   per sample: left then right eye, H×W×RGB uint8, top row first
  labels.csv  region, wall_hazard, forward_bias, turn_bias, opening_bearing, opening_strength, d0..dN
  meta.json   count, width, height, channels, columns, ...
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flybrain.vision.recurrent_head import CUE_NAMES, NUM_CUES

FRAMES_FILE = "frames.u8"
LABELS_FILE = "labels.csv"
META_FILE = "meta.json"
LABEL_COLUMNS = ("wall_hazard", "forward_bias", "turn_bias", "opening_bearing", "opening_strength")
# Signed bearings flip sign when the scene is mirrored left↔right.
MIRRORED_CUES = frozenset({"target_bearing", "turn_bias", "opening_bearing"})


@dataclass
class UnityVisionDataset:
    frames: np.ndarray  # (N, 2, H, W, 3) uint8, eye 0 = left
    labels: np.ndarray  # (N, NUM_CUES) float32 in CUE_NAMES order
    distances: np.ndarray  # (N, columns) float32
    regions: list[str]

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def batch(
        self,
        indices: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        mirror_prob: float = 0.5,
        brightness_jitter: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (left, right, targets): eyes (B, 3, H, W) float32 in [0, 1], targets (B, NUM_CUES)."""
        frames = self.frames[indices].astype(np.float32) / 255.0
        targets = self.labels[indices].copy()
        if rng is not None:
            if mirror_prob > 0.0:
                flip = rng.random(len(indices)) < mirror_prob
                frames[flip] = frames[flip][:, ::-1, :, ::-1, :]
                for name in MIRRORED_CUES:
                    targets[flip, CUE_NAMES.index(name)] *= -1.0
            if brightness_jitter > 0.0:
                gain = rng.uniform(
                    1.0 - brightness_jitter,
                    1.0 + brightness_jitter,
                    size=(len(indices), 1, 1, 1, 1),
                )
                frames = np.clip(frames * gain.astype(np.float32), 0.0, 1.0)
        eyes = np.ascontiguousarray(np.moveaxis(frames, -1, 2))  # (B, 2, 3, H, W)
        return eyes[:, 0], eyes[:, 1], targets


def load_unity_vision_dataset(directory: str | Path) -> UnityVisionDataset:
    root = Path(directory)
    meta = json.loads((root / META_FILE).read_text(encoding="utf-8"))
    count, height, width = int(meta["count"]), int(meta["height"]), int(meta["width"])
    channels = int(meta.get("channels", 3))
    raw = np.fromfile(root / FRAMES_FILE, dtype=np.uint8)
    expected = count * 2 * height * width * channels
    if raw.size != expected:
        raise ValueError(
            f"{FRAMES_FILE}: {raw.size} bytes, expected {expected} for {count} samples"
        )
    frames = raw.reshape(count, 2, height, width, channels)

    labels = np.zeros((count, NUM_CUES), dtype=np.float32)
    columns = int(meta.get("columns", 0))
    distances = np.zeros((count, columns), dtype=np.float32)
    regions: list[str] = []
    with (root / LABELS_FILE).open(encoding="utf-8", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i >= count:
                break
            regions.append(row["region"])
            for name in LABEL_COLUMNS:
                labels[i, CUE_NAMES.index(name)] = float(row[name])
            for c in range(columns):
                distances[i, c] = float(row[f"d{c}"])
    if len(regions) != count:
        raise ValueError(f"{LABELS_FILE}: {len(regions)} rows, expected {count}")
    return UnityVisionDataset(frames=frames, labels=labels, distances=distances, regions=regions)
