"""Split Unity vector observations into named multimodal signal tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.phase4 import DEFAULT_OBS_DIM

# Matches Unity SensorManager layout (ExpectedVectorSize = 197).
JOINT_DIM = 18 * 4  # 72
CONTACT_DIM = 6 * 5  # 30
FOOT_TASTE_DIM = 6 * 8  # 48
ANTENNA_DIM = 8 * 2  # 16
PROBOSCIS_DIM = 2 + 8  # 10
INTERNAL_DIM = 8
BODY_STATE_DIM = 13
INTERNAL_OFFSET = JOINT_DIM + CONTACT_DIM + FOOT_TASTE_DIM + ANTENNA_DIM + PROBOSCIS_DIM  # 176


@dataclass
class SensorySignals:
    """Named slices of a live observation vector plus optional vision."""

    body: np.ndarray  # joints + body state
    chemical: np.ndarray  # taste + odor (+ padded)
    internal: np.ndarray
    contacts: np.ndarray
    left_eye: np.ndarray  # (1, H, W) or (H, W)
    right_eye: np.ndarray
    raw: np.ndarray
    summary: dict[str, float] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "body_norm": float(np.linalg.norm(self.body)),
            "chemical_norm": float(np.linalg.norm(self.chemical)),
            "internal_norm": float(np.linalg.norm(self.internal)),
            "mean_contact": float(self.summary.get("mean_contact", 0.0)),
            "sugar": float(self.summary.get("sugar", 0.0)),
            "bitter": float(self.summary.get("bitter", 0.0)),
            "energy": float(self.summary.get("energy", 0.0)),
            "instability": float(self.summary.get("instability", 0.0)),
        }


class SignalBuilder:
    """Builds SensorySignals from a flat Unity observation vector."""

    def __init__(
        self,
        observation_dim: int = DEFAULT_OBS_DIM,
        chemical_dim: int = 64,
        vision_size: int = 64,
    ) -> None:
        self.observation_dim = observation_dim
        self.chemical_dim = chemical_dim
        self.vision_size = vision_size

    def build(
        self,
        observation: np.ndarray,
        *,
        left_eye: np.ndarray | None = None,
        right_eye: np.ndarray | None = None,
    ) -> SensorySignals:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] < self.observation_dim:
            padded = np.zeros(self.observation_dim, dtype=np.float32)
            padded[: obs.shape[0]] = obs
            obs = padded
        elif obs.shape[0] > self.observation_dim:
            obs = obs[: self.observation_dim]

        offset = 0
        joints = obs[offset : offset + JOINT_DIM]
        offset += JOINT_DIM
        contacts = obs[offset : offset + CONTACT_DIM]
        offset += CONTACT_DIM
        foot_taste = obs[offset : offset + FOOT_TASTE_DIM]
        offset += FOOT_TASTE_DIM
        antennae = obs[offset : offset + ANTENNA_DIM]
        offset += ANTENNA_DIM
        proboscis = obs[offset : offset + PROBOSCIS_DIM]
        offset += PROBOSCIS_DIM
        internal = obs[offset : offset + INTERNAL_DIM]
        offset += INTERNAL_DIM
        body_state = obs[offset : offset + BODY_STATE_DIM]

        body = np.concatenate([joints, body_state]).astype(np.float32)
        chem_raw = np.concatenate([foot_taste, antennae, proboscis[2:]]).astype(np.float32)
        chemical = np.zeros(self.chemical_dim, dtype=np.float32)
        n = min(self.chemical_dim, chem_raw.shape[0])
        chemical[:n] = chem_raw[:n]

        blank = np.zeros((1, self.vision_size, self.vision_size), dtype=np.float32)
        left = self._normalize_vision(left_eye) if left_eye is not None else blank.copy()
        right = self._normalize_vision(right_eye) if right_eye is not None else blank.copy()

        # Channel layout: sugar is chem index 0 per foot / antenna / proboscis.
        sugar = float(np.mean(foot_taste[0::8])) if foot_taste.size else 0.0
        bitter = float(np.mean(foot_taste[1::8])) if foot_taste.size else 0.0
        sugar += float(proboscis[2]) if proboscis.size > 2 else 0.0
        bitter += float(proboscis[3]) if proboscis.size > 3 else 0.0
        mean_contact = float(np.mean(contacts[0::5])) if contacts.size else 0.0
        energy = float(internal[0]) if internal.size else 0.0
        hunger = float(internal[6]) if internal.size > 6 else 0.0
        damage = float(internal[3]) if internal.size > 3 else 0.0
        # Body instability proxy: last body_state channel is height; use angular magnitude.
        instability = float(np.mean(np.abs(body_state[3:6]))) if body_state.size >= 6 else 0.0

        # Detect all-zero internal block (common when ML-Agents received a zero pad).
        if internal.size >= INTERNAL_DIM and float(np.max(np.abs(internal))) < 1e-8:
            summary_note = {"internal_all_zero": 1.0}
        else:
            summary_note = {}

        summary = {
            "sugar": max(0.0, sugar),
            "bitter": max(0.0, bitter),
            "mean_contact": max(0.0, min(1.0, mean_contact)),
            "energy": max(0.0, min(1.0, energy)),
            "hunger": max(0.0, min(1.0, hunger)),
            "damage": max(0.0, min(1.0, damage)),
            "instability": max(0.0, instability),
            "proboscis_contact": float(proboscis[0]) if proboscis.size else 0.0,
            **summary_note,
        }

        return SensorySignals(
            body=body,
            chemical=chemical,
            internal=internal.astype(np.float32),
            contacts=contacts.astype(np.float32),
            left_eye=left,
            right_eye=right,
            raw=obs,
            summary=summary,
        )

    def _normalize_vision(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3):
            # HWC -> CHW; keep RGB when present.
            if arr.shape[-1] >= 3:
                arr = np.moveaxis(arr[..., :3], -1, 0)
            else:
                arr = arr.mean(axis=-1, keepdims=True).transpose(2, 0, 1)
        if arr.shape[-2:] != (self.vision_size, self.vision_size):
            # Nearest-neighbor resize without extra deps
            h, w = arr.shape[-2], arr.shape[-1]
            ys = (np.linspace(0, h - 1, self.vision_size)).astype(np.int32)
            xs = (np.linspace(0, w - 1, self.vision_size)).astype(np.int32)
            arr = arr[..., ys][..., xs]
        # Preserve 1 or 3 channels (do not collapse RGB to gray).
        if arr.shape[0] not in (1, 3):
            arr = arr[:1]
        if float(arr.max()) > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
