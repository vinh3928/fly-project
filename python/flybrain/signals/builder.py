"""Split Unity vector observations into named multimodal signal tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.phase4 import DEFAULT_OBS_DIM, SMELL_VEC_DIM

# Matches Unity SensorManager layout (ExpectedVectorSize = 197).
JOINT_DIM = 18 * 4  # 72
CONTACT_DIM = 6 * 5  # 30
FOOT_TASTE_DIM = 6 * 8  # 48
ANTENNA_DIM = 8 * 2  # 16
# Unity ChemicalChannels: Sugar,Bitter,Salt,Acid,Water,Protein,Toxin,Odor
SUGAR_CHANNEL = 0
BITTER_CHANNEL = 1
ODOR_CHANNEL = 7
# Antenna tips sit 38 mm apart (Unity AntennaOdorSensor.HalfSpacing), so left/right
# plume differences are ~9% at most; amplify the normalized difference into a
# [-1, 1] bearing ≈ 1.75·sin(angle to source) (+ = source to the right).
ANTENNA_BEARING_GAIN = 20.0
PROBOSCIS_DIM = 2 + 8  # 10
# Unity ProboscisSensor.CopyObservation: [extension, contact, gated taste × 8].
PROBOSCIS_EXTENSION_INDEX = 0
PROBOSCIS_CONTACT_INDEX = 1
INTERNAL_DIM = 8
BODY_STATE_DIM = 13
INTERNAL_OFFSET = JOINT_DIM + CONTACT_DIM + FOOT_TASTE_DIM + ANTENNA_DIM + PROBOSCIS_DIM  # 176
BODY_STATE_OFFSET = INTERNAL_OFFSET + INTERNAL_DIM
# Unity SensorManager.LinearVelocityScale: body_state[0:3] = measured velocity / 2 m/s.
BODY_LINEAR_VELOCITY_SCALE = 2.0


def body_speed(observation: np.ndarray) -> float | None:
    """Horizontal body speed (m/s) from the Unity body-state block, or None if absent."""
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.shape[0] < BODY_STATE_OFFSET + 3:
        return None
    vx = float(obs[BODY_STATE_OFFSET])
    vz = float(obs[BODY_STATE_OFFSET + 2])
    return float(np.hypot(vx, vz) * BODY_LINEAR_VELOCITY_SCALE)


def antenna_smell_vec(observation: np.ndarray) -> np.ndarray:
    """Extract left+right antenna chemistry (16-d) from a flat Unity obs vector."""
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    start = JOINT_DIM + CONTACT_DIM + FOOT_TASTE_DIM
    out = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
    if obs.shape[0] <= start:
        return out
    n = min(SMELL_VEC_DIM, ANTENNA_DIM, obs.shape[0] - start)
    out[:n] = obs[start : start + n]
    return out


def _antenna_channel(antennae: np.ndarray, channel: int) -> tuple[float, float, float]:
    """Return (left, right, max) for one chemical channel on the two antennae."""
    a = np.asarray(antennae, dtype=np.float32).reshape(-1)
    left = float(a[channel]) if a.size > channel else 0.0
    right = float(a[8 + channel]) if a.size > 8 + channel else 0.0
    return left, right, max(left, right)


def _odor_from_antennae(antennae: np.ndarray) -> tuple[float, float, float]:
    """Return (odor_left, odor_right, odor=max) from the Odor channel."""
    return _antenna_channel(antennae, ODOR_CHANNEL)


def antenna_bearing(left: float, right: float, gain: float = ANTENNA_BEARING_GAIN) -> float:
    """Signed bearing from left/right antenna intensity (+ = stronger on the right)."""
    total = float(left) + float(right)
    if total <= 1e-6:
        return 0.0
    return float(np.clip(gain * (float(right) - float(left)) / total, -1.0, 1.0))


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
    smell_vec: np.ndarray = field(
        default_factory=lambda: np.zeros(SMELL_VEC_DIM, dtype=np.float32)
    )
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
            "odor": float(self.summary.get("odor", 0.0)),
            "odor_left": float(self.summary.get("odor_left", 0.0)),
            "odor_right": float(self.summary.get("odor_right", 0.0)),
            "sugar_smell": float(self.summary.get("sugar_smell", 0.0)),
            "sugar_bearing": float(self.summary.get("sugar_bearing", 0.0)),
            "bitter_smell": float(self.summary.get("bitter_smell", 0.0)),
            "bitter_bearing": float(self.summary.get("bitter_bearing", 0.0)),
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
        smell = np.asarray(antennae, dtype=np.float32).reshape(-1).copy()
        if smell.shape[0] != SMELL_VEC_DIM:
            padded_smell = np.zeros(SMELL_VEC_DIM, dtype=np.float32)
            n_sm = min(SMELL_VEC_DIM, smell.shape[0])
            padded_smell[:n_sm] = smell[:n_sm]
            smell = padded_smell
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
        odor_left, odor_right, odor = _odor_from_antennae(antennae)
        sugar_smell_left, sugar_smell_right, sugar_smell = _antenna_channel(antennae, SUGAR_CHANNEL)
        bitter_smell_left, bitter_smell_right, bitter_smell = _antenna_channel(
            antennae, BITTER_CHANNEL
        )
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
            "odor": max(0.0, odor),
            "odor_left": max(0.0, odor_left),
            "odor_right": max(0.0, odor_right),
            # Antenna smell (at range); "sugar"/"bitter" above stay contact taste.
            "sugar_smell": max(0.0, sugar_smell),
            "sugar_smell_left": max(0.0, sugar_smell_left),
            "sugar_smell_right": max(0.0, sugar_smell_right),
            "bitter_smell": max(0.0, bitter_smell),
            "sugar_bearing": antenna_bearing(sugar_smell_left, sugar_smell_right),
            "bitter_bearing": antenna_bearing(bitter_smell_left, bitter_smell_right),
            "mean_contact": max(0.0, min(1.0, mean_contact)),
            "energy": max(0.0, min(1.0, energy)),
            "hunger": max(0.0, min(1.0, hunger)),
            "damage": max(0.0, min(1.0, damage)),
            "instability": max(0.0, instability),
            "proboscis_contact": float(proboscis[PROBOSCIS_CONTACT_INDEX])
            if proboscis.size > PROBOSCIS_CONTACT_INDEX
            else 0.0,
            "proboscis_extension": float(proboscis[PROBOSCIS_EXTENSION_INDEX])
            if proboscis.size > PROBOSCIS_EXTENSION_INDEX
            else 0.0,
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
            smell_vec=smell,
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
