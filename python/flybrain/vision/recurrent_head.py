"""Recurrent stereo vision head for Phase 5 navigation cues."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain.encoders.vision import VisionEncoder


@dataclass
class VisionNavCues:
    target_bearing: float
    target_confidence: float
    wall_hazard: float
    forward_bias: float
    turn_bias: float
    # Way out of the box: mean bearing (−1 left … +1 right) and share of the view that sees out.
    opening_bearing: float = 0.0
    opening_strength: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


CUE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(VisionNavCues))
NUM_CUES = len(CUE_NAMES)
# Unit-interval cues (sigmoid); the rest are signed (tanh).
_SIGMOID_CUES = frozenset({"target_confidence", "wall_hazard", "opening_strength"})


class RecurrentVisionHead(nn.Module):
    """VisionEncoder + GRU → navigation cues (no privileged target XYZ).

    ``stateless`` heads were trained on independent frames, so inference starts from a
    zero hidden state every call instead of carrying it across ticks.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        *,
        in_channels: int = 1,
        stateless: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.stateless = stateless
        self.encoder = VisionEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, NUM_CUES),
        )
        self.hidden_dim = hidden_dim
        self._hidden: torch.Tensor | None = None
        self.register_buffer(
            "_sigmoid_mask",
            torch.tensor([name in _SIGMOID_CUES for name in CUE_NAMES]),
            persistent=False,
        )

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        dev = device or next(self.parameters()).device
        self._hidden = torch.zeros(1, batch_size, self.hidden_dim, device=dev)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # left/right: (B, 1, H, W) or (B, H, W)
        vision = 0.5 * (self.encoder(left) + self.encoder(right))
        seq = vision.unsqueeze(1)
        out, hidden_out = self.gru(seq, hidden)
        raw = self.head(out.squeeze(1))
        cues = torch.where(self._sigmoid_mask, torch.sigmoid(raw), torch.tanh(raw))
        return cues, hidden_out

    @torch.no_grad()
    def infer_cues(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        device: str = "cpu",
    ) -> VisionNavCues:
        self.eval()
        dev = torch.device(device)
        left_t = torch.as_tensor(left, device=dev, dtype=torch.float32)
        right_t = torch.as_tensor(right, device=dev, dtype=torch.float32)
        # Accept (H,W), (1|3,H,W), (B,C,H,W) — VisionEncoder converts RGB→luma when needed.
        if left_t.ndim == 2:
            left_t = left_t.unsqueeze(0).unsqueeze(0)
        elif left_t.ndim == 3:
            left_t = left_t.unsqueeze(0)
        if right_t.ndim == 2:
            right_t = right_t.unsqueeze(0).unsqueeze(0)
        elif right_t.ndim == 3:
            right_t = right_t.unsqueeze(0)
        if self.stateless or self._hidden is None or self._hidden.shape[1] != left_t.shape[0]:
            self.reset_state(batch_size=left_t.shape[0], device=dev)
        cues, self._hidden = self.forward(left_t, right_t, self._hidden)
        row = cues[0].cpu().numpy()
        return VisionNavCues(**{name: float(row[i]) for i, name in enumerate(CUE_NAMES)})


def save_vision_head(model: RecurrentVisionHead, path: str | Path, **extra: float) -> Path:
    """Checkpoint with the shape needed to rebuild the head (see :func:`load_vision_head`)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": model.latent_dim,
            "hidden_dim": model.hidden_dim,
            "in_channels": model.in_channels,
            "stateless": model.stateless,
            "cues": list(CUE_NAMES),
            **extra,
        },
        out,
    )
    return out


class IncompatibleVisionCheckpoint(ValueError):
    """Checkpoint was saved for a different cue set or head shape (e.g. pre-opening 5-cue heads)."""


def load_vision_head(path: str | Path) -> RecurrentVisionHead:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "model" not in state:
        state = {"model": state}
    saved_cues = state.get("cues")
    if saved_cues is not None and list(saved_cues) != list(CUE_NAMES):
        raise IncompatibleVisionCheckpoint(f"{path}: cues {list(saved_cues)} != {list(CUE_NAMES)}")
    model = RecurrentVisionHead(
        latent_dim=int(state.get("latent_dim", 128)),
        hidden_dim=int(state.get("hidden_dim", 128)),
        in_channels=int(state.get("in_channels", 1)),
        stateless=bool(state.get("stateless", False)),
    )
    try:
        model.load_state_dict(state["model"])
    except RuntimeError as exc:
        raise IncompatibleVisionCheckpoint(f"{path}: {exc}") from exc
    model.eval()
    return model


def heuristic_cues_from_frames(left: np.ndarray, right: np.ndarray) -> VisionNavCues:
    """Label-free heuristic for smoke tests before/without a trained checkpoint.

    Synthetic dark wall bars (Phase 5) and Unity light/near-field walls both raise
    ``wall_hazard`` so live Chamber play can avoid obstacles without a trained head.
    Accepts grayscale (H,W)/(1,H,W) or RGB (3,H,W)/(H,W,3); wall cues use luminance.
    """
    def _luma(eye: np.ndarray) -> np.ndarray:
        x = np.asarray(eye, dtype=np.float32)
        if x.ndim == 2:
            return x
        if x.ndim == 3 and x.shape[0] in (1, 3):
            return x[0] if x.shape[0] == 1 else x.mean(axis=0)
        if x.ndim == 3 and x.shape[-1] in (1, 3):
            return x[..., 0] if x.shape[-1] == 1 else x[..., :3].mean(axis=-1)
        return np.asarray(x, dtype=np.float32).reshape(x.shape[-2], x.shape[-1])

    l = _luma(left)
    r = _luma(right)
    frame = 0.5 * (l + r)
    h, w = frame.shape
    # Bright mass centroid → bearing
    bright = np.clip(frame - 0.6, 0.0, 1.0)
    mass = float(bright.sum()) + 1e-6
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    bearing = float((bright.sum(axis=0) * xs).sum() / mass)
    confidence = float(np.clip(mass / (h * w * 0.08), 0.0, 1.0))
    edge = max(1, w // 8)
    left_bar = float(frame[:, :edge].mean())
    right_bar = float(frame[:, -edge:].mean())
    left_std = float(frame[:, :edge].std())
    right_std = float(frame[:, -edge:].std())
    # Classic synthetic cue: dark edge columns (darker bar ⇒ stronger hazard).
    # 0.28 (was 0.35) so only quite dark edges score high — textured grass edges less so.
    dark_edge = float(np.clip(1.0 - min(left_bar, right_bar) / 0.28, 0.0, 1.0))
    # Unity chamber walls are often light + low-texture when close.
    def _uniform_bright(mean: float, std: float) -> float:
        if mean < 0.45:
            return 0.0
        # Grass is bright but high-std; require flatter columns (was /0.12).
        return float(np.clip(mean * (1.0 - min(1.0, std / 0.08)), 0.0, 1.0))

    light_edge = max(_uniform_bright(left_bar, left_std), _uniform_bright(right_bar, right_std))
    # Frontal fill: low stereo difference + low spatial variance ⇒ near wall.
    # Outdoor meadow has high spatial variance; keep this for near blank walls only.
    stereo = float(np.mean(np.abs(l - r)))
    global_std = float(frame.std())
    frontal = 0.0
    if stereo < 0.045 and global_std < 0.08:
        frontal = float(np.clip(0.55 + 0.35 * (0.045 - stereo) / 0.045, 0.0, 1.0))
    wall_hazard = float(np.clip(max(dark_edge, light_edge, frontal), 0.0, 1.0))
    forward = float(np.clip(0.7 * confidence - 0.8 * wall_hazard, -1.0, 1.0))
    turn = float(np.clip(bearing * confidence - 0.55 * (right_bar - left_bar), -1.0, 1.0))
    if wall_hazard > 0.45 and abs(turn) < 0.15:
        # Bias away from the darker / more occluded side when nearly frontal.
        turn = float(np.clip(-0.55 if left_bar <= right_bar else 0.55, -1.0, 1.0))
    return VisionNavCues(bearing, confidence, wall_hazard, forward, turn)
