"""Recurrent stereo vision head for Phase 5 navigation cues."""

from __future__ import annotations

from dataclasses import dataclass

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

    def to_dict(self) -> dict[str, float]:
        return {
            "target_bearing": float(self.target_bearing),
            "target_confidence": float(self.target_confidence),
            "wall_hazard": float(self.wall_hazard),
            "forward_bias": float(self.forward_bias),
            "turn_bias": float(self.turn_bias),
        }


class RecurrentVisionHead(nn.Module):
    """VisionEncoder + GRU → navigation cues (no privileged target XYZ)."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = VisionEncoder(in_channels=1, latent_dim=latent_dim)
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
        )
        self.hidden_dim = hidden_dim
        self._hidden: torch.Tensor | None = None

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
        # bearing, confidence, wall_hazard, forward_bias, turn_bias
        bearing = torch.tanh(raw[:, 0])
        confidence = torch.sigmoid(raw[:, 1])
        wall = torch.sigmoid(raw[:, 2])
        forward = torch.tanh(raw[:, 3])
        turn = torch.tanh(raw[:, 4])
        cues = torch.stack([bearing, confidence, wall, forward, turn], dim=-1)
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
        if left_t.ndim == 3:
            left_t = left_t.unsqueeze(0)
        if right_t.ndim == 3:
            right_t = right_t.unsqueeze(0)
        if self._hidden is None or self._hidden.shape[1] != left_t.shape[0]:
            self.reset_state(batch_size=left_t.shape[0], device=dev)
        cues, self._hidden = self.forward(left_t, right_t, self._hidden)
        row = cues[0].cpu().numpy()
        return VisionNavCues(
            target_bearing=float(row[0]),
            target_confidence=float(row[1]),
            wall_hazard=float(row[2]),
            forward_bias=float(row[3]),
            turn_bias=float(row[4]),
        )


def heuristic_cues_from_frames(left: np.ndarray, right: np.ndarray) -> VisionNavCues:
    """Label-free heuristic for smoke tests before/without a trained checkpoint."""
    l = np.asarray(left, dtype=np.float32).reshape(-1, left.shape[-2], left.shape[-1])
    r = np.asarray(right, dtype=np.float32).reshape(-1, right.shape[-2], right.shape[-1])
    frame = 0.5 * (l[0] + r[0])
    h, w = frame.shape
    # Bright mass centroid → bearing
    bright = np.clip(frame - 0.6, 0.0, 1.0)
    mass = float(bright.sum()) + 1e-6
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    bearing = float((bright.sum(axis=0) * xs).sum() / mass)
    confidence = float(np.clip(mass / (h * w * 0.08), 0.0, 1.0))
    left_bar = float(frame[:, : max(1, w // 8)].mean())
    right_bar = float(frame[:, -max(1, w // 8) :].mean())
    wall_hazard = float(np.clip(1.0 - min(left_bar, right_bar) / 0.35, 0.0, 1.0))
    forward = float(np.clip(0.7 * confidence - 0.8 * wall_hazard, -1.0, 1.0))
    turn = float(np.clip(bearing * confidence, -1.0, 1.0))
    return VisionNavCues(bearing, confidence, wall_hazard, forward, turn)
