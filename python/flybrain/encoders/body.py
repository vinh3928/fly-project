from __future__ import annotations

import torch
from torch import nn


class BodyEncoder(nn.Module):
    """MLP encoder for proprioception, contact, and body-state vectors."""

    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
