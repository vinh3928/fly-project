from __future__ import annotations

import torch
from torch import nn


class NextStateWorldModel(nn.Module):
    """Predict latent_state(t+1) from latent_state(t) and action(t).

    Phase-7 starter: single-step MLP. Later phases can add multi-step Dreamer-style rollouts.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))

    def prediction_loss(
        self, latent: torch.Tensor, action: torch.Tensor, next_latent: torch.Tensor
    ) -> torch.Tensor:
        predicted = self.forward(latent, action)
        return torch.mean((predicted - next_latent) ** 2)
