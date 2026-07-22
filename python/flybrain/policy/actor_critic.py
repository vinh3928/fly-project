from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class ActorCriticPolicy(nn.Module):
    """Gaussian actor and value critic over continuous high-level actions."""

    def __init__(self, latent_dim: int, action_dim: int = 5, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(latent)
        value = self.critic(latent).squeeze(-1)
        return mean, value

    def act(
        self, latent: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(latent)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
