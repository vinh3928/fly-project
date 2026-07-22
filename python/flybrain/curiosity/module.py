from __future__ import annotations

import torch
from torch import nn


class CuriosityModule(nn.Module):
    """Intrinsic reward from next-state prediction error, with capping and EMA learning progress."""

    def __init__(self, latent_dim: int, action_dim: int, reward_cap: float = 1.0) -> None:
        super().__init__()
        self.reward_cap = reward_cap
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.register_buffer("error_ema", torch.tensor(0.0))
        self.ema_decay = 0.99

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([latent, action], dim=-1))

    def intrinsic_reward(
        self, latent: torch.Tensor, action: torch.Tensor, next_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self.forward(latent, action)
        error = torch.mean((predicted - next_latent.detach()) ** 2, dim=-1)
        # Learning progress: positive when error falls relative to EMA.
        progress = torch.clamp(self.error_ema - error, min=0.0)
        reward = torch.clamp(0.5 * error + 0.5 * progress, max=self.reward_cap)
        with torch.no_grad():
            batch_mean = error.mean()
            self.error_ema.mul_(self.ema_decay).add_(batch_mean * (1.0 - self.ema_decay))
        loss = torch.mean(error)
        return reward, loss
