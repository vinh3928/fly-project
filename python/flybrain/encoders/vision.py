from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Convolutional encoder for low-resolution eye images (grayscale or RGB)."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W) or (B, H, W)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        return self.net(images)
