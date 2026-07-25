from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Convolutional encoder for low-resolution eye images (grayscale or RGB)."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=5, stride=2, padding=2),
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

    def _match_channels(self, images: torch.Tensor) -> torch.Tensor:
        """Map (B,H,W) / (B,1|3,H,W) onto ``self.in_channels`` (luma or RGB expand)."""
        if images.ndim == 3:
            images = images.unsqueeze(1)
        c = int(images.shape[1])
        want = self.in_channels
        if c == want:
            return images
        if c == 3 and want == 1:
            return images.mean(dim=1, keepdim=True)
        if c == 1 and want == 3:
            return images.repeat(1, 3, 1, 1)
        if c > want:
            return images[:, :want]
        pad = images[:, -1:].repeat(1, want - c, 1, 1)
        return torch.cat([images, pad], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W) or (B, H, W)
        return self.net(self._match_channels(images))
