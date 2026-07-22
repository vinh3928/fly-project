from __future__ import annotations

import torch
from torch import nn

from .body import BodyEncoder
from .chemical import ChemicalEncoder
from .internal import InternalStateEncoder
from .vision import VisionEncoder


class MultimodalEncoder(nn.Module):
    """Fuses modality encoders into a single latent representation with optional GRU memory."""

    def __init__(
        self,
        body_dim: int,
        chemical_dim: int = 64,
        internal_dim: int = 8,
        vision_channels: int = 1,
        latent_dim: int = 256,
        use_memory: bool = True,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder(in_channels=vision_channels, latent_dim=128)
        self.body = BodyEncoder(input_dim=body_dim, latent_dim=128)
        self.chemical = ChemicalEncoder(input_dim=chemical_dim, latent_dim=64)
        self.internal = InternalStateEncoder(input_dim=internal_dim, latent_dim=32)
        fused_in = 128 + 128 + 64 + 32
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.use_memory = use_memory
        self.latent_dim = latent_dim
        self.memory = nn.GRU(latent_dim, latent_dim, batch_first=True) if use_memory else None

    def encode_step(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        body: torch.Tensor,
        chemical: torch.Tensor,
        internal: torch.Tensor,
    ) -> torch.Tensor:
        left = self.vision(left_eye)
        right = self.vision(right_eye)
        vision = 0.5 * (left + right)
        parts = torch.cat(
            [vision, self.body(body), self.chemical(chemical), self.internal(internal)],
            dim=-1,
        )
        return self.fuse(parts)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        body: torch.Tensor,
        chemical: torch.Tensor,
        internal: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent = self.encode_step(left_eye, right_eye, body, chemical, internal)
        if self.memory is None:
            return latent, None
        # Treat single-step as sequence length 1.
        seq = latent.unsqueeze(1)
        out, hidden_out = self.memory(seq, None if hidden is None else hidden.unsqueeze(0))
        return out.squeeze(1), hidden_out.squeeze(0)
