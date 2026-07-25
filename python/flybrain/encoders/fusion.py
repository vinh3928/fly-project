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
        vision_channels: int = 3,
        latent_dim: int = 256,
        use_memory: bool = True,
    ) -> None:
        super().__init__()
        self.vision_channels = int(vision_channels)
        self.vision = VisionEncoder(in_channels=self.vision_channels, latent_dim=128)
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

    def _prepare_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Accept (B,H,W), (B,1,H,W), or (B,3,H,W) and match ``vision_channels``."""
        if images.ndim == 3:
            images = images.unsqueeze(1)
        c = int(images.shape[1])
        want = self.vision_channels
        if c == want:
            return images
        if c == 1 and want == 3:
            return images.repeat(1, 3, 1, 1)
        if c == 3 and want == 1:
            return images.mean(dim=1, keepdim=True)
        if c > want:
            return images[:, :want]
        # Pad missing channels by repeating the last.
        pad = images[:, -1:].repeat(1, want - c, 1, 1)
        return torch.cat([images, pad], dim=1)

    def encode_step(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        body: torch.Tensor,
        chemical: torch.Tensor,
        internal: torch.Tensor,
    ) -> torch.Tensor:
        parts = self.encode_parts(left_eye, right_eye, body, chemical, internal)
        return parts["latent"]

    def encode_parts(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        body: torch.Tensor,
        chemical: torch.Tensor,
        internal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-modality encodings plus the fused latent."""
        left = self.vision(self._prepare_vision(left_eye))
        right = self.vision(self._prepare_vision(right_eye))
        vision = 0.5 * (left + right)
        body_z = self.body(body)
        chem_z = self.chemical(chemical)
        status_z = self.internal(internal)
        sensors = torch.cat([body_z, chem_z], dim=-1)
        fused_in = torch.cat([vision, body_z, chem_z, status_z], dim=-1)
        latent = self.fuse(fused_in)
        return {
            "left_eye": left,
            "right_eye": right,
            "sensors": sensors,
            "status": status_z,
            "latent": latent,
        }

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
