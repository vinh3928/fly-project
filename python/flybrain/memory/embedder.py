"""Embed sensory + affect into a fixed STM vector."""

from __future__ import annotations

import numpy as np
import torch

from flybrain.affect.estimator import AffectState
from flybrain.encoders.fusion import MultimodalEncoder
from flybrain.phase4 import AFFECT_DIM, EMBEDDING_LATENT_DIM, TOTAL_EMBEDDING_DIM
from flybrain.signals.builder import SensorySignals


class MindEmbedder:
    """MultimodalEncoder latent concatenated with affect factor vector."""

    def __init__(
        self,
        *,
        body_dim: int = 85,
        chemical_dim: int = 64,
        internal_dim: int = 8,
        latent_dim: int = EMBEDDING_LATENT_DIM,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.encoder = MultimodalEncoder(
            body_dim=body_dim,
            chemical_dim=chemical_dim,
            internal_dim=internal_dim,
            latent_dim=latent_dim,
            use_memory=False,
        ).to(self.device)
        self.encoder.eval()
        self.latent_dim = latent_dim
        self.total_dim = TOTAL_EMBEDDING_DIM

    @torch.no_grad()
    def embed(self, signals: SensorySignals, affect: AffectState) -> np.ndarray:
        left = torch.as_tensor(signals.left_eye[None, ...], device=self.device)
        right = torch.as_tensor(signals.right_eye[None, ...], device=self.device)
        body = torch.as_tensor(signals.body[None, ...], device=self.device)
        chemical = torch.as_tensor(signals.chemical[None, ...], device=self.device)
        internal = torch.as_tensor(signals.internal[None, ...], device=self.device)
        latent = self.encoder.encode_step(left, right, body, chemical, internal)
        latent_np = latent.squeeze(0).cpu().numpy().astype(np.float32)
        affect_vec = affect.as_vector()
        assert affect_vec.shape[0] == AFFECT_DIM
        out = np.concatenate([latent_np, affect_vec]).astype(np.float32)
        assert out.shape[0] == self.total_dim
        return out
