"""Embed sensory + affect into a fixed STM vector (and per-modality parts)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flybrain.affect.estimator import AffectState
from flybrain.encoders.fusion import MultimodalEncoder
from flybrain.phase4 import (
    AFFECT_DIM,
    AFFECT_VEC_DIM,
    EMBEDDING_LATENT_DIM,
    EYE_VEC_DIM,
    SENSORS_VEC_DIM,
    STATUS_VEC_DIM,
    TOTAL_EMBEDDING_DIM,
)
from flybrain.signals.builder import SensorySignals


@dataclass
class EmbeddingParts:
    """Fused STM embedding plus per-modality vectors for multi-vector storage."""

    embedding: np.ndarray
    left_eye_vec: np.ndarray
    right_eye_vec: np.ndarray
    sensors_vec: np.ndarray
    status_vec: np.ndarray
    affect_vec: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "embedding": self.embedding,
            "left_eye_vec": self.left_eye_vec,
            "right_eye_vec": self.right_eye_vec,
            "sensors_vec": self.sensors_vec,
            "status_vec": self.status_vec,
            "affect_vec": self.affect_vec,
        }


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
            vision_channels=3,
            latent_dim=latent_dim,
            use_memory=False,
        ).to(self.device)
        self.encoder.eval()
        self.latent_dim = latent_dim
        self.total_dim = TOTAL_EMBEDDING_DIM

    @torch.no_grad()
    def embed_parts(self, signals: SensorySignals, affect: AffectState) -> EmbeddingParts:
        left = torch.as_tensor(signals.left_eye[None, ...], device=self.device)
        right = torch.as_tensor(signals.right_eye[None, ...], device=self.device)
        body = torch.as_tensor(signals.body[None, ...], device=self.device)
        chemical = torch.as_tensor(signals.chemical[None, ...], device=self.device)
        internal = torch.as_tensor(signals.internal[None, ...], device=self.device)
        parts = self.encoder.encode_parts(left, right, body, chemical, internal)

        def _np(key: str) -> np.ndarray:
            return parts[key].squeeze(0).detach().cpu().numpy().astype(np.float32)

        left_v = _np("left_eye")
        right_v = _np("right_eye")
        sensors_v = _np("sensors")
        status_v = _np("status")
        latent_np = _np("latent")
        affect_vec = affect.as_vector().astype(np.float32)
        assert left_v.shape[0] == EYE_VEC_DIM
        assert right_v.shape[0] == EYE_VEC_DIM
        assert sensors_v.shape[0] == SENSORS_VEC_DIM
        assert status_v.shape[0] == STATUS_VEC_DIM
        assert affect_vec.shape[0] == AFFECT_VEC_DIM == AFFECT_DIM
        embedding = np.concatenate([latent_np, affect_vec]).astype(np.float32)
        assert embedding.shape[0] == self.total_dim
        return EmbeddingParts(
            embedding=embedding,
            left_eye_vec=left_v,
            right_eye_vec=right_v,
            sensors_vec=sensors_v,
            status_vec=status_v,
            affect_vec=affect_vec,
        )

    @torch.no_grad()
    def embed(self, signals: SensorySignals, affect: AffectState) -> np.ndarray:
        return self.embed_parts(signals, affect).embedding
