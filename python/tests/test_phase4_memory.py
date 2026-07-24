from __future__ import annotations

import os

import numpy as np
import pytest

from flybrain.affect import AffectEstimator
from flybrain.memory import InMemoryShortTermMemory, open_short_term_memory
from flybrain.memory.embedder import MindEmbedder
from flybrain.phase4 import (
    AFFECT_VEC_DIM,
    EYE_VEC_DIM,
    SENSORS_VEC_DIM,
    STATUS_VEC_DIM,
    TOTAL_EMBEDDING_DIM,
)
from flybrain.signals import SignalBuilder


def _eye(val: float) -> np.ndarray:
    v = np.zeros(EYE_VEC_DIM, dtype=np.float32)
    v[0] = val
    return v


def test_in_memory_stm_insert_and_similar() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    a = np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32)
    b[0] = 0.9
    b[1] = 0.1
    r1 = mem.insert_embedding(embedding=a, signals={"sugar": 0.1}, affect={"joy": 0.2}, step=1)
    r2 = mem.insert_embedding(embedding=b, signals={"sugar": 0.2}, affect={"joy": 0.3}, step=2)
    assert r1.id != r2.id
    neighbors = mem.similar(a, fly_id="fly_001", top_k=2)
    assert neighbors[0].id == r1.id
    recent = mem.recent(fly_id="fly_001", limit=1)
    assert recent[0].id == r2.id
    thought = mem.insert_thought(
        stm_id=r2.id,
        reflection="hist",
        devil_critique="counter",
        refined_thought="ok",
        intent={"forward": 0.1},
    )
    assert thought.id >= 1
    mem.close()


def test_in_memory_multi_vector_insert_and_similar_by() -> None:
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    left_a = _eye(1.0)
    left_b = _eye(0.2)
    status_a = np.zeros(STATUS_VEC_DIM, dtype=np.float32)
    status_a[0] = 1.0
    status_b = np.zeros(STATUS_VEC_DIM, dtype=np.float32)
    status_b[1] = 1.0
    emb_a = np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32)
    emb_b[1] = 1.0

    aff_a = np.zeros(AFFECT_VEC_DIM, dtype=np.float32)
    aff_a[0] = 1.0
    aff_b = np.zeros(AFFECT_VEC_DIM, dtype=np.float32)
    aff_b[1] = 1.0
    r1 = mem.insert_embedding(
        embedding=emb_a,
        signals={},
        affect={},
        step=1,
        left_eye_vec=left_a,
        right_eye_vec=_eye(0.5),
        sensors_vec=np.ones(SENSORS_VEC_DIM, dtype=np.float32),
        status_vec=status_a,
        affect_vec=aff_a,
    )
    r2 = mem.insert_embedding(
        embedding=emb_b,
        signals={},
        affect={},
        step=2,
        left_eye_vec=left_b,
        right_eye_vec=_eye(0.9),
        sensors_vec=np.zeros(SENSORS_VEC_DIM, dtype=np.float32),
        status_vec=status_b,
        affect_vec=aff_b,
    )

    assert r1.left_eye_vec.shape == (EYE_VEC_DIM,)
    assert r1.sensors_vec.shape == (SENSORS_VEC_DIM,)
    assert r1.status_vec.shape == (STATUS_VEC_DIM,)
    assert r1.affect_vec.shape == (AFFECT_VEC_DIM,)

    by_eye = mem.similar_by(left_a, modality="left_eye", fly_id="fly_001", top_k=2)
    assert by_eye[0].id == r1.id

    by_status = mem.similar_by(status_b, modality="status", fly_id="fly_001", top_k=2)
    assert by_status[0].id == r2.id

    by_aff = mem.similar_by(aff_b, modality="affect", fly_id="fly_001", top_k=1)
    assert by_aff[0].id == r2.id
    mem.close()


def test_as_f32_accepts_pgvector_like_tolist() -> None:
    """pgvector.Vector uses to_list/to_numpy — np.asarray(Vector) raises TypeError."""
    from flybrain.memory.pgvector import PgVectorShortTermMemory, _pad_vec

    class FakeVector:
        def __init__(self, data: list[float]) -> None:
            self._data = data

        def to_list(self) -> list[float]:
            return list(self._data)

        def to_numpy(self) -> np.ndarray:
            return np.asarray(self._data, dtype=np.float32)

    out = PgVectorShortTermMemory._as_f32(FakeVector([1.0, 2.0, 3.0]), dim=4)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert float(out[0]) == 1.0 and float(out[2]) == 3.0 and float(out[3]) == 0.0

    # to_list-only path (no to_numpy)
    class FakeVectorListOnly:
        def to_list(self) -> list[float]:
            return [0.5, 0.25]

    out2 = _pad_vec(FakeVectorListOnly(), dim=3)
    assert out2.shape == (3,) and float(out2[0]) == 0.5 and float(out2[2]) == 0.0


def test_open_short_term_memory_defaults_in_memory() -> None:
    mem = open_short_term_memory(backend="in_memory")
    assert isinstance(mem, InMemoryShortTermMemory)
    mem.close()


def test_mind_embedder_shape() -> None:
    builder = SignalBuilder()
    signals = builder.build(np.random.randn(197).astype(np.float32))
    affect = AffectEstimator().estimate(signals)
    emb = MindEmbedder().embed(signals, affect)
    assert emb.shape == (TOTAL_EMBEDDING_DIM,)
    assert np.isfinite(emb).all()


def test_mind_embedder_parts_shapes() -> None:
    builder = SignalBuilder()
    signals = builder.build(np.random.randn(197).astype(np.float32))
    affect = AffectEstimator().estimate(signals)
    parts = MindEmbedder().embed_parts(signals, affect)
    assert parts.embedding.shape == (TOTAL_EMBEDDING_DIM,)
    assert parts.left_eye_vec.shape == (EYE_VEC_DIM,)
    assert parts.right_eye_vec.shape == (EYE_VEC_DIM,)
    assert parts.sensors_vec.shape == (SENSORS_VEC_DIM,)
    assert parts.status_vec.shape == (STATUS_VEC_DIM,)
    assert parts.affect_vec.shape == (AFFECT_VEC_DIM,)
    assert np.allclose(parts.embedding[-AFFECT_VEC_DIM:], parts.affect_vec)
    assert np.isfinite(parts.embedding).all()


@pytest.mark.skipif(
    os.environ.get("FLY_MIND_PGVECTOR_TEST", "").lower() not in {"1", "true", "yes"},
    reason="Set FLY_MIND_PGVECTOR_TEST=1 with Postgres+pgvector available",
)
def test_pgvector_multi_vector_roundtrip() -> None:
    from flybrain.memory.pgvector import PgVectorShortTermMemory

    dsn = os.environ.get("FLY_MIND_DSN", "postgresql://fly:fly@127.0.0.1:5433/fly_mind")
    mem = PgVectorShortTermMemory(dsn)
    left = _eye(0.77)
    status = np.zeros(STATUS_VEC_DIM, dtype=np.float32)
    status[3] = 1.0
    emb = np.zeros(TOTAL_EMBEDDING_DIM, dtype=np.float32)
    emb[5] = 1.0
    rec = mem.insert_embedding(
        embedding=emb,
        signals={"sugar": 0.4},
        affect={"joy": 0.5},
        episode_id="test_mv",
        step=99,
        fly_id="fly_mv_test",
        left_eye_vec=left,
        right_eye_vec=_eye(0.11),
        sensors_vec=np.linspace(0, 1, SENSORS_VEC_DIM, dtype=np.float32),
        status_vec=status,
        affect_vec=np.linspace(0, 1, AFFECT_VEC_DIM, dtype=np.float32),
    )
    assert rec.id >= 1
    neighbors = mem.similar(emb, fly_id="fly_mv_test", top_k=1)
    assert neighbors[0].id == rec.id
    assert neighbors[0].left_eye_vec.shape == (EYE_VEC_DIM,)
    by_eye = mem.similar_by(left, modality="left_eye", fly_id="fly_mv_test", top_k=1)
    assert by_eye[0].id == rec.id
    by_status = mem.similar_by(status, modality="status", fly_id="fly_mv_test", top_k=1)
    assert by_status[0].id == rec.id
    mem.close()
