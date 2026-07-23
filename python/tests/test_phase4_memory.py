from __future__ import annotations

import numpy as np

from flybrain.affect import AffectEstimator
from flybrain.memory import InMemoryShortTermMemory, open_short_term_memory
from flybrain.memory.embedder import MindEmbedder
from flybrain.phase4 import TOTAL_EMBEDDING_DIM
from flybrain.signals import SignalBuilder


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
