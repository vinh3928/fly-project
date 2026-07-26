"""Hazard conditioning + Present rewrite helpers."""

from __future__ import annotations

from flybrain.thinking.conditioning import (
    ConditioningConfig,
    HazardConditioning,
    rewrite_present_after_judgement,
)
from flybrain.thinking.two_layer_future import _stimulus_peaks, format_present_line


def test_repeated_observe_marks_conditioned_memory_block() -> None:
    store = HazardConditioning(ConditioningConfig(hit_threshold=3))
    for _ in range(3):
        store.observe("odor", confidence=0.8, reason="plume", signals={"odor": 0.9})
    assert store.is_conditioned("odor")
    block = store.memory_block()
    assert "odor×3" in block
    assert "conditioned" in block
    sig = store.to_signals()
    assert sig.get("conditioning_top_risk") == "odor"
    assert float(sig.get("conditioning_odor", 0.0)) >= 0.8


def test_acknowledge_omits_from_memory_until_judgement_cycle() -> None:
    store = HazardConditioning(ConditioningConfig(hit_threshold=2))
    store.observe("wall", confidence=0.7, reason="ahead")
    store.observe("wall", confidence=0.7, reason="ahead")
    store.acknowledge("wall")
    assert "wall" not in store.memory_block()
    assert store.is_acknowledged("wall")
    # Still listed when explicitly requested.
    assert "wall" in store.memory_block(include_acknowledged=True)
    store.begin_judgement_cycle()
    assert "wall" in store.memory_block()
    assert not store.is_acknowledged("wall")


def test_confirmation_streak_early_unlock_accept_only() -> None:
    store = HazardConditioning(ConditioningConfig(judgement_confirmations=2))
    assert store.record_judgement("accept", "odor") == 1
    assert not store.confirmed()
    assert store.record_judgement("accept", "odor") == 2
    assert store.confirmed()
    store.reset_confirmation()
    store.record_judgement("reject", "odor")
    store.record_judgement("reject", "odor")
    assert not store.confirmed()  # reject cannot early-unlock
    store.record_judgement("acknowledge", "wall")
    store.record_judgement("acknowledge", "wall")
    assert store.confirmed()


def test_mismatch_resets_confirmation_streak() -> None:
    store = HazardConditioning(ConditioningConfig(judgement_confirmations=2))
    store.record_judgement("accept", "odor")
    assert store.record_judgement("accept", "wall") == 1
    assert not store.confirmed()
    assert store.record_judgement("accept", "wall") == 2
    assert store.confirmed()


def test_present_peaks_include_odor() -> None:
    peaks = _stimulus_peaks({"odor": 0.5, "bitter": 0.0, "energy": 1.0}, {"survival_risk": 0.1})
    assert "odor" in peaks
    line = format_present_line(
        {"odor": 3.72, "bitter": 0.0, "sugar": 0.0, "energy": 1.0},
        {"survival_risk": 0.2},
        maturity=0.1,
        judgement={"decision": "acknowledge", "reason": "scores_acknowledge", "proposal": {"risk": "odor"}},
        conditioning_summary="CONDITIONING: odor×3 strength=0.80 (conditioned)",
    )
    assert "odor=3.72" in line
    assert "Judged: acknowledge odor" in line
    assert "CONDITIONING:" in line


def test_rewrite_present_after_judgement_keeps_future() -> None:
    refined = (
        "Present: energy=1.00, sugar=0.00, bitter=0.00, survival_risk=0.18, "
        "stimuli=['wall'], maturity=0.00. Future hypothesis (L1+L2): act=stop. "
        "Refined after critique: forward=0.07 turn=-0.14."
    )
    out = rewrite_present_after_judgement(
        refined,
        signals={"odor": 3.7, "bitter": 0.0, "sugar": 0.0, "energy": 1.0},
        affect={"survival_risk": 0.2},
        maturity=0.0,
        judgement={"decision": "accept", "reason": "scores_accept", "proposal": {"risk": "odor"}},
        conditioning_summary="CONDITIONING: odor×2 strength=0.70 (emerging)",
    )
    assert out.startswith("Present:")
    assert "odor=3.70" in out
    assert "Judged: accept odor" in out
    assert "Future hypothesis" in out
    assert "Refined after critique" in out


def test_load_from_dict_roundtrip() -> None:
    store = HazardConditioning(ConditioningConfig(hit_threshold=2))
    store.observe("odor", confidence=0.8, reason="plume")
    store.observe("odor", confidence=0.9, reason="plume")
    store.acknowledge("odor")
    snap = store.to_dict()
    restored = HazardConditioning(ConditioningConfig(hit_threshold=2))
    assert restored.load_from_dict(snap)
    assert restored.is_conditioned("odor")
    assert restored.is_acknowledged("odor")
    assert restored._entries["odor"].hits == 2


def test_conditioning_persisted_and_hydrated_via_stm() -> None:
    from flybrain.env import FakeFlyEnv
    from flybrain.memory.pgvector import InMemoryShortTermMemory
    from flybrain.phase4 import TOTAL_EMBEDDING_DIM
    from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline
    from flybrain.thinking.stimulus_interrupt import StimulusInterruptConfig

    env = FakeFlyEnv()
    mem = InMemoryShortTermMemory(dim=TOTAL_EMBEDDING_DIM)
    pipe = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            fly_id="fly_cond_stm",
            stimulus_interrupt=StimulusInterruptConfig(enabled=True),
            console_log=False,
        ),
        memory=mem,
    )
    pipe.conditioning.observe("wall", confidence=0.7, reason="ahead")
    pipe.conditioning.observe("wall", confidence=0.8, reason="ahead")
    pipe.conditioning.acknowledge("wall")
    rec = pipe._persist_conditioning_to_stm()
    assert rec is not None
    assert rec.signals.get("conditioning_checkpoint") is True
    assert "wall" in (rec.signals.get("conditioning") or {}).get("entries", {})

    # Fresh pipeline sharing the same STM hydrates entries + acknowledge flags.
    pipe2 = LiveMindPipeline(
        env,
        config=LiveMindConfig(
            backend="in_memory",
            llm_mock=True,
            fly_id="fly_cond_stm",
            stimulus_interrupt=StimulusInterruptConfig(enabled=True),
            console_log=False,
        ),
        memory=mem,
    )
    assert not pipe2.conditioning._entries
    assert pipe2._hydrate_conditioning_from_stm() is True
    assert pipe2.conditioning._entries["wall"].hits >= 2
    assert pipe2.conditioning.is_acknowledged("wall")
    pipe.close()
    pipe2.close()


def test_extract_from_signals_helper() -> None:
    from flybrain.thinking.conditioning import HazardConditioning

    assert HazardConditioning.extract_from_signals({}) is None
    assert HazardConditioning.extract_from_signals({"conditioning": {"entries": {}}}) is None
    blob = {"entries": {"odor": {"hits": 1, "strength": 0.5}}}
    assert HazardConditioning.extract_from_signals({"conditioning": blob}) == blob
