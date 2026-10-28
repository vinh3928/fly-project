"""Generic ModalChangeGate: 10s sample → diff → L1 stay / open expensive path."""

from __future__ import annotations

import numpy as np

from flybrain.phase4 import PATCH_VEC_DIM, SMELL_VEC_DIM
from flybrain.sensors.modal_gate import (
    ModalChangeGate,
    ModalGateConfig,
    modal_delta,
    pad_modal_vec,
)


def _vec(seed: int, dim: int = 16, scale: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(dim) * scale).astype(np.float32)


def test_pad_and_delta_helpers() -> None:
    assert pad_modal_vec(None, 4).shape == (4,)
    assert pad_modal_vec(np.ones(2), 4).tolist() == [1.0, 1.0, 0.0, 0.0]
    a = np.asarray([0.0, 0.1, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    assert abs(modal_delta(a, b, mode="max_abs") - 0.1) < 1e-6
    assert modal_delta(a, a, mode="cosine_distance") < 1e-6


def test_ten_second_sample_stays_l1_when_unchanged() -> None:
    gate = ModalChangeGate(
        modality="smell",
        ttl_seconds=10.0,
        change_min=0.05,
        refresh_seconds=None,
        dim=8,
    )
    v = np.asarray([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    first = gate.observe(v, now_s=0.0)
    assert first.sampled and first.changed and first.active
    assert first.reason == "smell_change"
    assert first.l1_hit is False

    # Inside the 10s window — no resample.
    quiet = gate.observe(v, now_s=5.0)
    assert quiet.sampled is False
    assert quiet.active is False
    assert quiet.l1_hit is False

    # At 10s: same vector → L1 hit, expensive path stays closed.
    same = gate.observe(v, now_s=10.0)
    assert same.sampled is True
    assert same.changed is False
    assert same.l1_hit is True
    assert same.active is False
    assert same.reason == ""


def test_ten_second_sample_opens_on_change() -> None:
    gate = ModalChangeGate(modality="vision", ttl_seconds=10.0, change_min=0.05, dim=4)
    a = np.asarray([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    b = np.asarray([0.9, 0.2, 0.2, 0.2], dtype=np.float32)
    gate.observe(a, now_s=0.0)
    changed = gate.observe(b, now_s=10.0)
    assert changed.l1_hit is False
    assert changed.changed is True
    assert changed.active is True
    assert changed.reason == "vision_change"
    assert changed.delta >= 0.05


def test_force_open_without_resample() -> None:
    gate = ModalChangeGate(modality="smell", ttl_seconds=10.0, change_min=0.05, dim=4)
    v = np.ones(4, dtype=np.float32) * 0.4
    gate.observe(v, now_s=0.0)
    forced = gate.observe(v, now_s=1.0, force_open=True, force_reason="vision_novelty")
    assert forced.sampled is False
    assert forced.active is True
    assert forced.reason == "vision_novelty"


def test_refresh_timer_reopens_expensive_path() -> None:
    gate = ModalChangeGate(
        modality="sensors",
        ttl_seconds=10.0,
        change_min=0.05,
        refresh_seconds=30.0,
        dim=4,
    )
    v = np.ones(4, dtype=np.float32) * 0.3
    gate.observe(v, now_s=0.0)
    # Unchanged at 10s: L1 hit, still inside refresh window.
    mid = gate.observe(v, now_s=10.0)
    assert mid.l1_hit is True
    assert mid.active is False
    # Past refresh window: reopen expensive path even though sample matches.
    due = gate.observe(v, now_s=40.0)
    assert due.active is True
    assert due.reason == "timer"


def test_step_interval_without_ttl() -> None:
    gate = ModalChangeGate(
        modality="smell",
        ttl_seconds=0.0,
        step_interval=4,
        change_min=0.05,
        refresh_seconds=None,
        dim=4,
    )
    v = np.ones(4, dtype=np.float32) * 0.5
    flags = [gate.observe(v, step=s, now_s=float(s)).sampled for s in range(8)]
    assert flags == [True, False, False, False, True, False, False, False]


def test_slow_drift_accumulates_against_l1_baseline() -> None:
    gate = ModalChangeGate(
        modality="smell",
        ttl_seconds=0.0,
        step_interval=1,
        change_min=0.05,
        refresh_seconds=None,
        dim=4,
    )
    gate.observe(np.asarray([0.50, 0, 0, 0], dtype=np.float32), step=0, now_s=0.0)
    assert gate.observe(np.asarray([0.52, 0, 0, 0], dtype=np.float32), step=1, now_s=1.0).changed is False
    assert gate.observe(np.asarray([0.54, 0, 0, 0], dtype=np.float32), step=2, now_s=2.0).changed is False
    tripped = gate.observe(np.asarray([0.56, 0, 0, 0], dtype=np.float32), step=3, now_s=3.0)
    assert tripped.changed is True


def test_from_config_dict_defaults_to_ten_seconds() -> None:
    cfg = ModalGateConfig.from_dict({"modality": "vision"})
    assert cfg.ttl_seconds == 10.0
    assert cfg.modality == "vision"
    gate = ModalChangeGate.from_config(cfg)
    assert gate.ttl_seconds == 10.0


def test_reusable_across_modalities() -> None:
    """Same gate class for vision patches and smell vectors."""
    vision = ModalChangeGate(modality="vision", ttl_seconds=10.0, dim=PATCH_VEC_DIM, change_min=0.1)
    smell = ModalChangeGate(modality="smell", ttl_seconds=10.0, dim=SMELL_VEC_DIM, change_min=0.05)
    v = _vec(1, PATCH_VEC_DIM)
    s = _vec(2, SMELL_VEC_DIM)
    assert vision.observe(v, now_s=0.0).modality == "vision"
    assert smell.observe(s, now_s=0.0).modality == "smell"
    assert vision.observe(v, now_s=10.0).l1_hit is True
    assert smell.observe(s, now_s=10.0).l1_hit is True
