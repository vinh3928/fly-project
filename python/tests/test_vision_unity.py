"""Unity vision dataset → trained head → opening cue steering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flybrain.env_visual_nav import FakeVisualNavEnv
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, format_live_step_log
from flybrain.thinking.two_layer_future import OPENING_TURN_GAIN, cue_turn
from flybrain.training.train_vision_head import main as train_main
from flybrain.training.train_vision_head import train_vision_head_unity
from flybrain.vision import (
    CUE_NAMES,
    NUM_CUES,
    IncompatibleVisionCheckpoint,
    RecurrentVisionHead,
    heuristic_cues_from_frames,
    load_vision_head,
    save_vision_head,
)
from flybrain.vision.unity_dataset import (
    FRAMES_FILE,
    LABEL_COLUMNS,
    LABELS_FILE,
    META_FILE,
    load_unity_vision_dataset,
)

H = W = 8
COLUMNS = 4


def _write_dataset(root: Path, labels: np.ndarray, frames: np.ndarray, regions: list[str] | None = None) -> Path:
    """Mirror VisionDatasetWriter.cs: frames (N,2,H,W,3) uint8, labels (N,5) in LABEL_COLUMNS order."""
    root.mkdir(parents=True, exist_ok=True)
    n = labels.shape[0]
    frames.astype(np.uint8).tofile(root / FRAMES_FILE)
    header = ["region", *LABEL_COLUMNS, *[f"d{i}" for i in range(COLUMNS)]]
    rows = [",".join(header)]
    for i in range(n):
        region = (regions or ["Interior"] * n)[i]
        rows.append(",".join([region, *[f"{v:g}" for v in labels[i]], *["1"] * COLUMNS]))
    (root / LABELS_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / META_FILE).write_text(
        json.dumps({"count": n, "width": W, "height": H, "channels": 3, "columns": COLUMNS}), encoding="utf-8"
    )
    return root


def _learnable_dataset(root: Path, n: int = 96, seed: int = 0) -> Path:
    """Wall hazard = brightness of both eyes; opening = bright right half of the right eye."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((n, 5), dtype=np.float32)
    frames = np.zeros((n, 2, H, W, 3), dtype=np.uint8)
    for i in range(n):
        hazard = float(rng.random())
        frames[i] = int(hazard * 255)
        opening = i % 2 == 0
        if opening:
            frames[i, 1, :, W // 2:, 1] = 255
        labels[i] = [hazard, 1 - 2 * hazard, 0.0, 0.5 if opening else 0.0, 0.25 if opening else 0.0]
    return _write_dataset(root, labels, frames)


def test_load_maps_labels_and_keeps_top_first_frames(tmp_path: Path) -> None:
    frames = np.zeros((2, 2, H, W, 3), dtype=np.uint8)
    frames[0, 0, 0, :, 0] = 200  # left eye, top row, red
    labels = np.array([[0.3, 0.4, -0.2, 0.6, 0.5], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ds = load_unity_vision_dataset(_write_dataset(tmp_path, labels, frames, ["NearWall", "Outside"]))

    assert len(ds) == 2
    assert ds.regions == ["NearWall", "Outside"]
    assert ds.distances.shape == (2, COLUMNS)
    for j, name in enumerate(LABEL_COLUMNS):
        assert ds.labels[0, CUE_NAMES.index(name)] == pytest.approx(labels[0, j])
    assert ds.labels[0, CUE_NAMES.index("target_confidence")] == 0.0

    left, right, targets = ds.batch(np.array([0]))
    assert left.shape == (1, 3, H, W) and right.shape == (1, 3, H, W)
    assert left[0, 0, 0, 0] == pytest.approx(200 / 255)
    assert left[0, 0, -1, 0] == 0.0
    assert targets.shape == (1, NUM_CUES)


def test_mirror_swaps_eyes_flips_images_and_negates_bearings(tmp_path: Path) -> None:
    frames = np.zeros((1, 2, H, W, 3), dtype=np.uint8)
    frames[0, 1, :, 0, 2] = 255  # right eye, left column, blue
    labels = np.array([[0.3, 0.4, -0.2, 0.6, 0.5]], dtype=np.float32)
    ds = load_unity_vision_dataset(_write_dataset(tmp_path, labels, frames))

    left, right, targets = ds.batch(np.array([0]), rng=np.random.default_rng(0), mirror_prob=1.0, brightness_jitter=0.0)

    assert left[0, 2, :, -1].min() == pytest.approx(1.0)  # came from the right eye, now on the right edge
    assert right.max() == 0.0
    assert targets[0, CUE_NAMES.index("opening_bearing")] == pytest.approx(-0.6)
    assert targets[0, CUE_NAMES.index("turn_bias")] == pytest.approx(0.2)
    assert targets[0, CUE_NAMES.index("wall_hazard")] == pytest.approx(0.3)


def test_load_rejects_truncated_frames(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path, np.zeros((2, 5), np.float32), np.zeros((2, 2, H, W, 3), np.uint8))
    (root / FRAMES_FILE).write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError):
        load_unity_vision_dataset(root)


def test_head_outputs_opening_cues() -> None:
    head = RecurrentVisionHead(in_channels=3)
    cues, _ = head(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    assert cues.shape == (2, NUM_CUES)
    strength = cues[:, CUE_NAMES.index("opening_strength")]
    assert torch.all((strength >= 0) & (strength <= 1))
    d = head.infer_cues(np.random.rand(3, 64, 64), np.random.rand(3, 64, 64)).to_dict()
    assert {"opening_bearing", "opening_strength"} <= d.keys()


def test_heuristic_reports_no_opening() -> None:
    cues = heuristic_cues_from_frames(np.random.rand(3, 64, 64), np.random.rand(3, 64, 64))
    assert cues.opening_strength == 0.0 and cues.opening_bearing == 0.0


def test_save_load_round_trip_keeps_shape_and_weights(tmp_path: Path) -> None:
    head = RecurrentVisionHead(in_channels=3, stateless=True)
    path = save_vision_head(head, tmp_path / "vh.pt", loss=0.5)
    loaded = load_vision_head(path)
    assert loaded.in_channels == 3 and loaded.stateless
    x = np.random.rand(3, 64, 64).astype(np.float32)
    assert head.infer_cues(x, x).to_dict() == pytest.approx(loaded.infer_cues(x, x).to_dict())


def test_stateless_head_ignores_previous_frames() -> None:
    head = RecurrentVisionHead(in_channels=3, stateless=True)
    a = np.random.rand(3, 64, 64).astype(np.float32)
    b = np.random.rand(3, 64, 64).astype(np.float32)
    first = head.infer_cues(a, a).to_dict()
    head.infer_cues(b, b)
    assert head.infer_cues(a, a).to_dict() == pytest.approx(first)


def test_unity_training_learns_and_saves(tmp_path: Path) -> None:
    root = _learnable_dataset(tmp_path / "ds", n=160)
    ckpt = tmp_path / "vh.pt"
    metrics = train_vision_head_unity(
        dataset_dir=root, epochs=40, batch_size=32, learning_rate=3e-3, val_fraction=0.25, checkpoint_path=ckpt
    )
    assert ckpt.exists()
    assert metrics["samples"] == 160
    assert metrics["val_mae_wall_hazard"] < 0.1
    assert metrics["val_mae_opening_strength"] < 0.1
    assert metrics["val_mae_opening_bearing"] < 0.2
    assert 0.0 < metrics["val_sees_out_fraction"] < 1.0
    head = load_vision_head(ckpt)
    assert head.in_channels == 3 and head.stateless


def test_cli_trains_from_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _learnable_dataset(tmp_path / "ds", n=16)
    ckpt = tmp_path / "vh.pt"
    cfg = tmp_path / "vn.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "train": {"checkpoint_path": str(ckpt)},
                "unity_train": {"dataset_dir": str(root), "epochs": 2, "batch_size": 8},
            }
        ),
        encoding="utf-8",
    )
    train_main(["--config", str(cfg), "--unity"])
    assert ckpt.exists()
    out = capsys.readouterr().out
    assert "epoch 1/2 loss=" in out and "epoch 2/2 loss=" in out and "eta=" in out
    assert "vision_head_unity_train_ok" in out


def test_on_epoch_reports_each_epoch_with_validation(tmp_path: Path) -> None:
    root = _learnable_dataset(tmp_path / "ds", n=16)
    seen: list[tuple[int, dict[str, float]]] = []
    train_vision_head_unity(
        dataset_dir=root, epochs=3, batch_size=8, val_fraction=0.25, on_epoch=lambda e, m: seen.append((e, m))
    )
    assert [e for e, _ in seen] == [1, 2, 3]
    assert all("loss" in m and "mae_wall_hazard" in m for _, m in seen)


def test_cue_turn_steers_toward_opening() -> None:
    turn, source = cue_turn({"opening_strength": 0.3, "opening_bearing": 0.5, "energy": 1.0, "hunger": 0.0})
    assert source == "opening"
    assert turn == pytest.approx(OPENING_TURN_GAIN * 0.5)


def test_cue_turn_prefers_novelty_and_ignores_weak_openings() -> None:
    sated = {"energy": 1.0, "hunger": 0.0}
    _, source = cue_turn({**sated, "opening_strength": 0.3, "opening_bearing": 0.5, "visual_novelty": 0.5})
    assert source == "novelty"
    _, source = cue_turn({**sated, "opening_strength": 0.05, "opening_bearing": 0.5, "turn_bias": 0.3})
    assert source == "wall"


def test_log_line_shows_opening() -> None:
    line = format_live_step_log(
        3,
        np.zeros(5, dtype=np.float32),
        {"vision_cues": {"target_bearing": 0.0, "opening_strength": 0.4, "opening_bearing": 0.5}},
    )
    assert "open=0.40@+0.50" in line


def test_live_mind_loads_unity_checkpoint(tmp_path: Path) -> None:
    ckpt = save_vision_head(RecurrentVisionHead(in_channels=3, stateless=True), tmp_path / "vh.pt")
    pipeline = LiveMindPipeline(
        FakeVisualNavEnv(),
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=1,
            llm_mock=True,
            use_vision_head=True,
            vision_checkpoint=str(ckpt),
            fly_id="fly_vision_ckpt",
        ),
    )
    try:
        assert pipeline._vision_trained
        assert pipeline.vision_head.in_channels == 3
        cues = pipeline._vision_cues(np.random.rand(3, 64, 64), np.random.rand(3, 64, 64))
        assert cues is not None and "opening_strength" in cues.to_dict()
    finally:
        pipeline.close()


def test_live_mind_labels_place_outside_from_opening_cue() -> None:
    from flybrain.memory.place_graph import PlaceGraphConfig
    from flybrain.vision import VisionNavCues

    pipeline = LiveMindPipeline(
        FakeVisualNavEnv(),
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=1,
            llm_mock=True,
            fly_id="fly_outside_label",
            place_graph=PlaceGraphConfig(enabled=True, backend="in_memory", cache={"enabled": False}),
        ),
    )
    try:
        frame = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
        payload: dict = {}
        report = pipeline._apply_place_map(
            left_eye=frame,
            right_eye=frame[:, :, ::-1].copy(),
            signal_payload=payload,
            info={},
            cues=VisionNavCues(0.0, 0.0, 0.0, 1.0, 0.0, opening_bearing=0.1, opening_strength=0.9),
        )
        assert report is not None and report.place is not None
        assert payload["place_label"] == "outside"
        assert report.place.props["senses"]["opening_strength"]["value"] == pytest.approx(0.9)
    finally:
        pipeline.close()


def _save_legacy_five_cue_head(path: Path) -> Path:
    state = RecurrentVisionHead().state_dict()
    state["head.2.weight"] = state["head.2.weight"][:5].clone()
    state["head.2.bias"] = state["head.2.bias"][:5].clone()
    torch.save({"model": state, "latent_dim": 128, "hidden_dim": 128}, path)
    return path


def test_load_rejects_legacy_five_cue_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(IncompatibleVisionCheckpoint):
        load_vision_head(_save_legacy_five_cue_head(tmp_path / "old.pt"))


def test_load_rejects_different_cue_names(tmp_path: Path) -> None:
    path = tmp_path / "renamed.pt"
    torch.save({"model": RecurrentVisionHead().state_dict(), "cues": ["a"] * NUM_CUES}, path)
    with pytest.raises(IncompatibleVisionCheckpoint, match="cues"):
        load_vision_head(path)


@pytest.mark.parametrize(
    "make_ckpt",
    [
        _save_legacy_five_cue_head,
        lambda p: save_vision_head(RecurrentVisionHead(in_channels=1), p),
    ],
    ids=["legacy_five_cues", "grayscale_synthetic"],
)
def test_live_mind_skips_unusable_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_ckpt
) -> None:
    ckpt = make_ckpt(tmp_path / "vh.pt")
    pipeline = LiveMindPipeline(
        FakeVisualNavEnv(),
        config=LiveMindConfig(
            backend="in_memory",
            max_steps=1,
            llm_mock=True,
            use_vision_head=True,
            vision_checkpoint=str(ckpt),
            fly_id="fly_vision_skip",
        ),
    )
    try:
        assert not pipeline._vision_trained
        assert "vision_checkpoint_skipped" in capsys.readouterr().out
        frame = np.random.rand(3, 64, 64)
        assert pipeline._vision_cues(frame, frame) == heuristic_cues_from_frames(frame, frame)
    finally:
        pipeline.close()
