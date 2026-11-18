"""Offline supervised training for RecurrentVisionHead.

Two data sources:
  * synthetic stereo frames (Phase 5 bars / bright target), grayscale;
  * Unity eye frames with ray-cast labels (``--unity``), RGB, incl. the opening cues.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain.vision.recurrent_head import CUE_NAMES, NUM_CUES, RecurrentVisionHead, save_vision_head
from flybrain.vision.synthetic import make_synthetic_batch
from flybrain.vision.unity_dataset import load_unity_vision_dataset

PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PYTHON_ROOT / "configs" / "visual_nav.yaml"
_SYNTHETIC_KEYS = ("target_bearing", "target_confidence", "wall_hazard", "forward_bias", "turn_bias")


def resolve_path(path: str | Path) -> Path:
    """Config paths are relative to the ``python/`` folder (same rule as the live mind)."""
    p = Path(path)
    return p if p.is_absolute() else (PYTHON_ROOT / p).resolve()


def train_vision_head(
    *,
    steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, float]:
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = RecurrentVisionHead().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    last_loss = 0.0

    for step in range(steps):
        batch = make_synthetic_batch(batch_size, seed=seed + step)
        left = torch.as_tensor(batch["left"], device=dev)
        right = torch.as_tensor(batch["right"], device=dev)
        target = torch.zeros(batch_size, NUM_CUES, device=dev)
        for key in _SYNTHETIC_KEYS:
            target[:, CUE_NAMES.index(key)] = torch.as_tensor(batch[key], device=dev)
        model.reset_state(batch_size=batch_size, device=dev)
        pred, _ = model(left, right, model._hidden)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    if checkpoint_path is not None:
        save_vision_head(model, checkpoint_path, loss=last_loss)

    return {"loss": last_loss, "steps": float(steps)}


def _masked_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over all cues; opening bearing only counts where some of the view sees out."""
    err = (pred - target) ** 2
    weight = torch.ones_like(err)
    strength = target[:, CUE_NAMES.index("opening_strength")]
    weight[:, CUE_NAMES.index("opening_bearing")] = (strength > 0).float()
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_vision_head(model: RecurrentVisionHead, dataset, indices: np.ndarray, *, device: str = "cpu") -> dict[str, float]:
    """Mean absolute error per cue (opening bearing only on samples that see out)."""
    model.eval()
    dev = torch.device(device)
    preds, targets = [], []
    for start in range(0, len(indices), 256):
        idx = indices[start:start + 256]
        left, right, target = dataset.batch(idx)
        model.reset_state(batch_size=len(idx), device=dev)
        pred, _ = model(torch.as_tensor(left, device=dev), torch.as_tensor(right, device=dev), model._hidden)
        preds.append(pred.cpu().numpy())
        targets.append(target)
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    err = np.abs(pred - target)
    out = {f"mae_{name}": float(err[:, i].mean()) for i, name in enumerate(CUE_NAMES)}
    sees_out = target[:, CUE_NAMES.index("opening_strength")] > 0
    bearing = CUE_NAMES.index("opening_bearing")
    out["mae_opening_bearing"] = float(err[sees_out, bearing].mean()) if sees_out.any() else 0.0
    out["sees_out_fraction"] = float(sees_out.mean())
    return out


def train_vision_head_unity(
    *,
    dataset_dir: str | Path,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.1,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Train an RGB, stateless head on Unity frames; returns final train loss + validation MAEs.

    ``on_epoch(epoch, metrics)`` gets the epoch's mean train loss and validation MAEs.
    """
    dataset = load_unity_vision_dataset(dataset_dir)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(dataset))
    n_val = int(round(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    val_idx, train_idx = order[:n_val], order[n_val:]

    dev = torch.device(device)
    model = RecurrentVisionHead(in_channels=3, stateless=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_loss = 0.0
    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_idx)
        epoch_losses: list[float] = []
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start:start + batch_size]
            left, right, target = dataset.batch(idx, rng=rng)
            model.reset_state(batch_size=len(idx), device=dev)
            pred, _ = model(
                torch.as_tensor(left, device=dev), torch.as_tensor(right, device=dev), model._hidden
            )
            loss = _masked_loss(pred, torch.as_tensor(target, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            epoch_losses.append(last_loss)
        if on_epoch is not None:
            report = {"loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0}
            if n_val > 0:
                report.update(evaluate_vision_head(model, dataset, val_idx, device=device))
            on_epoch(epoch + 1, report)

    metrics: dict[str, float] = {"loss": last_loss, "epochs": float(epochs), "samples": float(len(dataset))}
    if n_val > 0:
        metrics.update({f"val_{k}": v for k, v in evaluate_vision_head(model, dataset, val_idx, device=device).items()})
    if checkpoint_path is not None:
        save_vision_head(model, checkpoint_path, **metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None, help="synthetic training steps")
    parser.add_argument(
        "--unity",
        nargs="?",
        const="",
        default=None,
        metavar="DATASET_DIR",
        help="train on a Unity dataset (default: unity_train.dataset_dir from the config)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    train_cfg = cfg.get("train", {})
    checkpoint = train_cfg.get("checkpoint_path")
    checkpoint_path = resolve_path(checkpoint) if checkpoint else None

    if args.unity is not None:
        unity_cfg = cfg.get("unity_train", {})
        dataset_dir = resolve_path(args.unity or unity_cfg.get("dataset_dir", "../data/vision_unity"))
        epochs = args.epochs if args.epochs is not None else int(unity_cfg.get("epochs", 15))
        started = time.perf_counter()
        print(f"vision_head_unity_train dataset={dataset_dir} epochs={epochs}", flush=True)

        def report_epoch(epoch: int, m: dict[str, float]) -> None:
            elapsed = time.perf_counter() - started
            eta = elapsed / epoch * (epochs - epoch)
            print(
                f"epoch {epoch}/{epochs} loss={m['loss']:.4f} "
                f"val_wall={m.get('mae_wall_hazard', 0.0):.3f} "
                f"val_open={m.get('mae_opening_strength', 0.0):.3f} "
                f"val_open_bear={m.get('mae_opening_bearing', 0.0):.3f} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

        metrics = train_vision_head_unity(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=int(unity_cfg.get("batch_size", 64)),
            learning_rate=float(unity_cfg.get("learning_rate", 1e-3)),
            val_fraction=float(unity_cfg.get("val_fraction", 0.1)),
            device=str(train_cfg.get("device", "cpu")),
            checkpoint_path=checkpoint_path,
            seed=int(cfg.get("seed", 0)),
            on_epoch=report_epoch,
        )
        print("vision_head_unity_train_ok", {k: round(v, 4) for k, v in metrics.items()})
        print("checkpoint", checkpoint_path)
        return

    steps = args.steps if args.steps is not None else int(train_cfg.get("steps", 200))
    metrics = train_vision_head(
        steps=steps,
        batch_size=int(train_cfg.get("batch_size", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        device=str(train_cfg.get("device", "cpu")),
        checkpoint_path=checkpoint_path,
        seed=int(cfg.get("seed", 0)),
    )
    print("vision_head_train_ok", metrics)


if __name__ == "__main__":
    main()
