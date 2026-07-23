"""Offline supervised training for RecurrentVisionHead on synthetic stereo frames."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flybrain.vision.recurrent_head import RecurrentVisionHead
from flybrain.vision.synthetic import make_synthetic_batch


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
        target = torch.stack(
            [
                torch.as_tensor(batch["target_bearing"], device=dev),
                torch.as_tensor(batch["target_confidence"], device=dev),
                torch.as_tensor(batch["wall_hazard"], device=dev),
                torch.as_tensor(batch["forward_bias"], device=dev),
                torch.as_tensor(batch["turn_bias"], device=dev),
            ],
            dim=-1,
        )
        model.reset_state(batch_size=batch_size, device=dev)
        pred, _ = model(left, right, model._hidden)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "loss": last_loss}, path)

    return {"loss": last_loss, "steps": float(steps)}


def main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else Path(__file__).resolve().parents[2] / "configs" / "visual_nav.yaml"
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    train_cfg = cfg.get("train", {})
    steps = args.steps if args.steps is not None else int(train_cfg.get("steps", 200))
    metrics = train_vision_head(
        steps=steps,
        batch_size=int(train_cfg.get("batch_size", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        device=str(train_cfg.get("device", "cpu")),
        checkpoint_path=train_cfg.get("checkpoint_path"),
        seed=int(cfg.get("seed", 0)),
    )
    print("vision_head_train_ok", metrics)


if __name__ == "__main__":
    main()
