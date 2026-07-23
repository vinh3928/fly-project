"""Train NextStateWorldModel on synthetic latent transitions (Phase 7)."""

from __future__ import annotations

from pathlib import Path

import torch

from flybrain.world_model.predictor import NextStateWorldModel


def make_synthetic_transitions(
    *,
    batch_size: int,
    latent_dim: int,
    action_dim: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch_size, latent_dim, generator=g)
    action = torch.randn(batch_size, action_dim, generator=g)
    # Smooth dynamics: next ≈ latent + small action projection + noise
    proj = torch.randn(action_dim, latent_dim, generator=g) * 0.05
    noise = torch.randn(batch_size, latent_dim, generator=g) * 0.02
    next_latent = latent + action @ proj + noise
    return latent, action, next_latent


def train_world_model(
    *,
    steps: int = 200,
    batch_size: int = 64,
    latent_dim: int = 256,
    action_dim: int = 5,
    hidden_dim: int = 512,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, float]:
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = NextStateWorldModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Untrained baseline loss on a fixed eval batch
    eval_z, eval_a, eval_zn = make_synthetic_transitions(
        batch_size=batch_size, latent_dim=latent_dim, action_dim=action_dim, seed=seed + 999
    )
    eval_z, eval_a, eval_zn = eval_z.to(dev), eval_a.to(dev), eval_zn.to(dev)
    with torch.no_grad():
        baseline = float(model.prediction_loss(eval_z, eval_a, eval_zn).item())

    last_loss = baseline
    for step in range(steps):
        z, a, zn = make_synthetic_transitions(
            batch_size=batch_size,
            latent_dim=latent_dim,
            action_dim=action_dim,
            seed=seed + step,
        )
        z, a, zn = z.to(dev), a.to(dev), zn.to(dev)
        loss = model.prediction_loss(z, a, zn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        final_eval = float(model.prediction_loss(eval_z, eval_a, eval_zn).item())

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "loss": last_loss,
                "baseline_loss": baseline,
                "eval_loss": final_eval,
                "latent_dim": latent_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
            },
            path,
        )

    return {
        "loss": last_loss,
        "baseline_loss": baseline,
        "eval_loss": final_eval,
        "steps": float(steps),
        "improved": float(final_eval < baseline),
    }


def main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg_path = (
        Path(args.config)
        if args.config
        else Path(__file__).resolve().parents[2] / "configs" / "mind.yaml"
    )
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    wm = cfg.get("world_model") or {}
    steps = args.steps if args.steps is not None else 200
    ckpt = args.checkpoint or wm.get("checkpoint_path")
    metrics = train_world_model(
        steps=steps,
        latent_dim=int(wm.get("latent_dim", 256)),
        action_dim=int(wm.get("action_dim", 5)),
        hidden_dim=int(wm.get("hidden_dim", 512)),
        checkpoint_path=ckpt,
    )
    print("train_world_model_ok")
    print(metrics)
    assert metrics["improved"] >= 1.0 or metrics["eval_loss"] <= metrics["baseline_loss"]


if __name__ == "__main__":
    main()
