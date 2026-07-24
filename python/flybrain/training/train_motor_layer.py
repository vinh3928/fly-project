"""Train a small goal-conditioned motor policy on FakeLocomotionEnv (Phase 9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain.env_locomotion import FakeLocomotionConfig, FakeLocomotionEnv
from flybrain.motor.layer import HeuristicMotorTracker, MotorPolicyNet, clip_goal
from flybrain.phase9 import BODY_OBS_DIM, MOTOR_GOAL_DIM
from flybrain.seeding import seed_everything


def train_motor_policy(
    *,
    steps: int = 400,
    batch_episodes: int = 4,
    episode_len: int = 40,
    lr: float = 1e-3,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """Imitate the heuristic tracker (behavior cloning) with a light tracking loss."""
    seed_everything(seed)
    dev = torch.device(device)
    policy = MotorPolicyNet(obs_dim=BODY_OBS_DIM, goal_dim=MOTOR_GOAL_DIM).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = FakeLocomotionEnv(FakeLocomotionConfig(seed=seed, max_steps=episode_len))

    losses: list[float] = []
    track_errs: list[float] = []

    for step in range(steps):
        obs_batch = []
        goal_batch = []
        expert_batch = []
        for ep in range(batch_episodes):
            tracker = HeuristicMotorTracker(smoothing=0.4, max_delta=0.5)
            goal = clip_goal(
                np.asarray(
                    [
                        float(np.random.uniform(-1.0, 1.0)),
                        float(np.random.uniform(-1.0, 1.0)),
                        float(np.random.uniform(0.3, 0.7)),
                        float(np.random.uniform(0.3, 0.8)),
                        0.0,
                    ],
                    dtype=np.float32,
                )
            )
            obs, _ = env.reset(seed=seed + step * 17 + ep, options={"goal": goal})
            tracker.reset(goal)
            env.set_goal(goal)
            for _ in range(episode_len):
                expert = tracker.act(goal, observation=obs)
                obs_batch.append(obs[:BODY_OBS_DIM].copy())
                goal_batch.append(goal.copy())
                expert_batch.append(expert.copy())
                obs, _, term, trunc, info = env.step(expert)
                track_errs.append(float(info["tracking_error"]))
                if term or trunc:
                    break

        o = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=dev)
        g = torch.as_tensor(np.stack(goal_batch), dtype=torch.float32, device=dev)
        expert = torch.as_tensor(np.stack(expert_batch), dtype=torch.float32, device=dev)
        pred = policy(o, g)
        loss = nn.functional.mse_loss(pred, expert)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    final_loss = float(np.mean(losses[-20:])) if losses else 0.0
    mean_track = float(np.mean(track_errs[-200:])) if track_errs else 0.0

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": policy.state_dict(), "loss": final_loss}, path)

    return {"loss": final_loss, "expert_tracking_error": mean_track, "steps": float(steps)}


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/motor_policy.pt")
    args = parser.parse_args()
    metrics = train_motor_policy(
        steps=args.steps,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    print("motor_train_ok")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
