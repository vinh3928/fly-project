"""
Exchange a single observation/action with a fake environment.

Validates the Python side of the Unity↔Python loop before a Unity build exists.
"""

from __future__ import annotations

import numpy as np
import torch

from flybrain.env import FakeFlyEnv
from flybrain.policy import ActorCriticPolicy
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward


def main() -> None:
    env = FakeFlyEnv()
    obs, info = env.reset(seed=0)
    policy = ActorCriticPolicy(latent_dim=obs.shape[0], action_dim=env.action_space.shape[0])
    latent = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    action, _, _ = policy.act(latent, deterministic=True)
    action_np = action.squeeze(0).detach().cpu().numpy()
    next_obs, reward, terminated, truncated, step_info = env.step(action_np)

    total = compute_total_reward(
        curiosity=0.1,
        energy_improvement=float(next_obs[0] - obs[0]),
        hydration_improvement=0.0,
        damage=0.0,
        motor_effort=float(step_info["effort"]),
        instability=0.0,
        config=HomeostaticRewardConfig(),
    )

    print("smoke_exchange_ok")
    print(f"obs_shape={obs.shape} action_shape={action_np.shape}")
    print(f"env_reward={reward:.4f} homeostatic_reward={total:.4f}")
    print(f"terminated={terminated} truncated={truncated} seed={info['seed']}")
    assert obs.shape == next_obs.shape
    assert action_np.shape == (5,)
    assert np.isfinite(reward)


if __name__ == "__main__":
    main()
