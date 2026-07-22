from __future__ import annotations

import numpy as np
import pytest
import torch

from flybrain.curiosity import CuriosityModule
from flybrain.encoders import (
    BodyEncoder,
    ChemicalEncoder,
    InternalStateEncoder,
    MultimodalEncoder,
)
from flybrain.env import FakeFlyConfig, FakeFlyEnv, make_env
from flybrain.policy import ActorCriticPolicy
from flybrain.replay import ReplayBuffer, Transition
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward
from flybrain.training.train import TrainConfig, Trainer
from flybrain.unity_env import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM, default_build_path, repo_root
from flybrain.world_model import NextStateWorldModel


def _transition(obs_dim: int = 8, reward: float = 0.0, done: bool = False) -> Transition:
    return Transition(
        observation=np.zeros(obs_dim, dtype=np.float32),
        action=np.zeros(5, dtype=np.float32),
        reward=reward,
        next_observation=np.ones(obs_dim, dtype=np.float32),
        done=done,
    )


def test_obs_action_dim_constants_match_fake_env() -> None:
    assert DEFAULT_OBS_DIM == 197
    assert DEFAULT_ACTION_DIM == 5
    env = FakeFlyEnv()
    assert env.observation_space.shape == (DEFAULT_OBS_DIM,)
    assert env.action_space.shape == (DEFAULT_ACTION_DIM,)


def test_repo_root_contains_unity_and_python() -> None:
    root = repo_root()
    assert (root / "unity").is_dir()
    assert (root / "python").is_dir()


def test_make_env_returns_fake_by_default() -> None:
    env = make_env(prefer_unity=False)
    assert isinstance(env, FakeFlyEnv)


def test_make_env_prefer_unity_without_build_raises() -> None:
    if default_build_path() is not None:
        pytest.skip("Unity build present; cannot assert missing-build error")
    with pytest.raises(FileNotFoundError):
        make_env(prefer_unity=True)


def test_fake_env_truncates_at_max_steps() -> None:
    env = FakeFlyEnv(FakeFlyConfig(max_steps=3, seed=0))
    env.reset(seed=0)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(np.zeros(5, dtype=np.float32))
        assert not terminated
    assert truncated


def test_fake_env_energy_decays_with_effort() -> None:
    env = FakeFlyEnv(FakeFlyConfig(max_steps=50, seed=0))
    obs, _ = env.reset(seed=0)
    idle = np.zeros(5, dtype=np.float32)
    hard = np.ones(5, dtype=np.float32)
    _, _, _, _, idle_info = env.step(idle)
    obs2, _, _, _, hard_info = env.step(hard)
    assert hard_info["effort"] > idle_info["effort"]
    assert float(obs2[0]) < float(obs[0])


def test_replay_buffer_overwrites_at_capacity() -> None:
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add(
            Transition(
                observation=np.array([float(i)], dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=float(i),
                next_observation=np.array([float(i + 1)], dtype=np.float32),
                done=False,
            )
        )
    assert len(buf) == 3
    values = sorted(float(t.observation[0]) for t in buf.sample(3))
    assert values == [2.0, 3.0, 4.0]


def test_replay_buffer_sample_rejects_oversized_batch() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.add(_transition())
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(2)


def test_replay_sample_tensors_includes_done() -> None:
    buf = ReplayBuffer(capacity=10)
    buf.add(_transition(done=True))
    buf.add(_transition(done=False))
    batch = buf.sample_tensors(2)
    assert batch["done"].shape == (2,)
    assert set(batch["done"].tolist()) <= {0.0, 1.0}


def test_policy_deterministic_act_is_stable() -> None:
    torch.manual_seed(0)
    policy = ActorCriticPolicy(latent_dim=16, action_dim=5)
    latent = torch.randn(2, 16)
    a1, _, v1 = policy.act(latent, deterministic=True)
    a2, _, v2 = policy.act(latent, deterministic=True)
    assert torch.allclose(a1, a2)
    assert torch.allclose(v1, v2)
    mean, value = policy.forward(latent)
    assert mean.shape == (2, 5)
    assert value.shape == (2,)


def test_modality_encoder_shapes() -> None:
    body = BodyEncoder(input_dim=32, latent_dim=16)
    chemical = ChemicalEncoder(input_dim=8, latent_dim=8)
    internal = InternalStateEncoder(input_dim=8, latent_dim=4)
    assert body(torch.randn(3, 32)).shape == (3, 16)
    assert chemical(torch.randn(3, 8)).shape == (3, 8)
    assert internal(torch.randn(3, 8)).shape == (3, 4)


def test_multimodal_without_memory() -> None:
    enc = MultimodalEncoder(
        body_dim=32, chemical_dim=8, internal_dim=8, latent_dim=64, use_memory=False
    )
    latent, hidden = enc(
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 32),
        torch.randn(2, 8),
        torch.randn(2, 8),
    )
    assert latent.shape == (2, 64)
    assert hidden is None


def test_world_model_forward_shape() -> None:
    model = NextStateWorldModel(latent_dim=16, action_dim=5)
    pred = model(torch.randn(4, 16), torch.randn(4, 5))
    assert pred.shape == (4, 16)


def test_curiosity_cap_boundary() -> None:
    module = CuriosityModule(latent_dim=8, action_dim=3, reward_cap=0.25)
    # Force a large prediction error by contrasting unrelated latents.
    reward, loss = module.intrinsic_reward(
        torch.zeros(4, 8),
        torch.zeros(4, 3),
        torch.ones(4, 8) * 10,
    )
    assert float(reward.max().detach()) <= 0.25 + 1e-5
    assert torch.isfinite(loss)


def test_homeostatic_reward_clamps_curiosity() -> None:
    low = compute_total_reward(
        curiosity=-5.0,
        energy_improvement=0.0,
        hydration_improvement=0.0,
        damage=0.0,
        motor_effort=0.0,
        instability=0.0,
        config=HomeostaticRewardConfig(curiosity_weight=1.0),
    )
    high = compute_total_reward(
        curiosity=5.0,
        energy_improvement=0.0,
        hydration_improvement=0.0,
        damage=0.0,
        motor_effort=0.0,
        instability=0.0,
        config=HomeostaticRewardConfig(curiosity_weight=1.0),
    )
    assert low == 0.0
    assert high == 1.0


def test_train_step_skips_when_buffer_small(tmp_path) -> None:
    config = TrainConfig(
        body_dim=32,
        chemical_dim=8,
        internal_dim=8,
        latent_dim=32,
        action_dim=5,
        batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
    )
    trainer = Trainer(config)
    trainer.add_transition(_transition(obs_dim=8 + 8 + 32))
    assert trainer.train_step() == {}
