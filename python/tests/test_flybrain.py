from __future__ import annotations

import numpy as np
import torch

from flybrain.curiosity import CuriosityModule
from flybrain.encoders import MultimodalEncoder, VisionEncoder
from flybrain.env import FakeFlyEnv
from flybrain.policy import ActorCriticPolicy
from flybrain.replay import ReplayBuffer, Transition
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward
from flybrain.training.train import TrainConfig, Trainer
from flybrain.world_model import NextStateWorldModel


def test_vision_encoder_shape() -> None:
    enc = VisionEncoder(in_channels=1, latent_dim=128)
    x = torch.randn(4, 1, 64, 64)
    y = enc(x)
    assert y.shape == (4, 128)


def test_multimodal_encoder_forward() -> None:
    enc = MultimodalEncoder(body_dim=200, chemical_dim=64, internal_dim=8, latent_dim=256)
    left = torch.randn(2, 1, 64, 64)
    right = torch.randn(2, 1, 64, 64)
    body = torch.randn(2, 200)
    chemical = torch.randn(2, 64)
    internal = torch.randn(2, 8)
    latent, hidden = enc(left, right, body, chemical, internal)
    assert latent.shape == (2, 256)
    assert hidden is not None and hidden.shape == (2, 256)


def test_world_model_loss() -> None:
    model = NextStateWorldModel(latent_dim=32, action_dim=5)
    latent = torch.randn(8, 32)
    action = torch.randn(8, 5)
    next_latent = torch.randn(8, 32)
    loss = model.prediction_loss(latent, action, next_latent)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_curiosity_reward_capped() -> None:
    module = CuriosityModule(latent_dim=16, action_dim=4, reward_cap=0.5)
    latent = torch.randn(8, 16)
    action = torch.randn(8, 4)
    next_latent = torch.randn(8, 16)
    reward, loss = module.intrinsic_reward(latent, action, next_latent)
    assert reward.shape == (8,)
    assert float(reward.max().detach()) <= 0.5 + 1e-5
    assert torch.isfinite(loss)


def test_policy_action_shape() -> None:
    policy = ActorCriticPolicy(latent_dim=64, action_dim=5)
    latent = torch.randn(3, 64)
    action, log_prob, value = policy.act(latent)
    assert action.shape == (3, 5)
    assert log_prob.shape == (3,)
    assert value.shape == (3,)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_replay_buffer_sample() -> None:
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.add(
            Transition(
                observation=np.zeros(8, dtype=np.float32) + i,
                action=np.zeros(5, dtype=np.float32),
                reward=float(i),
                next_observation=np.zeros(8, dtype=np.float32) + i + 1,
                done=False,
            )
        )
    batch = buf.sample_tensors(4)
    assert batch["observation"].shape == (4, 8)
    assert batch["action"].shape == (4, 5)
    assert batch["reward"].shape == (4,)


def test_homeostatic_reward() -> None:
    reward = compute_total_reward(
        curiosity=0.5,
        energy_improvement=0.1,
        hydration_improvement=0.0,
        damage=0.2,
        motor_effort=0.1,
        instability=0.0,
        config=HomeostaticRewardConfig(
            curiosity_weight=0.1,
            energy_weight=1.0,
            hydration_weight=1.0,
            damage_weight=2.0,
            effort_weight=0.05,
            instability_weight=0.5,
        ),
    )
    # 0.1*0.5 + 1.0*0.1 - 2.0*0.2 - 0.05*0.1 = 0.05 + 0.1 - 0.4 - 0.005 = -0.255
    assert abs(reward - (-0.255)) < 1e-6


def test_checkpoint_roundtrip(tmp_path) -> None:
    config = TrainConfig(
        body_dim=32,
        chemical_dim=8,
        internal_dim=8,
        latent_dim=64,
        action_dim=5,
        batch_size=4,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
    )
    trainer = Trainer(config)
    for _ in range(8):
        obs = np.random.randn(8 + 8 + 32).astype(np.float32)
        nxt = np.random.randn(8 + 8 + 32).astype(np.float32)
        trainer.add_transition(
            Transition(
                observation=obs,
                action=np.zeros(5, dtype=np.float32),
                reward=0.0,
                next_observation=nxt,
                done=False,
            )
        )
    metrics = trainer.train_step()
    assert "loss/total" in metrics
    path = trainer.save_checkpoint("test.pt")
    trainer2 = Trainer(config)
    trainer2.load_checkpoint(path)
    assert trainer2.step == trainer.step


def test_default_build_path_helper() -> None:
    from flybrain.unity_env import DEFAULT_OBS_DIM, default_build_path

    # Helper must not crash when no build exists (Phase 0 CI).
    path = default_build_path()
    assert path is None or path.exists()
    assert DEFAULT_OBS_DIM == 197


def test_fake_env_episode() -> None:
    env = FakeFlyEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (197,)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert "effort" in info
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
