from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from flybrain.curiosity import CuriosityModule
from flybrain.encoders import MultimodalEncoder
from flybrain.policy import ActorCriticPolicy
from flybrain.replay import ReplayBuffer, Transition
from flybrain.world_model import NextStateWorldModel


@dataclass
class TrainConfig:
    body_dim: int = 200
    chemical_dim: int = 64
    internal_dim: int = 8
    vision_channels: int = 1
    latent_dim: int = 256
    action_dim: int = 5
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_capacity: int = 50_000
    device: str = "cpu"
    checkpoint_dir: str = "../data/checkpoints"
    log_dir: str = "../data/exports/tensorboard"
    seed: int = 42


class Trainer:
    """Offline/self-play trainer skeleton for world model, curiosity, and policy."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()
        torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device)

        self.encoder = MultimodalEncoder(
            body_dim=self.config.body_dim,
            chemical_dim=self.config.chemical_dim,
            internal_dim=self.config.internal_dim,
            vision_channels=self.config.vision_channels,
            latent_dim=self.config.latent_dim,
        ).to(self.device)
        self.world_model = NextStateWorldModel(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        ).to(self.device)
        self.curiosity = CuriosityModule(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        ).to(self.device)
        self.policy = ActorCriticPolicy(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        ).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.world_model.parameters())
            + list(self.curiosity.parameters())
            + list(self.policy.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self.step = 0

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.config.log_dir)

    def add_transition(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def train_step(self) -> dict[str, float]:
        if len(self.buffer) < self.config.batch_size:
            return {}

        batch = self.buffer.sample_tensors(self.config.batch_size)
        obs = torch.as_tensor(batch["observation"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(
            batch["next_observation"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)

        # Synthetic vision placeholders when eyes are absent in vector-only buffers.
        bsz = obs.shape[0]
        left = torch.zeros(bsz, self.config.vision_channels, 64, 64, device=self.device)
        right = left.clone()
        chem_dim = self.config.chemical_dim
        int_dim = self.config.internal_dim
        chemical = obs[:, :chem_dim]
        internal = obs[:, chem_dim : chem_dim + int_dim]
        body = obs[:, chem_dim + int_dim :]
        # Pad/truncate body to configured size.
        if body.shape[-1] < self.config.body_dim:
            pad = torch.zeros(bsz, self.config.body_dim - body.shape[-1], device=self.device)
            body = torch.cat([body, pad], dim=-1)
        else:
            body = body[:, : self.config.body_dim]

        next_chemical = next_obs[:, :chem_dim]
        next_internal = next_obs[:, chem_dim : chem_dim + int_dim]
        next_body = next_obs[:, chem_dim + int_dim :]
        if next_body.shape[-1] < self.config.body_dim:
            pad = torch.zeros(bsz, self.config.body_dim - next_body.shape[-1], device=self.device)
            next_body = torch.cat([next_body, pad], dim=-1)
        else:
            next_body = next_body[:, : self.config.body_dim]

        latent, _ = self.encoder(left, right, body, chemical, internal)
        next_latent, _ = self.encoder(left, right, next_body, next_chemical, next_internal)
        next_latent = next_latent.detach()

        world_loss = self.world_model.prediction_loss(latent, actions, next_latent)
        curiosity_reward, curiosity_loss = self.curiosity.intrinsic_reward(
            latent, actions, next_latent
        )
        _, log_prob, value = self.policy.act(latent)
        advantage = (rewards + curiosity_reward.detach() - value).detach()
        policy_loss = -(log_prob * advantage).mean() + 0.5 * torch.mean(
            (value - (rewards + curiosity_reward.detach())) ** 2
        )

        loss = world_loss + curiosity_loss + policy_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1
        metrics = {
            "loss/total": float(loss.item()),
            "loss/world": float(world_loss.item()),
            "loss/curiosity": float(curiosity_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "reward/batch_mean": float(rewards.mean().item()),
            "reward/curiosity_mean": float(curiosity_reward.mean().item()),
        }
        for key, value_f in metrics.items():
            self.writer.add_scalar(key, value_f, self.step)
        return metrics

    def save_checkpoint(self, name: str = "latest.pt") -> Path:
        path = Path(self.config.checkpoint_dir) / name
        payload = {
            "step": self.step,
            "config": asdict(self.config),
            "encoder": self.encoder.state_dict(),
            "world_model": self.world_model.state_dict(),
            "curiosity": self.curiosity.state_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, path)
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps({"step": self.step, "path": str(path)}, indent=2))
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.step = int(payload["step"])
        self.encoder.load_state_dict(payload["encoder"])
        self.world_model.load_state_dict(payload["world_model"])
        self.curiosity.load_state_dict(payload["curiosity"])
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])


def main() -> None:
    trainer = Trainer()
    print(f"flybrain trainer ready on {trainer.device}; buffer={len(trainer.buffer)}")
    trainer.save_checkpoint("init.pt")
    print("Wrote initial checkpoint.")


if __name__ == "__main__":
    main()
