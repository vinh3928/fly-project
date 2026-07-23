"""Live multimodal mind pipeline: observe → embed → STM → think → act."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flybrain.affect.estimator import AffectEstimator
from flybrain.env import FakeFlyEnv
from flybrain.memory.embedder import MindEmbedder
from flybrain.memory.pgvector import ShortTermMemory, open_short_term_memory
from flybrain.signals.builder import SignalBuilder
from flybrain.thinking.service import ThinkingConfig, ThinkingService


def load_mind_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "mind.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class LiveMindConfig:
    fly_id: str = "fly_001"
    experiment_id: str = "experiment_mind_000001"
    max_steps: int = 200
    backend: str = "in_memory"
    dsn: str | None = None
    llm_mock: bool = True
    action_clamps: dict[str, float] | None = None

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> LiveMindConfig:
        db = data.get("database", {})
        thinking = data.get("thinking", {})
        live = data.get("live", {})
        llm = data.get("llm", {})
        return cls(
            fly_id=str(data.get("fly_id", "fly_001")),
            experiment_id=str(data.get("experiment_id", "experiment_mind_000001")),
            max_steps=int(live.get("max_steps", 200)),
            backend=str(db.get("backend", "in_memory")),
            dsn=db.get("dsn"),
            llm_mock=bool(llm.get("mock", True)),
            action_clamps=thinking.get("action_clamps"),
        )


class LiveMindPipeline:
    """One-process orchestrator for Phase 4 live mind loop."""

    def __init__(
        self,
        env: Any,
        *,
        config: LiveMindConfig | None = None,
        memory: ShortTermMemory | None = None,
    ) -> None:
        self.env = env
        self.config = config or LiveMindConfig()
        self.signals = SignalBuilder()
        self.affect = AffectEstimator()
        self.embedder = MindEmbedder()
        self.memory = memory or open_short_term_memory(
            backend=self.config.backend,
            dsn=self.config.dsn,
            dim=self.embedder.total_dim,
        )
        self.thinking = ThinkingService(
            self.memory,
            ThinkingConfig(
                fly_id=self.config.fly_id,
                action_clamps=self.config.action_clamps,
                llm_mock=self.config.llm_mock,
            ),
        )
        self._step = 0

    def close(self) -> None:
        self.memory.close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def process_observation(
        self,
        observation: np.ndarray,
        *,
        sim_time: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sensory = self.signals.build(observation)
        affect = self.affect.estimate(sensory)
        embedding = self.embedder.embed(sensory, affect)
        stm, thought, thought_rec = self.thinking.think_on_embedding(
            embedding=embedding,
            signals=sensory.to_jsonable(),
            affect=affect.to_jsonable(),
            episode_id=self.config.experiment_id,
            step=self._step,
            sim_time=float(sim_time if sim_time is not None else self._step * 0.05),
            fly_id=self.config.fly_id,
        )
        info = {
            "stm_id": stm.id,
            "thought_id": thought_rec.id,
            "refined_thought": thought.refined_thought,
            "devil_critique": thought.devil_critique,
            "intent": thought.intent,
            "affect": affect.to_jsonable(),
        }
        return thought.action, info

    def run(self, *, max_steps: int | None = None) -> dict[str, Any]:
        limit = max_steps if max_steps is not None else self.config.max_steps
        obs, _ = self.env.reset()
        last_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(limit):
            action, info = self.process_observation(obs)
            last_info = info
            obs, _reward, terminated, truncated, _step_info = self.env.step(action)
            self._step += 1
            if terminated or truncated:
                break
        return {
            "steps": self._step,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "last": last_info,
        }


def make_live_env(*, fake: bool = True, editor: bool = False, build: str | None = None, port: int = 5005):
    if fake:
        return FakeFlyEnv()
    from flybrain.unity_env import UnityFlyConfig, UnityFlyEnv

    file_name = None if editor else build
    return UnityFlyEnv(
        UnityFlyConfig(
            file_name=file_name,
            base_port=port,
            no_graphics=not editor and file_name is not None,
        )
    )
