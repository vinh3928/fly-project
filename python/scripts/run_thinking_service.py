"""
Standalone thinking-service helper (same-process API for Phase 4).

For v1, thinking runs inside run_live_mind. This entrypoint demonstrates
consuming a synthetic STM write and producing a thought/intent.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from flybrain.affect.estimator import AffectEstimator
from flybrain.memory.embedder import MindEmbedder
from flybrain.memory.pgvector import open_short_term_memory
from flybrain.pipeline.live_mind import LiveMindConfig, load_mind_config
from flybrain.signals.builder import SignalBuilder
from flybrain.thinking.service import ThinkingConfig, ThinkingService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pgvector", action="store_true")
    args = parser.parse_args(argv)

    raw = load_mind_config(args.config)
    cfg = LiveMindConfig.from_yaml(raw)
    if args.pgvector:
        cfg.backend = "pgvector"

    memory = open_short_term_memory(backend=cfg.backend, dsn=cfg.dsn)
    service = ThinkingService(
        memory,
        ThinkingConfig(fly_id=cfg.fly_id, action_clamps=cfg.action_clamps, llm_mock=cfg.llm_mock),
    )
    builder = SignalBuilder()
    affect_est = AffectEstimator()
    embedder = MindEmbedder()

    obs = np.zeros(197, dtype=np.float32)
    obs[72 + 30] = 0.8  # sugar on first foot taste channel region approx
    # Set internal energy mid-vector: joints72+contacts30+taste48+antenna16+prob10 = 176, energy at 176
    obs[176] = 0.4
    sensory = builder.build(obs)
    # Force sugar in summary path via chemical
    sensory.summary["sugar"] = 0.7
    sensory.summary["bitter"] = 0.1
    sensory.summary["energy"] = 0.4
    affect = affect_est.estimate(sensory)
    embedding = embedder.embed(sensory, affect)
    stm, thought, thought_rec = service.think_on_embedding(
        embedding=embedding,
        signals=sensory.to_jsonable(),
        affect=affect.to_jsonable(),
        fly_id=cfg.fly_id,
    )
    memory.close()

    print("thinking_service_ok")
    print(
        json.dumps(
            {
                "stm_id": stm.id,
                "thought_id": thought_rec.id,
                "devil_critique": thought.devil_critique,
                "refined_thought": thought.refined_thought,
                "action": thought.action.tolist(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
