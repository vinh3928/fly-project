"""Deterministic Phase 0–8 fake-env benchmark suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flybrain.curiosity import CuriosityConfig
from flybrain.env import FakeFlyConfig, FakeFlyEnv
from flybrain.pipeline.live_mind import LiveMindConfig, LiveMindPipeline, make_live_env
from flybrain.policy import ActorCriticPolicy
from flybrain.seeding import seed_everything
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward
from flybrain.training.train_world_model import train_world_model
from flybrain.unity_env import DEFAULT_ACTION_DIM, DEFAULT_OBS_DIM

DEFAULT_SEED = 42
DEFAULT_EXPECTED_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "expected_seed42.json"

# Absolute tolerances for float regression checks.
FLOAT_ABS_TOL = {
    "homeostatic_reward": 1e-5,
    "env_reward": 1e-5,
    "distance_reduction": 1e-5,
    "coverage": 1e-6,
    "curiosity_reward_mean": 1e-5,
    "exploration_xp": 1e-4,
    "baseline_loss": 1e-4,
    "eval_loss": 1e-4,
    "sugar_consumed": 1e-6,
    "maturity": 1e-6,
}


@dataclass
class BenchmarkConfig:
    seed: int = DEFAULT_SEED
    p4_steps: int = 40
    p5_steps: int = 80
    p6_steps: int = 80
    p7_wm_steps: int = 40
    p8_steps: int = 60


def _round_metrics(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return float(f"{obj:.8g}")
    return obj


def _live_config(
    *,
    seed: int,
    steps: int,
    fly_id: str,
    curiosity: CuriosityConfig | None = None,
    use_vision_head: bool = False,
) -> LiveMindConfig:
    return LiveMindConfig(
        backend="in_memory",
        max_steps=steps,
        llm_mock=True,
        fly_id=fly_id,
        seed=seed,
        use_vision_head=use_vision_head,
        use_heuristic_if_untrained=True,
        curiosity=curiosity if curiosity is not None else CuriosityConfig(enabled=True),
        enable_retrospective=True,
    )


def run_p0_smoke(*, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    env = FakeFlyEnv(FakeFlyConfig(seed=seed))
    obs, info = env.reset(seed=seed)
    policy = ActorCriticPolicy(latent_dim=int(obs.shape[0]), action_dim=DEFAULT_ACTION_DIM)
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
    return {
        "ok": True,
        "obs_dim": int(obs.shape[0]),
        "action_dim": int(action_np.shape[0]),
        "env_reward": float(reward),
        "homeostatic_reward": float(total),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "seed": int(info.get("seed", seed)),
        "expected_obs_dim": DEFAULT_OBS_DIM,
        "expected_action_dim": DEFAULT_ACTION_DIM,
    }


def _run_live(
    *,
    seed: int,
    steps: int,
    fly_id: str,
    fake: bool = False,
    fake_visual_nav: bool = False,
    fake_food_search: bool = False,
    fake_explore: bool = False,
    curiosity_enabled: bool = True,
    use_vision_head: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    env = make_live_env(
        fake=fake,
        fake_visual_nav=fake_visual_nav,
        fake_food_search=fake_food_search,
        fake_explore=fake_explore,
        seed=seed,
    )
    cfg = _live_config(
        seed=seed,
        steps=steps,
        fly_id=fly_id,
        curiosity=CuriosityConfig(enabled=curiosity_enabled),
        use_vision_head=use_vision_head,
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        result = pipeline.run(max_steps=steps)
    finally:
        pipeline.close()
    last = result.get("last") or {}
    start = float(result.get("start_distance", 0.0))
    end = float(result.get("end_distance", start))
    return {
        "steps": int(result["steps"]),
        "has_stm": last.get("stm_id") is not None,
        "has_thought": last.get("thought_id") is not None,
        "action_dim": len((last.get("intent") or {}).get("action") or []),
        "start_distance": start,
        "end_distance": end,
        "distance_reduction": float(start - end),
        "wall_hits": int(result.get("wall_hits", 0)),
        "success": bool(result.get("success", False)),
        "sugar_consumed": float(result.get("sugar_consumed", 0.0)),
        "bitter_contacts": int(result.get("bitter_contacts", 0)),
        "maturity": float(result.get("maturity", 0.0)),
        "labeled_outcomes": int(result.get("labeled_outcomes", 0)),
        "coverage": float(result.get("coverage", 0.0)),
        "unique_cells": int(result.get("unique_cells", 0)),
        "exploration_xp": float(result.get("exploration_xp", 0.0)),
        "curiosity_reward_mean": float(result.get("curiosity_reward_mean", 0.0)),
        "curiosity_enabled": bool(curiosity_enabled),
    }


def run_p7_world_model(*, seed: int, steps: int) -> dict[str, Any]:
    seed_everything(seed)
    metrics = train_world_model(steps=steps, batch_size=32, seed=seed, device="cpu")
    return {
        "baseline_loss": float(metrics["baseline_loss"]),
        "eval_loss": float(metrics["eval_loss"]),
        "train_loss": float(metrics["loss"]),
        "improved": bool(metrics["eval_loss"] < metrics["baseline_loss"]),
    }


def run_phase_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    cfg = config or BenchmarkConfig()
    seed = int(cfg.seed)
    phases = {
        "p0_smoke": run_p0_smoke(seed=seed),
        "p4_fake": _run_live(
            seed=seed,
            steps=cfg.p4_steps,
            fly_id="bench_p4",
            fake=True,
            curiosity_enabled=True,
        ),
        "p5_visual": _run_live(
            seed=seed,
            steps=cfg.p5_steps,
            fly_id="bench_p5",
            fake_visual_nav=True,
            use_vision_head=True,
            curiosity_enabled=True,
        ),
        "p6_food": _run_live(
            seed=seed,
            steps=cfg.p6_steps,
            fly_id="bench_p6",
            fake_food_search=True,
            curiosity_enabled=True,
        ),
        "p7_world_model": run_p7_world_model(seed=seed, steps=cfg.p7_wm_steps),
        "p8_explore": _run_live(
            seed=seed,
            steps=cfg.p8_steps,
            fly_id="bench_p8",
            fake_explore=True,
            curiosity_enabled=True,
        ),
        "p8_baseline": _run_live(
            seed=seed,
            steps=cfg.p8_steps,
            fly_id="bench_p8b",
            fake_explore=True,
            curiosity_enabled=False,
        ),
    }
    report = {
        "seed": seed,
        "suite": "phase0_8_fake",
        "config": {
            "p4_steps": cfg.p4_steps,
            "p5_steps": cfg.p5_steps,
            "p6_steps": cfg.p6_steps,
            "p7_wm_steps": cfg.p7_wm_steps,
            "p8_steps": cfg.p8_steps,
        },
        "phases": phases,
        "gates": _evaluate_gates(phases),
    }
    return _round_metrics(report)


def _evaluate_gates(phases: dict[str, Any]) -> dict[str, bool]:
    p0 = phases["p0_smoke"]
    p4 = phases["p4_fake"]
    p5 = phases["p5_visual"]
    p6 = phases["p6_food"]
    p7 = phases["p7_world_model"]
    p8 = phases["p8_explore"]
    p8b = phases["p8_baseline"]
    return {
        "p0_shapes_ok": (
            p0["obs_dim"] == p0["expected_obs_dim"]
            and p0["action_dim"] == p0["expected_action_dim"]
        ),
        "p4_stm_ok": bool(p4["has_stm"] and p4["has_thought"] and p4["action_dim"] == 5),
        "p5_progress_ok": bool(p5["success"] or p5["distance_reduction"] >= 0.02),
        "p6_progress_ok": bool(
            p6["sugar_consumed"] > 0.0
            or p6["distance_reduction"] >= 0.02
            or p6["end_distance"] < p6["start_distance"]
        ),
        "p7_wm_improved": bool(p7["improved"]),
        "p8_coverage_ge_baseline": int(p8["unique_cells"]) >= int(p8b["unique_cells"]),
        "p8_xp_positive": float(p8["exploration_xp"]) > 0.0,
    }


def all_gates_passed(report: dict[str, Any]) -> bool:
    gates = report.get("gates") or {}
    return bool(gates) and all(bool(v) for v in gates.values())


def compare_reports(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    float_tol: dict[str, float] | None = None,
) -> list[str]:
    """Return human-readable diffs; empty list means match within tolerance."""
    tol = {**FLOAT_ABS_TOL, **(float_tol or {})}
    diffs: list[str] = []

    def walk(a: Any, e: Any, path: str) -> None:
        if isinstance(e, dict) and isinstance(a, dict):
            for key in sorted(set(e) | set(a)):
                if key not in a:
                    diffs.append(f"{path}.{key}: missing in actual")
                elif key not in e:
                    continue  # extra actual keys ok (forward compatible)
                else:
                    walk(a[key], e[key], f"{path}.{key}" if path else key)
            return
        if isinstance(e, bool) or isinstance(a, bool):
            if bool(a) != bool(e):
                diffs.append(f"{path}: {a!r} != {e!r}")
            return
        if isinstance(e, (int, float)) and isinstance(a, (int, float)):
            leaf = path.rsplit(".", 1)[-1]
            if isinstance(e, int) and isinstance(a, int) and not isinstance(e, bool):
                if a != e:
                    diffs.append(f"{path}: {a} != {e}")
                return
            abs_tol = float(tol.get(leaf, 1e-5))
            if abs(float(a) - float(e)) > abs_tol:
                diffs.append(f"{path}: {a} != {e} (tol={abs_tol})")
            return
        if a != e:
            diffs.append(f"{path}: {a!r} != {e!r}")

    walk(actual.get("phases"), expected.get("phases"), "phases")
    walk(actual.get("gates"), expected.get("gates"), "gates")
    if int(actual.get("seed", -1)) != int(expected.get("seed", -2)):
        diffs.append(f"seed: {actual.get('seed')} != {expected.get('seed')}")
    return diffs


def load_report(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
