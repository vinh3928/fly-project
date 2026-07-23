"""
Run the Phase 4–8 live mind loop: observe → embed → STM → think → act.

Examples:
  python -m scripts.run_live_mind --fake
  python -m scripts.run_live_mind --fake-visual-nav
  python -m scripts.run_live_mind --fake-food-search
  python -m scripts.run_live_mind --fake-explore
  python -m scripts.run_live_mind --editor   # Unity Play + useWalkingDemo=false
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from flybrain.pipeline.live_mind import (
    LiveMindConfig,
    LiveMindPipeline,
    load_mind_config,
    make_live_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Use FakeFlyEnv (no Unity)")
    parser.add_argument(
        "--fake-visual-nav",
        action="store_true",
        help="Use FakeVisualNavEnv with synthetic stereo vision cues",
    )
    parser.add_argument(
        "--fake-food-search",
        action="store_true",
        help="Use FakeFoodSearchEnv with sugar/bitter homeostasis",
    )
    parser.add_argument(
        "--fake-explore",
        action="store_true",
        help="Use FakeExplorationEnv (high energy, sparse sugar, coverage metrics)",
    )
    parser.add_argument("--editor", action="store_true", help="Connect to Unity Editor Play mode")
    parser.add_argument("--build", type=str, default=None, help="Path to Unity player build")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="ML-Agents port (default: 5004 for --editor, 5005 for builds)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to mind.yaml")
    parser.add_argument("--visual-nav-config", type=str, default=None, help="Path to visual_nav.yaml")
    parser.add_argument("--pgvector", action="store_true", help="Force pgvector backend")
    parser.add_argument(
        "--no-curiosity",
        action="store_true",
        help="Disable Phase 8 curiosity drive (baseline coverage comparison)",
    )
    args = parser.parse_args(argv)

    raw = load_mind_config(args.config)
    vn_path = (
        Path(args.visual_nav_config)
        if args.visual_nav_config
        else Path(__file__).resolve().parents[1] / "configs" / "visual_nav.yaml"
    )
    visual_nav = {}
    if vn_path.exists():
        with open(vn_path, encoding="utf-8") as handle:
            visual_nav = yaml.safe_load(handle) or {}
    cfg = LiveMindConfig.from_yaml(raw, visual_nav=visual_nav)
    if args.pgvector:
        cfg.backend = "pgvector"
    if args.no_curiosity and cfg.curiosity is not None:
        cfg.curiosity.enabled = False

    use_fake = (
        args.fake
        or args.fake_visual_nav
        or args.fake_food_search
        or args.fake_explore
        or (not args.editor and args.build is None)
    )
    if args.editor:
        port = args.port if args.port is not None else 5004
        print(
            "Connecting to Unity Editor on port",
            port,
            flush=True,
        )
        print(
            "IMPORTANT order:\n"
            "  1) Stop Play in Unity if it is already playing\n"
            "  2) Keep this script running until you see Listening / unity_env_ready\n"
            "  3) THEN press Play in Chamber.unity\n"
            "If Unity Console says 'Couldn't connect to trainer... inference instead',\n"
            "Stop Play and press Play again while this script is waiting.",
            flush=True,
        )
    else:
        port = args.port
    env = make_live_env(
        fake=use_fake
        and not args.fake_visual_nav
        and not args.fake_food_search
        and not args.fake_explore,
        fake_visual_nav=args.fake_visual_nav,
        fake_food_search=args.fake_food_search,
        fake_explore=args.fake_explore,
        editor=args.editor,
        build=args.build,
        port=port,
    )
    print("unity_env_ready", flush=True)
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
        if args.editor or args.build:
            from flybrain.signals.builder import INTERNAL_OFFSET

            probe = getattr(env, "_current_obs", None)
            if callable(probe):
                raw_obs = probe()
            else:
                raw_obs, _ = env.reset()
            raw_obs = __import__("numpy").asarray(raw_obs, dtype="float32").reshape(-1)
            energy_raw = float(raw_obs[INTERNAL_OFFSET]) if raw_obs.size > INTERNAL_OFFSET else None
            print(
                "unity_obs_probe",
                {
                    "shape": list(raw_obs.shape),
                    "nnz": int((abs(raw_obs) > 1e-8).sum()),
                    "energy_at_176": energy_raw,
                    "head5": [float(x) for x in raw_obs[:5]],
                    "internal8": [float(x) for x in raw_obs[INTERNAL_OFFSET : INTERNAL_OFFSET + 8]]
                    if raw_obs.size >= INTERNAL_OFFSET + 8
                    else [],
                },
                flush=True,
            )
        result = pipeline.run(max_steps=args.steps or cfg.max_steps)
    finally:
        pipeline.close()

    print("live_mind_ok")
    print(json.dumps(result, indent=2, default=str))
    last = result.get("last") or {}
    assert last.get("stm_id") is not None
    assert last.get("thought_id") is not None
    assert last.get("devil_critique")
    return 0


if __name__ == "__main__":
    sys.exit(main())
