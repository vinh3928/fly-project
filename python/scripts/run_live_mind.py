"""
Run the Phase 4/5/6 live mind loop: observe → embed → STM → think → act.

Examples:
  python -m scripts.run_live_mind --fake
  python -m scripts.run_live_mind --fake-visual-nav
  python -m scripts.run_live_mind --fake-food-search
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
    parser.add_argument("--editor", action="store_true", help="Connect to Unity Editor Play mode")
    parser.add_argument("--build", type=str, default=None, help="Path to Unity player build")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to mind.yaml")
    parser.add_argument("--visual-nav-config", type=str, default=None, help="Path to visual_nav.yaml")
    parser.add_argument("--pgvector", action="store_true", help="Force pgvector backend")
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

    use_fake = (
        args.fake
        or args.fake_visual_nav
        or args.fake_food_search
        or (not args.editor and args.build is None)
    )
    env = make_live_env(
        fake=use_fake and not args.fake_visual_nav and not args.fake_food_search,
        fake_visual_nav=args.fake_visual_nav,
        fake_food_search=args.fake_food_search,
        editor=args.editor,
        build=args.build,
        port=args.port,
    )
    pipeline = LiveMindPipeline(env, config=cfg)
    try:
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
