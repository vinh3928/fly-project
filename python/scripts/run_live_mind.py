"""
Run the Phase 4–9 live mind loop: observe → embed → STM → think → motor → act.

Examples:
  python -m scripts.run_live_mind --fake --minutes 10 --realtime
  python -m scripts.run_live_mind --editor --continuous --mind-interval 4
  python -m scripts.run_live_mind --fake --continuous --realtime   # Ctrl+C to stop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Line-buffer stdout so editor/IDE terminals show live logs immediately.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
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
    parser.add_argument(
        "--fake-locomotion",
        action="store_true",
        help="Use FakeLocomotionEnv (Phase 9 goal-conditioned motor)",
    )
    parser.add_argument("--editor", action="store_true", help="Connect to Unity Editor Play mode")
    parser.add_argument("--build", type=str, default=None, help="Path to Unity player build")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="ML-Agents port (default: 5004 for --editor, 5005 for builds)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Fixed step count")
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Run for N minutes of decision-time (steps = minutes*60*decision_hz)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run until Ctrl+C (ignores --steps/--minutes)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace fake envs to decision_hz wall-clock (Unity already runs in sim time)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to mind.yaml")
    parser.add_argument("--visual-nav-config", type=str, default=None, help="Path to visual_nav.yaml")
    parser.add_argument("--pgvector", action="store_true", help="Force pgvector backend")
    parser.add_argument(
        "--no-curiosity",
        action="store_true",
        help="Disable Phase 8 curiosity drive (baseline coverage comparison)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=None,
        help="Print per-step live mind / motor lines (default: on unless --quiet)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-step console logging",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Log every N steps (default 1 = every step)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Append [live …] lines to this file (default: logs/live_mind.log)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not write a live_mind.log file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Longer mind text + full DETAIL JSON on each MIND* tick",
    )
    parser.add_argument(
        "--mind-interval",
        type=int,
        default=None,
        help="Think every N motor steps (1 = mind updates every step)",
    )
    return parser.parse_args(argv)


def _open_editor_env_early(port: int):
    """Bind ML-Agents port *before* importing torch-heavy live_mind.

    If Unity Play starts while Python is still loading Torch, Academy falls back to
    inference (zero actions) and the fly stands still. Opening the socket first
    closes that race.
    """
    import logging

    from mlagents_envs.logging_util import get_logger, set_log_level

    set_log_level(logging.INFO)
    get_logger("mlagents_envs").setLevel(logging.INFO)

    from flybrain.unity_env import UnityFlyConfig, UnityFlyEnv

    print(
        f"editor_trainer_binding port={port}\n"
        ">>> STOP Play in Unity if already playing.\n"
        ">>> When you see 'Listening on port …', press Play in Chamber.unity NOW.\n"
        ">>> Do NOT wait for unity_env_ready first — that prints only AFTER Unity connects.\n"
        ">>> If Unity says 'Couldn't connect to trainer... inference instead':\n"
        "    Stop Play, confirm this process still says Listening, press Play again.\n"
        ">>> If you Played once while Python was still importing, quit Play and retry\n"
        "    (or restart the Unity Editor) so Academy is not stuck in inference.",
        flush=True,
    )
    return UnityFlyEnv(
        UnityFlyConfig(
            file_name=None,
            base_port=port,
            no_graphics=False,
            editor_connect=True,
            timeout_wait=600,
            seed=42,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    use_fake = (
        args.fake
        or args.fake_visual_nav
        or args.fake_food_search
        or args.fake_explore
        or args.fake_locomotion
        or (not args.editor and args.build is None)
    )
    port = args.port
    if args.editor:
        port = args.port if args.port is not None else 5004
    elif args.build is not None and port is None:
        port = 5005

    # Critical: open the editor socket before importing the torch stack.
    env = None
    if args.editor and not use_fake:
        env = _open_editor_env_early(int(port))

    import yaml

    from flybrain.pipeline.live_mind import (
        LiveMindConfig,
        LiveMindPipeline,
        load_mind_config,
        make_live_env,
    )

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
    if args.quiet:
        cfg.console_log = False
    else:
        cfg.console_log = True
    if args.log_every is not None:
        cfg.console_log_every = max(1, int(args.log_every))
    else:
        cfg.console_log_every = 1
    if args.no_log_file:
        cfg.log_path = None
    else:
        log_file = (
            Path(args.log_file)
            if args.log_file
            else Path(__file__).resolve().parents[1] / "logs" / "live_mind.log"
        )
        cfg.log_path = str(log_file)
        # Fresh run so "tail -f" is unambiguous.
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("", encoding="utf-8")
    if args.verbose:
        cfg.console_log_verbose = True
    if args.mind_interval is not None:
        cfg.mind_interval = max(1, int(args.mind_interval))

    thinking = raw.get("thinking") or {}
    decision_hz = float(thinking.get("decision_hz", 20))
    if args.continuous:
        max_steps = None
        continuous = True
    elif args.minutes is not None:
        max_steps = max(1, int(float(args.minutes) * 60.0 * decision_hz))
        continuous = False
    elif args.steps is not None:
        max_steps = int(args.steps)
        continuous = False
    else:
        max_steps = cfg.max_steps
        continuous = False

    realtime_hz = decision_hz if (args.realtime and use_fake and not args.editor) else None

    if cfg.console_log:
        dest = cfg.log_path or "(console only)"
        print(
            ">>> LIVE MIND LOGS appear in THIS terminal as lines starting with [live …]\n"
            f">>> Also written to: {dest}\n"
            ">>> Tip: tail -f logs/live_mind.log   (from the python/ directory)\n"
            ">>> They do NOT appear in the Unity Console.",
            flush=True,
        )

    if continuous:
        print(
            f"live_run continuous decision_hz={decision_hz} (Ctrl+C to stop)",
            flush=True,
        )
    else:
        minutes_equiv = float(max_steps) / (60.0 * decision_hz)
        print(
            f"live_run steps={max_steps} (~{minutes_equiv:.1f} min at {decision_hz} Hz)"
            + (" realtime" if realtime_hz else ""),
            flush=True,
        )

    if env is None:
        env = make_live_env(
            fake=use_fake
            and not args.fake_visual_nav
            and not args.fake_food_search
            and not args.fake_explore
            and not args.fake_locomotion,
            fake_visual_nav=args.fake_visual_nav,
            fake_food_search=args.fake_food_search,
            fake_explore=args.fake_explore,
            fake_locomotion=args.fake_locomotion,
            editor=args.editor,
            build=args.build,
            port=port,
        )
    print("unity_env_ready", flush=True)
    if args.editor:
        has_eyes = bool(getattr(env, "left_eye", None) is not None and getattr(env, "right_eye", None) is not None)
        print(f"unity_connected has_eyes={has_eyes}", flush=True)

    pipeline = LiveMindPipeline(env, config=cfg)
    result: dict = {}
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
        print(
            "live_loop_starting "
            "(logs print before each Unity step; ensure Play is running with useWalkingDemo=false)",
            flush=True,
        )
        result = pipeline.run(
            max_steps=max_steps,
            continuous=continuous,
            realtime_hz=realtime_hz,
        )
    except KeyboardInterrupt:
        print("\nlive_mind_interrupted", flush=True)
        result = {
            "steps": pipeline._step,
            "interrupted": True,
            "last": pipeline._last_mind_info,
            "mind_ticks": pipeline._mind_ticks,
            "motor_ticks": pipeline._motor_ticks,
        }
    finally:
        pipeline.close()

    print("live_mind_ok")
    # Full end-of-run snapshot (what you see after Ctrl+C) — also saved to disk.
    result_json = json.dumps(result, indent=2, default=str)
    print(result_json)
    if cfg.log_path:
        last_path = Path(cfg.log_path).with_name("live_mind_last.json")
        last_path.write_text(result_json + "\n", encoding="utf-8")
        print(f">>> Full run snapshot written to: {last_path}", flush=True)
        print(f">>> Per-step logs: {cfg.log_path}", flush=True)
    last = result.get("last") or {}
    if not result.get("interrupted"):
        assert last.get("stm_id") is not None
        assert last.get("thought_id") is not None
        assert last.get("devil_critique")
    return 0


if __name__ == "__main__":
    sys.exit(main())
