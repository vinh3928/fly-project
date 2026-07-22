"""
Launch the Unity fly chamber from Python and exchange one observation/action.

Prerequisites:
  1. Build the player: ./scripts/build_unity_player.sh
     (or Tools → Fly → Build Training Player in the Editor)
  2. pip install -e '.[mlagents]'

Editor mode (no build):
  1. Open Chamber.unity, press Play
  2. python -m scripts.launch_unity --editor
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from flybrain.unity_env import UnityFlyConfig, UnityFlyEnv, default_build_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--editor",
        action="store_true",
        help="Connect to a running Unity Editor play session (file_name=None)",
    )
    parser.add_argument(
        "--build",
        type=str,
        default=None,
        help="Path to FlyChamber player (.app / .exe / .x86_64). Default: builds/FlyChamber.*",
    )
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--graphics", action="store_true", help="Show the player window")
    args = parser.parse_args(argv)

    if args.editor:
        file_name = None
        print("Connecting to Unity Editor on port", args.port)
        print("Make sure Chamber.unity is in Play mode first.")
    else:
        build = args.build or (str(default_build_path()) if default_build_path() else None)
        if build is None:
            print(
                "No Unity build found under builds/.\n"
                "Run: ./scripts/build_unity_player.sh\n"
                "Or connect to the editor: python -m scripts.launch_unity --editor",
                file=sys.stderr,
            )
            return 2
        file_name = build
        print("Launching Unity build:", file_name)

    config = UnityFlyConfig(
        file_name=file_name,
        base_port=args.port,
        no_graphics=not args.graphics and file_name is not None,
    )

    with UnityFlyEnv(config) as env:
        obs, info = env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[0] = 0.5  # walk forward
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        print("unity_launch_ok")
        print(f"behavior={info.get('behavior')}")
        print(f"obs_shape={obs.shape} action_shape={action.shape}")
        print(f"reward={reward:.4f} terminated={terminated} truncated={truncated}")
        print(f"steps={step_info.get('steps')}")
        assert obs.shape == next_obs.shape
        assert np.isfinite(reward)
        assert obs.shape[0] >= 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
