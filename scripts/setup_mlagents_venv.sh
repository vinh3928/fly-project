#!/usr/bin/env bash
# Create a Python 3.10 venv with mlagents_envs matching Unity ML-Agents 4.0.x.
# Official PyPI pins require <=3.10.12; we install with --ignore-requires-python
# when the local 3.10 patch level is newer (e.g. 3.10.20).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_DIR="$ROOT/python"
VENV="$PY_DIR/.venv-mlagents"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.10.12 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Need Python 3.10 for ML-Agents. Install python@3.10 (Homebrew) or set PYTHON_BIN." >&2
  exit 1
fi

echo "Using $PYTHON_BIN ($("$PYTHON_BIN" -V))"
rm -rf "$VENV"
"$PYTHON_BIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

# Pins that import cleanly on modern macOS arm64 + Python 3.10.x
python -m pip install \
  'numpy>=1.26,<2' \
  'protobuf==3.20.3' \
  'gym==0.26.2' \
  'grpcio>=1.48' \
  'cloudpickle' 'attrs' 'cattrs==1.5.0' 'Pillow' 'pyyaml' 'filelock' \
  'gymnasium>=0.29,<0.30'

python -m pip install 'mlagents_envs==1.1.0' --ignore-requires-python --no-deps
python -m pip install -e "$PY_DIR" --no-deps

# Keep protobuf pinned last: other deps (e.g. tensorboard) may pull 4+/6+.
python -m pip install 'protobuf==3.20.3'

python - <<'PY'
from mlagents_envs.environment import UnityEnvironment
from flybrain.unity_env import UnityFlyEnv
print("mlagents_envs ready:", UnityEnvironment)
print("UnityFlyEnv ready:", UnityFlyEnv)
PY

echo
echo "Activate with: source python/.venv-mlagents/bin/activate"
echo "Then in Unity: Tools → Fly → Build Training Player"
echo "Then run: cd python && python -m scripts.launch_unity"
