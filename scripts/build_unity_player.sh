#!/usr/bin/env bash
# Build the FlyChamber standalone player for Python / ML-Agents.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITY_PROJECT="$ROOT/unity"
OUT_DIR="$ROOT/builds"
LOG="$OUT_DIR/unity_build.log"

UNITY_BIN="${UNITY_BIN:-}"
if [[ -z "$UNITY_BIN" ]]; then
  for candidate in \
    "/Applications/Unity/Hub/Editor/6000.5.4f1/Unity.app/Contents/MacOS/Unity" \
    "/Applications/Unity/Hub/Editor/6000.0.58f2/Unity.app/Contents/MacOS/Unity"
  do
    if [[ -x "$candidate" ]]; then
      UNITY_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$UNITY_BIN" || ! -x "$UNITY_BIN" ]]; then
  echo "Unity editor not found. Set UNITY_BIN to your Unity executable." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "Building with: $UNITY_BIN"
echo "Log: $LOG"

"$UNITY_BIN" \
  -batchmode \
  -nographics \
  -quit \
  -projectPath "$UNITY_PROJECT" \
  -executeMethod FlySimulation.EditorTools.FlyPlayerBuilder.BuildCli \
  -logFile "$LOG"

echo "Build finished. Player should be under $OUT_DIR"
ls -la "$OUT_DIR" || true
