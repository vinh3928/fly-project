#!/usr/bin/env bash
# Run Unity EditMode tests in batchmode.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITY_PROJECT="$ROOT/unity"
OUT_DIR="$ROOT/builds"
RESULTS="$OUT_DIR/editmode-results.xml"
LOG="$OUT_DIR/unity_editmode_tests.log"

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
echo "Running EditMode tests with: $UNITY_BIN"
echo "Results: $RESULTS"
echo "Log: $LOG"

"$UNITY_BIN" \
  -batchmode \
  -nographics \
  -projectPath "$UNITY_PROJECT" \
  -runTests \
  -testPlatform EditMode \
  -testResults "$RESULTS" \
  -logFile "$LOG"

echo "EditMode tests finished. Inspect $RESULTS"
