#!/usr/bin/env bash
# Train Phase 9 motor stack.
# 1) Fast fake-env motor policy (always available):
# 2) Optional ML-Agents PPO on Unity player (requires rebuild + mlagents venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/python"

echo "==> Training fake-env motor policy (architecture B tracker)"
python -m scripts.train_motor_layer --steps "${MOTOR_STEPS:-400}" --checkpoint checkpoints/motor_policy.pt

if [[ "${TRAIN_MLAGENTS:-0}" == "1" ]]; then
  echo "==> ML-Agents PPO (set useWalkingDemo=false, includeMotorGoalsInObs=true, useLocomotionReward=true)"
  echo "    Rebuild player first: ./scripts/build_unity_player.sh"
  # shellcheck disable=SC1091
  source "$ROOT/python/.venv-mlagents/bin/activate"
  mlagents-learn "$ROOT/python/configs/mlagents_walk.yaml" \
    --run-id "${RUN_ID:-fly_walk_ppo}" \
    --force
fi

echo "phase9_train_ok"
