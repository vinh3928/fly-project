# flybrain

Python training package for the fly embodied-learning simulation.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional Unity ML-Agents bridge:

```bash
pip install -e ".[mlagents]"
```

## Modules

| Module | Role |
| --- | --- |
| `flybrain.encoders` | Vision / body / chemical / internal encoders |
| `flybrain.world_model` | Next-state prediction |
| `flybrain.curiosity` | Intrinsic reward from prediction error |
| `flybrain.policy` | Actor-critic locomotion policy |
| `flybrain.replay` | Transition replay buffer |
| `flybrain.training` | Training loop and checkpointing |

## Smoke test

```bash
python -m scripts.smoke_exchange
pytest
```

## Phase 0–8 deterministic benchmark

Seeded fake-env suite (no Unity). Writes/compares `benchmarks/expected_seed42.json`.

```bash
python -m scripts.run_phase_benchmark --check --self-check
# After intentional metric changes:
python -m scripts.run_phase_benchmark --write
```

## Unity launch

```bash
# from repo root
../scripts/build_unity_player.sh   # or: Tools → Fly → Build Training Player
pip install -e '.[mlagents]'
python -m scripts.launch_unity
```
