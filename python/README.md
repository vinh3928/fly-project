# flybrain

Python training and mind package for the fly embodied-learning simulation.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mind]"
```

Optional Unity ML-Agents bridge (prefer the repo script so Python 3.10 pins match):

```bash
# from repo root
../scripts/setup_mlagents_venv.sh
# or: pip install -e ".[mlagents]"
```

## Modules

| Module | Role |
| --- | --- |
| `flybrain.pipeline` | Live mind loop (sync / async Architecture B) |
| `flybrain.thinking` | Composer, two-layer future, route_home, stimulus interrupt, devil’s advocate |
| `flybrain.memory` | pgvector STM, Neo4j place graph + L1/L2 cache |
| `flybrain.motor` | Motor layer (heuristic tracker / optional policy net) |
| `flybrain.curiosity` | Novelty, curiosity drive, boredom / blocked escape |
| `flybrain.encoders` | Vision / body / chemical / internal encoders |
| `flybrain.world_model` | Next-state prediction |
| `flybrain.policy` | Actor-critic locomotion policy |
| `flybrain.replay` | Transition replay buffer |
| `flybrain.training` | Training loops and checkpointing |
| `flybrain.benchmark` | Phase 0–8 deterministic fake-env scoreboard |
| `flybrain.debug` | Unity SideChannels (route path, mind log) |

Fake envs: `env.py`, `env_visual_nav`, `env_food_search`, `env_explore`, `env_locomotion`.

## Smoke / live mind

```bash
python -m scripts.smoke_exchange
python -m scripts.run_live_mind --fake --steps 80
pytest
```

Against Unity Editor (ML-Agents venv):

```bash
# Play Chamber.unity, then:
../.venv-mlagents/bin/python -m scripts.run_live_mind --editor --continuous --mind-interval 4
```

Memory services: `docker compose -f ../docker/docker-compose.mind.yml up -d`.

## Phase 0–8 deterministic benchmark

Seeded fake-env suite (no Unity). Writes/compares `benchmarks/expected_seed42.json`.

```bash
export PYTHONHASHSEED=0
python -m scripts.run_phase_benchmark --check --self-check
# After intentional metric changes:
python -m scripts.run_phase_benchmark --write
```

## Unity launch

```bash
# from repo root
../scripts/build_unity_player.sh   # or: Tools → Fly → Build Training Player
source .venv-mlagents/bin/activate
python -m scripts.launch_unity
```

See the [repo README](../README.md) and [docs/](../docs/) for phases, architecture, and Unity setup.
