# Fly Embodied-Learning Simulation

A controlled 3D simulation in which a virtual fly walks, senses, records synchronized multimodal data, and learns via curiosity and homeostatic reinforcement learning.

This repository is an **embodied fly-inspired learning simulation**, not a biologically exact digital fly.

## Status

**Phases 0–9 are complete** (foundation through Architecture B motor layer). Next: Phase 10 (simplified flight).

| Area             | What’s in                                                                         |
| ---------------- | --------------------------------------------------------------------------------- |
| Simulation       | Unity 6 chamber + outside world, six-leg body, sensors, ML-Agents agent           |
| Locomotion       | Balance, tripod gait, root locomotion; mind goals → motor layer every tick        |
| Sensing / record | Vision (hex eyes), proprioception, taste, odor, pose; synced episode logs         |
| Mind             | Affective embeddings → pgvector STM → async thinking + devil’s advocate           |
| Place memory     | Neo4j place graph (walls/senses) with L1/L2 cache; `route_home` when hungry       |
| Learning         | Curiosity / novelty, homeostasis, next-state world model, visual nav, food search |
| Viz              | Display 1 overview, Display 2 fly-follow cam, Display 3 mind/deep-thought log     |
| Tooling          | Fake envs, Phase 0–8 deterministic benchmark, CI, standalone player build         |

Flight is deferred until walking, sensing, recording, and learning stay stable.

## Performance

With Unity live: the bottleneck is Unity’s 20 Hz Academy decision tick (env.step waits for the next decision). Python can run hundreds–thousands of fake steps/sec, so the sim is paced by Unity, not by the mind loop.

**Designed rates** (Unity / mind config):

| Domain                      | Rate                                                                |
| --------------------------- | ------------------------------------------------------------------- |
| Unity physics               | 200 Hz (`Fixed Timestep` 0.005 s)                                   |
| Agent decision / motor tick | 20 Hz (`decision_hz` in `mind.yaml`)                                |
| Vision capture              | 30 Hz                                                               |
| Sensor recording            | 50 Hz                                                               |
| Mind think                  | async worker (default); or every `mind_interval` steps in sync mode |

At **20 Hz decision**, 1 simulated minute ≈ **1,200 steps**; 10 minutes ≈ **12,000 steps**.

**Measured on a standard Apple MacBook Pro (M1 Pro, 32 GB)** — FakeFlyEnv, no Unity graphics, local Neo4j/pgvector when noted, mock or background Ollama devil:

| Mode                                     | Approx. decision throughput | vs 20 Hz realtime |
| ---------------------------------------- | --------------------------- | ----------------- |
| Async mind + in-memory place/STM         | ~3,000–4,000 Hz             | ~150–200×         |
| Async mind + Neo4j (L1 cache) + pgvector | ~1,500 Hz                   | ~75×              |
| Sync mind, `mind_interval=4`, in-memory  | ~500–700 Hz                 | ~25–35×           |

So the Python mind loop is not the bottleneck for realtime on this class of laptop. With **Unity Editor / player**, the loop is paced by Academy decisions at **20 Hz**; an M1/M2/M3 MacBook Pro typically holds that realtime target when `async_mind` is on and the devil LLM runs in the background (`llm.background: true`). Use `--realtime` on fake runs to pace wall-clock to 20 Hz.

Throughput drops when thoughts block the motor path: sync mind with a live local LLM, vision VL calls, or cold Neo4j/pgvector without cache. Numbers vary with thermal state, Docker, and model choice — treat the table as order-of-magnitude, not a guarantee.

## Technology stack

| Layer      | Tools                                                                      |
| ---------- | -------------------------------------------------------------------------- |
| Simulation | Unity 6, C#, Unity Physics, ML-Agents 4.x                                  |
| Learning   | Python 3.10+, PyTorch, NumPy, Gymnasium-compatible wrappers                |
| Memory     | PostgreSQL + pgvector (STM), Neo4j Community (place graph)                 |
| Data       | Parquet, JSON/JSONL, image sequences, PyTorch checkpoints                  |
| Tooling    | Git LFS, Docker, Unity Test Framework, Pytest, Ruff, Black, GitHub Actions |

## Repository layout

```text
fly-project/
├── unity/          # Unity simulation (Chamber + FlyBoot scenes)
├── python/         # flybrain package, configs, tests, scripts
├── data/           # experiments, checkpoints, exports
├── docker/         # pgvector + Neo4j for mind memory
├── docs/           # architecture, phases, Unity setup
├── scripts/        # build player, EditMode tests, ML-Agents venv
└── .github/        # CI workflows
```

## Quick start

### Prerequisites

- Unity 6 (6000.x) — ML-Agents is listed in `unity/Packages/manifest.json`
- Python 3.10+ for `flybrain`; **Python 3.10** for the ML-Agents launch bridge (`.venv-mlagents`)
- Docker (optional; needed for real pgvector / Neo4j)

### Python setup

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mind]"
```

Optional ML-Agents bridge (Unity launch / live mind against Editor or player):

```bash
# from repo root
./scripts/setup_mlagents_venv.sh
```

### Unity setup (interactive chamber)

1. Open `unity/` in Unity Hub (Unity 6).
2. Wait for package resolution.
3. **Tools → Fly → Setup Chamber Scene** (creates/refreshes `Assets/Scenes/Chamber.unity`).
4. Press **Play**. Use the Game tab **Display** dropdown for overview (1), follow cam (2), or mind log (3).

More detail: [docs/unity-setup.md](docs/unity-setup.md).

### Live mind (main loop)

Start memory services (optional for in-memory tests; required for real STM / place graph):

```bash
docker compose -f docker/docker-compose.mind.yml up -d
```

**Fake env** (no Unity):

```bash
cd python && source .venv/bin/activate
python -m scripts.run_live_mind --fake --steps 80
```

**Unity Editor** (Play `Chamber.unity` with walking demo off, then):

```bash
cd python
../.venv-mlagents/bin/python -m scripts.run_live_mind --editor --continuous --mind-interval 4
# or, after activating the mlagents venv:
# source .venv-mlagents/bin/activate && python -m scripts.run_live_mind --editor --continuous --mind-interval 4
```

Useful variants: `--fake-visual-nav`, `--fake-food-search`, `--fake-explore`, `--fake-locomotion`, `--sync-mind`, `--no-curiosity`. Config: `python/configs/mind.yaml`.

## Test locally

### 1. Python unit tests

```bash
cd python
source .venv/bin/activate
pytest
```

### 2. Phase 0–8 deterministic benchmark (fake envs)

```bash
cd python
export PYTHONHASHSEED=0
python -m scripts.run_phase_benchmark --check --self-check
```

After intentional metric changes: `python -m scripts.run_phase_benchmark --write`.

### 3. Unity EditMode tests

**Close the Unity Editor first** if the project is open.

```bash
./scripts/run_unity_editmode_tests.sh
```

### 4. Unity ↔ Python launch

**Close the Unity Editor first** — batch builds cannot open a locked project.

```bash
./scripts/setup_mlagents_venv.sh   # once
./scripts/build_unity_player.sh    # → builds/FlyChamber.app

source python/.venv-mlagents/bin/activate
cd python
python -m scripts.launch_unity
```

Expect `unity_launch_ok` with `obs_shape=(197,)` and `action_shape=(5,)`.  
Training player uses `FlyBoot.unity`; `Chamber.unity` is the interactive editor scene.

Editor-only (no rebuild): Play `FlyBoot` or `Chamber`, then `python -m scripts.launch_unity --editor`.

### 5. Optional — Fake env smoke / Docker

```bash
cd python && source .venv/bin/activate
python -m scripts.smoke_exchange

# from repo root
docker build -f docker/Dockerfile -t flybrain:latest .
docker run --rm -v "$(pwd)/data:/data" flybrain:latest pytest
```

## Development phases

See [docs/phases.md](docs/phases.md) for exit criteria. Recommended order:

1. Chamber and deterministic experiments ✅
2. Passive fly body ✅
3. Articulated legs and hand-coded walking ✅
4. Sensor recording ✅
5. Multimodal mind pipeline (STM + thinking) ✅
6. Visual navigation ✅
7. Taste and homeostasis ✅
8. Curiosity-driven exploration ✅
9. Learned locomotion (Architecture B) ✅
10. Flight (simplified, then articulated)

Architecture overview, async mind, stimulus interrupt, and place graph: [docs/architecture.md](docs/architecture.md).

## First release target

A walking fly with six articulated legs, eye cameras, foot contact/taste sensors, internal state, sugar/bitter surfaces, synchronized recording, a learned walking policy, food-search / route-home behavior, place memory, and a basic next-state world model.

context for system design here:

https://medium.com/@VinhNguyen

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
with a **Patent Covenant Addendum** — noncommercial use only, and recipients
may not patent ideas, methods, or designs disclosed by or derived from this
software. See [LICENSE](LICENSE).
