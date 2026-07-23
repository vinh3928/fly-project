# Fly Embodied-Learning Simulation

A controlled 3D simulation in which a virtual fly walks, senses, records synchronized multimodal data, and learns via curiosity and homeostatic reinforcement learning.

This repository is an **embodied fly-inspired learning simulation**, not a biologically exact digital fly.

## Status

**Phases 0–7 are complete** (foundation through two-layer next-state / low-load retrospective).

- Unity 6 project with chamber, fly body, sensors, motors, and ML-Agents agent wiring
- Deterministic seed resets, physics stability monitor, episode telemetry
- Balance + tripod gait + root locomotion (stand / walk / turn / recover)
- Synchronized multimodal recording (joints/contacts/taste/odor/pose/vision) + Python replay
- Multimodal mind pipeline: affective embeddings → pgvector STM → thinking (+ devil’s advocate)
- Hybrid visual navigation: recurrent vision head + FakeVisualNavEnv + Unity VisualTarget arena
- Taste/homeostasis food-search: FakeFoodSearchEnv + consume/aversion via mind pipeline
- Two-layer present/future (priority scan + outcome reflection) with low-load retrospective
- Python `flybrain` package with encoders, world model, curiosity, and policy modules
- FakeFlyEnv smoke test + CI
- Standalone player build (`./scripts/build_unity_player.sh`) and Python launch
  (`python -m scripts.launch_unity`)

Next: Phase 9 (learned locomotion). Phase 8 curiosity-driven exploration is complete.

Flight is intentionally deferred until walking, sensing, recording, and learning are stable.

## Technology stack

| Layer | Tools |
| --- | --- |
| Simulation | Unity 6, C#, Unity Physics, ArticulationBody, ML-Agents |
| Learning | Python 3.11+, PyTorch, NumPy, Gymnasium-compatible wrappers |
| Data | Parquet, JSON/JSONL, MP4/image sequences, PyTorch checkpoints |
| Tooling | Git LFS, Docker, Unity Test Framework, Pytest, Ruff, Black, GitHub Actions |

## Repository layout

```text
fly-project/
├── unity/          # Unity simulation
├── python/         # flybrain training package
├── data/           # experiments, checkpoints, exports
├── docker/         # training containers
├── docs/           # architecture and phase docs
└── .github/        # CI workflows
```

## Quick start

### Prerequisites

- Unity 6 (6000.x) — ML-Agents package is listed in `unity/Packages/manifest.json`
- Python 3.10+ (main package); **Python 3.10** for the ML-Agents launch bridge
- Docker (optional, for reproducible training)

### Python setup

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Unity setup (interactive chamber)

1. Open `unity/` in Unity Hub (Unity 6).
2. Wait for package resolution.
3. **Tools → Fly → Setup Chamber Scene** (creates/refreshes `Assets/Scenes/Chamber.unity`).
4. Press **Play** to view the chamber, fly, and legs.

## Test locally

### 1. Python unit tests

```bash
cd python
source .venv/bin/activate
pytest
```

Expect all tests to pass.

### 2. Unity EditMode tests (Phase 1)

**Close the Unity Editor first** if the project is open.

```bash
./scripts/run_unity_editmode_tests.sh
```

### 3. Unity ↔ Python launch (Phase 0)

**Close the Unity Editor first** — batch builds cannot open a project that is already open.

```bash
cd /path/to/fly-project

# One-time: ML-Agents Python 3.10 env
./scripts/setup_mlagents_venv.sh

# Build standalone player → builds/FlyChamber.app
./scripts/build_unity_player.sh

# Launch player and exchange one observation/action
source python/.venv-mlagents/bin/activate
cd python
python -m scripts.launch_unity
```

Expect:

```text
unity_launch_ok
obs_shape=(197,) action_shape=(5,)
```

The training player uses `Assets/Scenes/FlyBoot.unity` (minimal ML-Agents agent).
`Chamber.unity` remains the interactive editor scene.

### 3. Optional — Editor play mode (no rebuild)

1. Open `unity/` in Unity Hub.
2. Open `Assets/Scenes/FlyBoot.unity` (or `Chamber.unity` for visuals).
3. Press **Play**.
4. In another terminal:

```bash
source python/.venv-mlagents/bin/activate
cd python
python -m scripts.launch_unity --editor
```

### 4. Optional — Fake env smoke (no Unity)

```bash
cd python
source .venv/bin/activate
python -m scripts.smoke_exchange
```

### 5. Optional — Docker

```bash
docker build -f docker/Dockerfile -t flybrain:latest .
docker run --rm -v "$(pwd)/data:/data" flybrain:latest pytest
```

More detail: [docs/unity-setup.md](docs/unity-setup.md), [docs/phases.md](docs/phases.md).

## Development phases

See [docs/phases.md](docs/phases.md) for exit criteria. Recommended order:

1. Chamber and deterministic experiments
2. Passive fly body
3. Articulated legs and hand-coded walking
4. Sensor recording
5. Multimodal mind pipeline (STM + thinking)
6. Visual navigation (hybrid vision head + arena)
7. Taste and homeostasis (food-search)
8. Curiosity-driven exploration ✅ (novelty / learning-progress rewards; see docs/phases.md)
9. Learned locomotion (deferred)
10. Flight (simplified, then articulated)

## First release target

A walking fly with six articulated legs, eye cameras, foot contact/taste sensors, internal state, sugar/bitter surfaces, synchronized recording, a learned walking policy, food-search behavior, and a basic next-state world model.

## License

MIT — see [LICENSE](LICENSE).
