# Unity setup

## Requirements

- Unity 6 (6000.x)
- Git LFS (for later model/recording assets)

## Open the project

1. Install Unity Hub and Unity 6.
2. Add the `unity/` folder as a project.
3. Allow packages to resolve. `com.unity.ml-agents` is listed in `Packages/manifest.json`.

This project uses **ML-Agents 4.0.3** with **Inference Engine** (`com.unity.ai.inference`).
Do not use ML-Agents 3.0.0 on Unity 6.5 — it still depends on the removed
`Unity.Sentis` namespace. Prefer 4.0.3+ for Unity 6.5 `GetEntityId` compatibility.

`FlySimulation.Runtime.asmdef` references `Unity.ML-Agents` and defines
`UNITY_MLAGENTS` when the package is present so `FlyAgent` subclasses `Agent`.

## Create the chamber scene

In the Unity **top menu bar** (not the Project window `Assets/Fly` folder):

**Tools → Fly → Setup Chamber Scene**

(or **Fly → Setup Chamber Scene**)

This creates `Assets/Scenes/Chamber.unity` with:

- 0.1×0.05×0.1 m chamber (open ceiling + cutaway near walls)
- Sugar and bitter chemical sources
- BodyScale=1 fly with six legs planted on the floor
- Overview camera + episode manager and recorder

After it runs, open `Assets/Scenes/Chamber.unity` if it is not already open. Select `Chamber` in the Hierarchy and press **F** to frame it.

Press **Play** and watch the **Game** tab (not only Scene view). You should see the grey floor, yellow/purple pads, and dark fly through the cutaway.

If Play mode looks like a solid blue box with a grey square:

**Tools → Fly → Fix Chamber Visibility**

(or rebuild with **Setup Chamber Scene**). Walls need a transparent shader; `Unlit/Color` alone ignores alpha.

## Heuristic / demo controls

With the scene playing and focus in Game view, `WalkingDemoDriver` cycles stand → walk → turn.

ML-Agents `Heuristic` (when connected): Vertical = walk, Horizontal = turn, Space = proboscis.

### Phase 4 live mind (Python actions)

1. On `FlyAgent`, set **Use Walking Demo = false**.
2. **Stop Play** if the Editor is already playing.
3. Start Python first (it listens on **5004**):

```bash
source .venv-mlagents/bin/activate
cd python
python -m scripts.run_live_mind --editor --steps 20
```

4. When the terminal is waiting / shows the connect message, press **Play** in `Chamber.unity`.

If the Unity Console says `Couldn't connect to trainer on port 5004... Will perform inference instead`, the Editor started before Python was listening. **Stop Play**, leave Python running, press **Play** again.

Do **not** use `--port 5005` with `--editor`.
Optional pgvector STM:

```bash
docker compose -f docker/docker-compose.mind.yml up -d
python -m scripts.run_live_mind --fake --pgvector
```

After **Tools → Fly → Setup Chamber Scene**, Phase 5 adds a green **VisualTarget**, floor marks, and dark obstacle pillars. Phase 6 uses existing sugar/bitter sources with hardened proboscis consume.
## Recording output

Episode folders are written under the repo `data/experiments/` directory:

```text
experiment_000001_ep0001/
├── metadata.json
├── environment.json
├── observations.csv
├── actions.csv
├── internal_state.csv
└── events.jsonl
```

Convert CSV → Parquet from Python after collection.

## Batch training build

From the repo root:

```bash
./scripts/build_unity_player.sh
```

Or in the Editor: **Tools → Fly → Build Training Player**.

This writes `builds/FlyChamber.app` (macOS), `FlyChamber.exe` (Windows), or
`FlyChamber.x86_64` (Linux).

## Launch from Python

ML-Agents 4.0.x needs the `mlagents_envs==1.1.0` bridge on **Python 3.10**:

```bash
./scripts/setup_mlagents_venv.sh
source python/.venv-mlagents/bin/activate
cd python && python -m scripts.launch_unity
```

Editor play-mode (no build): press Play on `Chamber.unity`, then:

```bash
python -m scripts.launch_unity --editor
```

Keep a rendered evaluation build (`--graphics`) for inspecting behavior.
