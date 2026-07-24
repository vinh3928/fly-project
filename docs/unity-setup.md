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

- 0.5×0.25×0.5 m chamber (open ceiling + cutaway near walls)
- Sugar and bitter chemical sources
- BodyScale=1 fly with six legs planted on the floor
- Overview camera + episode manager and recorder

After it runs, open `Assets/Scenes/Chamber.unity` if it is not already open. Select `Chamber` in the Hierarchy and press **F** to frame it.

Press **Play** and watch the **Game** tab (not only Scene view). You should see the grey floor, yellow/purple pads, and dark fly through the cutaway.

**Fly eye view (Game tab):** On Play, bottom-corner PIP panels show Left/Right eye
RenderTextures in the **Game** view (Canvas overlay). Press **V** to toggle.
Do not expect selecting `LeftEye` in the Hierarchy to change the Game camera — those
cameras stay disabled on purpose. Scene/Simulator camera previews are separate from Game view.

If Play mode looks like a solid blue box with a grey square:

**Tools → Fly → Fix Chamber Visibility**

(or rebuild with **Setup Chamber Scene**). Walls need a transparent shader; `Unlit/Color` alone ignores alpha.

## Heuristic / demo controls

With the scene playing and focus in Game view, `WalkingDemoDriver` cycles stand → walk → turn.

ML-Agents `Heuristic` (when connected): Vertical = walk, Horizontal = turn, Space = proboscis.

### Phase 4 live mind (Python actions)

1. On `FlyAgent`, leave **Use Walking Demo = false** (Chamber scene default for live mind).
2. **Stop Play** in Unity (required).
3. Start Python (**binds port 5004 before loading Torch**):

```bash
source .venv-mlagents/bin/activate
cd python
python -m scripts.run_live_mind --editor --continuous --mind-interval 40
```

4. Wait until the terminal prints **`Listening on port 5004`** (from ML-Agents).
5. **Then** press **Play** in `Chamber.unity` (do **not** wait for `unity_env_ready` first — that appears only *after* Unity connects).

Unity Console must say **`connected to Python trainer`**. If it says **`NO Python trainer`** or
`Couldn't connect to trainer... inference instead`, Stop Play and press Play again while Python
is still Listening (or restart the Unity Editor if Academy got stuck in inference).

Eye-frame wall hazards (Phase 5): on Play, `FlyEyeMlAgentsBridge` registers LeftEye/RightEye
camera sensors. Live logs should include `wall=…` when vision cues are active; approaching a
wall should raise `wall` and bias the mind toward turn/stop instead of pure `explore`.

**Where are the live mind logs?**

| What | Where |
|------|--------|
| Per-step lines (`[live 0000 MIND*] …`) | **Python terminal** and `python/logs/live_mind.log` |
| Full end snapshot (`live_mind_ok` JSON) | Terminal after Ctrl+C / finish, and `python/logs/live_mind_last.json` |
| Unity Console | Does **not** show these |

```bash
# watch steps live
tail -f logs/live_mind.log

# open last full mind dump
open logs/live_mind_last.json   # or: less / cat / jq

# richer per-MIND* DETAIL JSON lines
python -m scripts.run_live_mind --editor --continuous --mind-interval 40 -v
```

Use `--quiet` only if you want to silence per-step lines.

To verify Unity locomotion alone: enable the `WalkingDemoDriver` component **and** set
**Use Walking Demo = true**, then Play without Python — stand → walk → turn.

If the Unity Console says `Couldn't connect to trainer on port 5004... Will perform inference instead`, the Editor started before Python was listening. **Stop Play**, leave Python running, press **Play** again.

Do **not** use `--port 5005` with `--editor`.

`WalkingDemoDriver` auto-disables when **Use Walking Demo** is false so it cannot overwrite Python motor commands.

Exploration smoke (Phase 8):

```bash
python -m scripts.run_live_mind --fake-explore --steps 120
python -m scripts.run_live_mind --fake-explore --no-curiosity --steps 120
```

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
