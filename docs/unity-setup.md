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

- 0.5×0.25×0.5 m chamber; walls and ceiling are opaque to the fly's eyes but culled
  from the overview camera (layer 9 "Environment")
- Walls are hard: the fly has solid thorax, head and abdomen spheres, and every walk/turn step
  is swept against layer 9 (`BodyClearance`) so no sphere enters a wall. The eye near clip
  (0.6 mm) stays inside the head sphere at the 4 mm eye spacing (±2° outward yaw), so a
  wall touching the head is never clipped away
- Exit tunnel through the +Z wall: 250 mm wide (half the box), open from floor to ceiling,
  80 mm deep (`ExperimentConfiguration.Tunnel*`; set `TunnelFullHeight = false` to use
  `TunnelHeight` instead; geometry in `TunnelLayout`)
- Outside world at the fly's real scale (`OutsideWorldLayout` / `OutsideWorldBuilder`):
  a meadow of 3–12 cm grass and 12–35 cm flowers in front of the exit, 5–12 m trees
  6–30 m away, and a town 40–70 m out past the tunnel
- Sugar and bitter chemical sources
- BodyScale=1 fly with six legs planted on the floor
- Overview camera + episode manager and recorder

**Asset-pack slot:** drop prefabs into `Assets/OutsideWorld/Flowers`, `Trees` or `Buildings`,
then run **Tools → Fly → Rebuild Outside World**. Any folder with prefabs replaces that
feature's primitives; each prefab is scaled to the layout height and stood on the ground.
The scene uses the Built-in render pipeline, so pick Built-in versions of packs (URP/HDRP
materials render pink).

After it runs, open `Assets/Scenes/Chamber.unity` if it is not already open. Select `Chamber` in the Hierarchy and press **F** to frame it.

Press **Play** and watch the **Game** tab (not only Scene view). You should see the grey floor, yellow/purple pads, and dark fly through the cutaway.

**Fly profiles:** on Play, a **Choose your fly** panel appears on Display 1. Pick **Fly 1**
(current default), another existing fly, or **+ Create new fly** (Fly 2, 3, …). The choice is
saved to `data/fly_profiles.json` (and StreamingAssets) so Python live-mind uses the same
`fly_id` for STM / place-graph memory. From Python you can also pass `--fly 2`, `--new-fly Scout`,
or `--select-fly`.

**Three displays:** Display 1 is the fixed chamber overview (cutaway). Display 2 is
`FlyFollowCamera` — a third-person chase view that always centers on the fly.
Display 3 is `MindLogDisplay` — human-readable **mind** and **deep thought** panels
(plan, energy/hunger, route, thought, devil) streamed from Python over a SideChannel.
When a stimulus interrupt arms, a **Father advocate** dialog also appears on Display 3:
the question about the novel situation, a text box for your teaching reply, **Ask LLM**,
or **Skip**. The fly’s motor hard-stops until you reply (or **60 s** elapses → auto LLM).
The reply is a **one-shot lesson**: stored on the current Place as
`(:Place)-[:HAS_LESSON]->(:Lesson)`, structured as `approach` / `avoid` / `taste` / `idle`
(+ turn bias / confidence), and bound into the motor directive when the freeze clears.
Same place+cue later reuses the lesson without re-asking.
In the Game tab, open the **Display** dropdown and pick **Display 2** / **Display 3**
(or plug in extra monitors; Play activates them when present). Re-run
**Tools → Fly → Fix Chamber Visibility** (or Setup Chamber Scene) if a cam is missing.

**Fly eye view (Game tab):** On Play, bottom-corner PIP panels show Left/Right eye
RenderTextures in the **Game** view (Canvas overlay). Each eye is a **pointy-top
hexagon** (outside pixels masked black) with optional green facet spokes.
Press **V** to toggle. Do not expect selecting `LeftEye` in the Hierarchy to
change the Game camera — those cameras stay disabled on purpose. Scene/Simulator
camera previews are separate from Game view.

If Play mode looks like a solid blue box with a grey square:

**Tools → Fly → Fix Chamber Visibility**

(or rebuild with **Setup Chamber Scene**). It moves walls onto the wall layer the overview camera culls.
In the Scene view, hide layer "Environment" (Layers dropdown) to look inside the box.

## Vision model (walls and the way out)

Without a checkpoint the live mind uses brightness heuristics, which read the dark meadow
beyond the tunnel as a wall. To train the vision head on the real scene:

1. Open the Chamber scene (not in Play mode) and run **Tools → Fly → Record Vision Dataset**.
   It poses the fly at 6000 sampled spots (open interior, by walls, facing the exit, in the
   tunnel, outside), renders both eyes and writes `data/vision_unity/` with ray-cast labels
   (`EyeRayLabels`: wall hazard, free-space turn, opening bearing and strength).
2. `cd python && .venv/bin/python -m scripts.train_vision_head --unity`
   (settings under `unity_train` in `configs/visual_nav.yaml`; writes
   `data/checkpoints/vision_head.pt` and prints validation errors per cue).
3. Run the live mind: it loads the checkpoint automatically. The log shows
   `open=<strength>@<bearing>`; roaming plans turn toward a seen opening
   (`cue_turn`, after visual novelty, before wall avoidance).

Re-record after changing the scene (tunnel size, colours, outside world).

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

**Route path debug light:** when `route_home` is active, Python sends the place-graph polyline
over an ML-Agents SideChannel (`RoutePathDebugChannel`); Unity draws a ~1px cyan
`LineRenderer` (`RoutePathVisualizer` on `Fly`). Hungry routing treats each place node as a
world-XYZ checkpoint back to the scent (advance on proximity even if place-id rematch
creates a new outdoor node).
If an older scene lacks the component, run **Tools → Fly → Ensure Route Path Visualizer**
(also attached by Setup Chamber Scene / Fix Chamber Visibility).

**Latent place-graph map:** each live-mind step also sends a version=2 snapshot of remembered
places (world XYZ). `PlaceGraphVisualizer` shows one text label on the current/home place;
the cyan ~1px route polyline is `RoutePathVisualizer` only (debug objects use the Fly layer
so eye cameras do not render them — avoids hitching). Graph snapshots are throttled from
Python (route/current/home only; no full Neo4j edge dump each step).
Label tags: `*` current, `H` home, `R` on route, `s=` sugar smell when &gt; 0.05.

Use **Tools → Fly → Ensure Place Graph Visualizer** on older scenes. Watch whether plume vs
puddle labels are where you expect while debugging route/food handoff.

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

Mind databases:

- pgvector STM: `fly_mind.short_term_memory`, port `5433`
- Neo4j place graph: Browser `http://127.0.0.1:7474`, Bolt port `7687`
- Neo4j skill cortex: Browser `http://127.0.0.1:7475`, Bolt port `7688`

```bash
docker compose -f docker/docker-compose.mind.yml up -d
# Place Neo4j: neo4j / flymindgraph  (override: NEO4J_PASSWORD)
# Skill Neo4j: neo4j / flymindskill  (override: NEO4J_SKILL_PASSWORD)
# mind.yaml defaults to backend: pgvector; override with -- or database.backend: in_memory
python -m scripts.run_live_mind --fake
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
