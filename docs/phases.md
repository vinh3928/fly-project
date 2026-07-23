# Development phases

## Phase 0 — Project foundation ✅ complete

Deliverables: Unity repo layout, Python package, Docker, CI, configuration, ML-Agents stubs,
standalone player build path, Python launch bridge.

Exit criteria:

- [x] Unity launches from Python (build player + `python -m scripts.launch_unity`)
- [x] One observation and one action can be exchanged (FakeFlyEnv smoke test)
- [x] Automated Python tests run in CI

How to verify the Unity launch path locally:

```bash
# Close the Unity Editor first (batch builds cannot share the project lock), then:
./scripts/build_unity_player.sh
# Builds FlyChamber.app from Assets/Scenes/FlyBoot.unity (minimal ML-Agents agent).
# Chamber.unity remains the interactive editor scene.

./scripts/setup_mlagents_venv.sh   # once
source python/.venv-mlagents/bin/activate
cd python && python -m scripts.launch_unity
# Expect: unity_launch_ok  obs_shape=(197,) action_shape=(5,)
```

Editor-only alternative (no binary): Play `Chamber.unity` or `FlyBoot.unity`, then
`python -m scripts.launch_unity --editor`.

## Phase 1 — Passive fly and chamber ✅ complete

Deliverables: controlled chamber, simplified rigidbody body, deterministic seed resets,
physics stability monitoring, episode telemetry (`episode_summary.json`).

Exit criteria:

- [x] Fly spawns at configured pose with cleared velocities (`FlyBodyController.ResetPose`)
- [x] Physics remains stable for automated soak (`Phase1PhysicsSoakTests`, 10s / 2000 steps @ 200 Hz);
      full 10 simulated minutes is the Play-mode target (`Phase1ExitCriteria.StabilitySeconds`)
- [x] Same seed → same obstacle layout (`DeterministicLayout` / `EnvironmentRandomizer`)

How to verify:

```bash
cd python && pytest
./scripts/run_unity_editmode_tests.sh
```

## Phase 2 — Articulated walking body ✅ complete

Deliverables: six legs / 18 joints (animated tripod gait), raycast foot contacts, balance
controller, hand-coded standing + walking + turning via root locomotion forces.

Note: joints are driven as kinematic targets with root-body forces for locomotion (full
`ArticulationBody` ground-reaction walking is deferred to later refinement / Phase 4).

Exit criteria:

- [x] Stand without falling (automated 10s soak; Play-mode target 30s — `Phase2ExitCriteria.StandSeconds`)
- [x] Walk forward (`Phase2LocomotionTests.WalkForward_DisplacesAlongForwardAxis`)
- [x] Turn (`Phase2LocomotionTests.Turn_ChangesYaw`)
- [x] Recover from minor disturbance (`Phase2LocomotionTests.Disturbance_RecoversTowardUpright`)

How to verify:

```bash
cd python && pytest
./scripts/run_unity_editmode_tests.sh

# Visual: Unity → Tools → Fly → Setup Chamber Scene → Play
# WalkingDemoDriver cycles stand → walk → turn
```

## Phase 3 — Sensory recording ✅ complete

Deliverables: synchronized multimodal episode logs (CSV/JSONL/PNG), taste gated to contact,
vision frames with eye/body orientation metadata, Python `EpisodeReplay` loader.

Exit criteria:

- [x] Every observation timestamped (`sim_time` / `timestamp` on CSV + JSONL streams)
- [x] Recorded vision matches fly orientation (`vision/manifest.jsonl` body/eye forwards)
- [x] Taste only during valid contact (`TasteGating` + gated `taste.csv`)
- [x] Episodes replayable from logs (`flybrain.recording.EpisodeReplay`)

How to verify:

```bash
cd python && pytest
./scripts/run_unity_editmode_tests.sh

# Visual / data: Unity → Tools → Fly → Setup Chamber Scene → Play
# Episode folders under data/experiments/<experiment_id>_ep####/
```

## Phase 4 — Multimodal mind pipeline ✅ complete

Deliverables: live sensory → signals → affective embeddings → pgvector short-term
memory → always-on thinking service (reflection + present/future stub + devil’s-advocate
LLM) that emits high-level motor intents back to Unity.

Exit criteria:

- [x] Live (or FakeFlyEnv) obs produces a stored STM embedding with affect factors
- [x] Thinking service consumes STM and writes a refined thought with devil’s-advocate critique
- [x] Thinking emits a valid 5-d high-level action for ≥1 env step
- [x] Unit/integration tests for signals, affect, memory, thinking, live mind smoke

How to verify:

```bash
# Start pgvector (optional for InMemory tests; required for real STM)
docker compose -f docker/docker-compose.mind.yml up -d

cd python && pip install -e '.[mind]' && pytest
python -m scripts.run_live_mind --fake   # smoke without Unity
# Live: Unity Play Chamber.unity with useWalkingDemo=false, then:
# python -m scripts.run_live_mind --editor
```

## Phase 5 — Visual navigation ✅ complete

Deliverables: recurrent vision head (offline synthetic training), FakeVisualNavEnv,
mind-pipeline vision cues, Unity VisualTarget + visible obstacles + eval metrics.

Exit criteria:

- [x] Agent avoids walls (FakeVisualNavEnv wall-hit budget with vision cues)
- [x] Agent navigates toward a visible target (distance reduction / success)
- [x] Agent operates without direct target coordinates in the observation vector

How to verify:

```bash
cd python && pytest
python -m scripts.train_vision_head --steps 100
python -m scripts.run_live_mind --fake-visual-nav --steps 80

# Unity: Tools → Fly → Setup Chamber Scene (VisualTarget + dark obstacles)
./scripts/run_unity_editmode_tests.sh
```

## Phase 6 — Taste and homeostasis ✅ complete

Deliverables: FakeFoodSearchEnv, homeostatic reward terms, mind-pipeline food-search
intents, Unity consume hardening (gated taste + extension), FoodSearchMetrics.

Exit criteria:

- [x] Hungry agents search for food
- [x] Agents distinguish sugar from bitter
- [x] Agents consume sugar using the proboscis
- [x] Behavior changes based on internal state

How to verify:

```bash
cd python && pytest
python -m scripts.run_live_mind --fake-food-search --steps 80
./scripts/run_unity_editmode_tests.sh
```

## Phase 7 — Two-layer next-state + low-load retrospective ✅ complete

Deliverables: outcome labeling + life-experience priority weights (survival → altruism +
spiritual growth), Layer-1 latent-stimulus scan, Layer-2 desirable/undesirable reflection,
low-load retrospective, `NextStateWorldModel` rollout + synthetic train script, live mind wiring.

Exit criteria:

- [x] Layer 1 picks an action that improves scored state (survival-dominant for young flies;
      altruism/growth terms move ranking for high-experience flies)
- [x] Layer 2 changes or confirms Layer 1 using labeled past outcomes
- [x] Retrospective runs only under low load and writes outcome labels / thoughts
- [x] World-model one-step prediction loss improves vs untrained baseline on synthetic replay
- [x] Phase 4–6 tests still pass; Phase 7 unit tests cover weights, labeling, L1/L2, low-load

How to verify:

```bash
cd python && pytest
python -m scripts.train_world_model --steps 100
python -m scripts.run_live_mind --fake --steps 80
./scripts/run_unity_editmode_tests.sh
```

## Phase 8 — Curiosity-driven exploration ✅ complete

Goal: make the fly *want* to explore. Survival and feeding stay primary under threat or
hunger; when load is moderate, novelty and learning progress pull behavior toward new
places, stimuli, and action patterns — light internal “exploration XP,” not a UI.

### Includes (not limited to)

**Reward / progression (gamify new experiences)**

- Intrinsic reward from prediction error + learning progress (`CuriosityModule`)
- Novelty bonuses for first encounters: chamber cells, chem signatures, landmark events
- Soft exploration XP / streaks that bias Layer-1 explore ranking and motor intent when
  energy/survival allow
- Diminishing returns so farming the same cell/event stops paying
- Homeostatic gates so curiosity never overrides critical survival/feeding

**World-model & surprise**

- Live prediction-error signal from `CuriosityModule` on STM embedding transitions
- Learning-progress via error EMA inside the module

**Memory & thinking hooks**

- Curiosity / novelty / explore_bias injected into thinking signals + affect bump
- Composer + Layer-1 explore-candidate boost when gated curiosity is high
- Optional `explore_policy_action` blend on `FakeExplorationEnv`

**Environment / metrics**

- `FakeExplorationEnv` (high energy, sparse sugar, landmarks, coverage telemetry)
- Live mind reports coverage, unique_cells, exploration_xp, curiosity_reward_mean

### Deliverables

- `NoveltyTracker`, `CuriosityDrive`, `curiosity_gate` under `flybrain.curiosity`
- Live mind wiring + `mind.yaml` / `default.yaml` knobs
- `--fake-explore` smoke + Phase 8 unit tests

### Exit criteria

- [x] Under low hunger/survival risk, curious agents reach ≥ baseline spatial coverage
- [x] First-visit / events yield intrinsic novelty; repeats decay
- [x] Curiosity gated: high survival risk or critical hunger suppresses explore bias
- [x] Hungry gate closes so Phase 6 feeding remains dominant when needed
- [x] Unit/integration tests cover novelty, gates, diminishing returns, live-mind explore
- [x] Config documents curiosity weights / gates (`configs/mind.yaml`)

### How to verify

```bash
cd python && pytest
python -m scripts.run_live_mind --fake-explore --steps 120
# Baseline comparison:
python -m scripts.run_live_mind --fake-explore --no-curiosity --steps 120
./scripts/run_unity_editmode_tests.sh
```

## Phase 9 — Learned locomotion (deferred from original Phase 4)

- ML-Agents walking env, locomotion reward, checkpoints

## Phase 10 — Simplified flight

## Phase 11 — Articulated-wing flight

Flight begins only after walking, sensing, recording, and learning are stable.

## Phase 0–8 deterministic benchmark (fake envs)

Repeatable Python scoreboard (seed 42). Does **not** include Unity EditMode/Play.

```bash
cd python
export PYTHONHASHSEED=0
python -m scripts.run_phase_benchmark --check --self-check
```

- Implementation: `flybrain/benchmark.py`, `flybrain/seeding.py`
- Snapshot: `benchmarks/expected_seed42.json`
- Update snapshot after intentional metric changes: `python -m scripts.run_phase_benchmark --write`
