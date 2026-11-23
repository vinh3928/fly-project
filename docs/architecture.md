# Architecture overview

```text
Unity simulation
│
├── Environment system (chamber, chemistry, lighting)
├── Fly body (rigidbody + six articulated legs)
├── Sensor system (vision, proprioception, touch, taste, internal)
├── Motor system (high-level gait → joint targets)
├── Experiment system (episodes, seeding, recording)
└── Python learning service (encoders, world model, curiosity, policy)
```

## Simulation stack

- Unity 6 with **URP** (Universal Render Pipeline) as the active render pipeline
- Physics timestep 200 Hz; agent decisions ~20 Hz

## Data flow

```text
Sense → Decide → Act → Change environment → Record → Learn
```

## Timing

| Domain | Rate |
| --- | --- |
| Unity physics | 200 Hz (`Fixed Timestep` 0.005) |
| Agent decision | 20 Hz |
| Vision capture | 30 Hz |
| Sensor recording | 50 Hz |
| Training updates | Asynchronous |
| Live mind (Python) | Asynchronous (`motor.async_mind`) |

### Asynchronous mind

The motor acts on every Unity decision and never waits for the mind. Each
thought runs on one background worker (`flybrain/pipeline/async_mind.py`) from
a snapshot of the senses taken on the motor loop. When it finishes, its plan is
committed to the motor on the next step. A new thought starts once the worker
is idle and at least `async_min_steps` steps have passed.

New stimuli are handled by deep thinking (pause and reframe):

- the motor-side stimulus interrupt arms on a novel cue and holds the fly still
  (`idle`)
- the in-flight thought is invalidated; it stops at its next LLM boundary and
  its plan, made before the stimulus, never reaches the motor
- the worker then runs deep thoughts back to back on the heightened senses
- the freeze clears after `deep_thought_steps` *and* at least one reframed deep
  thought has committed (capped by `async_reframe_max_steps`), or earlier on a
  confirmed judgement unlock

The worker and motor loop share state under a FIFO lock that the worker drops
around LLM calls; short-term memory is serialized separately. Tests, the
benchmark and `--sync-mind` / `--mind-interval N` use the synchronous loop, which
thinks inline every `mind_interval` steps.

## Learning description

The project combines:

1. Self-supervised representation learning (next-state prediction)
2. Intrinsic-motivation reinforcement learning (curiosity / learning progress)
3. Homeostatic objectives (energy, hydration, damage, effort, stability)

It is **not** purely unsupervised learning.

## Vision representation follow-up

Each eye uses a **pointy-top hexagonal aperture** inscribed in the 64×64 RGB
frame. Pixels outside the hex are masked to zero. The hex is subdivided into
**6 triangles** by diagonals through the center (center → each vertex). Each
triangle stores five features: mean R/G/B, luminance standard deviation, and one
vertical-edge statistic. Across both eyes this produces the 60-dimensional STM
`patch_vec` (`2 eyes × 6 facets × 5 features`). Older 90-d (3×3 square) STM /
place rows should be treated as cold-start after upgrade (pad/truncate).

When revisiting the eye/vision architecture, increase the per-facet feature
extraction before treating this representation as biologically or perceptually
rich. Candidate additions include:

- horizontal, vertical, and diagonal edge/orientation responses
- local contrast, gradients, and multi-scale spatial-frequency features
- hue, saturation, opponent-color, and chromatic-gradient features
- motion/optic-flow direction and magnitude
- texture statistics, temporal change, and flicker
- learned per-facet features trained with contrastive or predictive objectives

Preserve independent left/right and per-facet identity when expanding the
vector, and migrate the pgvector `patch_vec` dimension and existing STM rows
together.

## Stimulus interrupt history (joint vision + smell + sensors)

Interrupt Architecture-B arms on a candidate cue under soft load, then runs an
**80-step deep-thought freeze** (`deep_thought_steps`). While the freeze is
active:

- mind pushes an ``idle`` motor directive (act-state bus); motor holds `STOP_GOAL`
  every step (no locomotion)
- scheduled mind ticks (every `deep_thought_mind_interval`, default 4) run a
  deep-thought stub: senses are amplified by `deep_thought_gain` (default ×10)
  for the devil LLM / mind view only — including antenna `odor` /
  `odor_left` / `odor_right`; STM and embeddings stay on the unscaled arm-time
  vectors
- each deep mind tick runs judgement; repeated devil risks accumulate into a
  **hazard conditioning** memory block injected into the next draft
- once a hazard is **acknowledged**, it stays out of the conditioning memory
  block / Present for the rest of the freeze (and until a later non-deep
  judgement cycle)
- **early unlock**: when `judgement_confirmations` consecutive deep judgements
  agree on the same `accept` or `acknowledge` decision+risk, the freeze clears
  early; `accept` installs the blended motor goal so motion resumes conditioned
- if the countdown ends without confirmation, the last accept blend (if any)
  is still installed on clear
- **conditioning STM**: hazard entries are snapshotted into short-term memory
  (`signals.conditioning` JSON on checkpoint / mind-tick rows) and rehydrated
  at the start of `run()` from the newest matching STM row

### Motor act-state (mind push)

Architecture B motor control is a small act-state machine above `MotorLayer`:

| Mode | Meaning | Motor behavior |
|------|---------|----------------|
| `act_anything` | Execute an L1 plan (explore, forward, turn_*, …) | Hold mind goal; `motor.act` each tick |
| `still` | Intentional non-action (`stop` plan) | Hold `STOP_GOAL` |
| `idle` | Deep processing (stimulus interrupt) | Hold `STOP_GOAL`; deep-thought mind cadence |

When a mind think completes, live mind **pushes** a `MotorDirective` (mode + goal +
plan name). Between pushes, `process_motor_only` only tracks that directive — no
per-tick embed, curiosity, place map, or explore replan. Stimulus interrupt arms
by pushing `idle` on the same bus (mind interrupting motor), not a parallel freeze
channel.

Cues include contact taste, visual novelty / new stimulus, and **STM-first smell
novelty**: a non-zero antenna `smell_vec` that is unfamiliar versus recent STM
smell neighbors opens `smell_novelty` on its own.

**Passive smell (cheap gate).** The STM smell probe is a pgvector query, so
running it every step dominated step latency. `PassiveSmellGate` (a thin
wrapper over the generic `ModalChangeGate`) samples the antenna vector every
`passive_smell_interval` steps, compares it in memory against the last stored
reading, and never touches STM. Unchanged samples are **L1 hits** — the
expensive path stays closed. The expensive probe only runs when the passive
reading changed by `passive_smell_change_min`, vision saw novelty, or
`passive_smell_refresh_ms` (default 500 ms) elapsed since the last probe.
`ModalChangeGate` is modality-agnostic (default `ttl_seconds=10`): the same
sample→diff→stay-L1 pattern can gate vision patches, sensors, or smell.

History is one joint embedding per interrupt row (saved when the freeze clears):

- `patch_vec` (60-d): stereo hexagonal-eye triangular facet features
- `smell_vec` (16-d): left+right antenna chemistry (`2 × 8` channels)
- `sensors_vec` (192-d): body + chemical encoder features
- `joint_vec` (268-d): `L2(concat(L2(patch), L2(smell), L2(sensors)))`

**L1 familiarity (current):** neighbor retrieval uses whole-vector cosine on
`joint_vec` (top-k). Then only the column that triggered the cue is compared
against that neighbor set (`patches` for visual cues, `smell` for
`smell_novelty`, `sensors` reserved for future). A history hit requires best
column cosine ≥ that column's match min. L2 whole-joint familiarity is deferred.
Old rows without `joint_vec` are skipped (migration-safe). After clear, a
`cooldown_steps` window blocks re-arming.

## Spatial place graph (mental map)

Location memory is a **graph**, not conditioning string keys. A **place node** is
the fly's picture of being in the chamber: a 60-d `view_vec` derived from stereo
`patch_vec`. Because patch features are non-negative, identity uses
**mean-centered + L2-normalized** vectors (centered cosine / Pearson); raw
cosine collapses unrelated views near 0.99. Default `match_min` is `0.97`.
Looking up a node shows chamber details — today that
means four linked **Wall nodes** (`north` / `east` / `south` / `west`) with
per-wall familiarity fields:

```text
(:Place {place_id, fly_id, view_vec})
    -[:HAS_WALL {direction}]->
(:Wall {direction, familiarity, last_seen_step, notes})
```

The persistent backend is **Neo4j Community**. It creates unique constraints
for place/wall IDs, an index for `(fly_id, updated_at)`, and a cosine vector
index on `Place.view_vec` for fast recognition. Place-to-place `ADJACENT`
edges are written whenever the fly transitions `from_place → current_place`.
Matching is **context-aware**: stay on the same place if the view still
matches; otherwise prefer a lookalike already linked by `ADJACENT` from the
previous place; otherwise **create** a new Place even if a visual twin exists
(identical rooms reached via different references stay distinct). Python API:
`InMemoryPlaceGraph` / `Neo4jPlaceGraph` / `open_place_graph` in
`flybrain.memory.place_graph`.

Live mind upserts the current stereo `patch_vec` into the place graph each step.
The first sighting creates a Place with four Wall details. **Sensor + smell
readings are correlated onto that vision Place** (not separate place keys):
`props.label` (e.g. `puddle`), `props.senses.bitter.value = 0.49`, and optional
`props.smell_vec`. Neo4j also links `(:Place)-[:HAS_SENSE]->(:Sense)`.
Father-advocate teachings attach as `(:Place)-[:HAS_LESSON]->(:Lesson)`
(cue / question / answer / risk plus structured `action`, `turn_bias`,
`confidence`) so revisiting a place can reuse the lesson without re-asking.
On interrupt clear the structured fields blend into the motor directive bus
(`approach` / `avoid` / `taste` / `idle`) so the body obeys the teaching, not
only the mind-panel text. `ModalChangeGate` (10s / per-step sample-diff) keeps
unchanged smell/sensor snapshots on L1 so Neo4j is not rewritten every tick.

**Sensor feedback** (`MappedSensorFeedback`) is where decay lives: raw
eye/antenna/taste transducers stay honest (`*_raw`), but once an item is mapped
on the matched Place, the *reported* channel (`wall_hazard`, `odor`, `bitter`, …)
is scaled by `max(floor, 1 - strength * familiarity)` before affect and thinking
run.

Neo4j traffic is buffered by a **two-layer place cache** (`CachedPlaceGraph`):

- **L1 (10s working memory):** RAM + disk mirror under `logs/place_cache/`.
  On an L1 match the pipeline **stops** — update L1 only, bump familiarity /
  decay sensors from that node, **no L2 or Neo4j round-trip**.
- **L2 (24h short-term):** used only after L1 miss. On L2 hit (or Neo4j fill),
  the place is written to **L1 + L2 + Neo4j**. L2 fully resets every 24h.
- Dirty L1 entries flush to L2 + Neo4j when they age out of the 10s window
  (or on `close()` / `flush()`).

Start memory databases (pgvector + place Neo4j + skill Neo4j):

```bash
docker compose -f docker/docker-compose.mind.yml up -d
```

Place-graph Neo4j Browser is at `http://127.0.0.1:7474`; Bolt is
`bolt://127.0.0.1:7687`. Development login defaults to
`neo4j` / `flymindgraph`; set `NEO4J_PASSWORD` to override it.

### Skill cortex (separate Neo4j partition)

Community Edition cannot host two logical DBs on one server, so discovered
motor programs live on a **second** Neo4j container:

| | Place graph | Skill cortex |
|---|---|---|
| Compose service | `neo4j` | `neo4j-skill` |
| Browser | `http://127.0.0.1:7474` | `http://127.0.0.1:7475` |
| Bolt | `7687` | `7688` |
| Auth default | `neo4j` / `flymindgraph` | `neo4j` / `flymindskill` |
| Env override | `NEO4J_PASSWORD` | `NEO4J_SKILL_PASSWORD` |

Graph shape: `(:Skill)-[:HAS_STEP]->(:SkillStep)` with optional
`(:Skill)-[:DISCOVERED_AT]->(:Context)` (`place_id` is a soft string only —
no cross-cluster edges). Status flows `candidate` → `promoted`.

**Discovery:** low-load motor babbling samples short timed 5-d goal sequences as
open combos of primitives (forward / turn / pause / extend / crouch / …) —
anonymous `candidate_move_*` names, no dance/touch hard labels. Candidates are
upserted into the skill cortex. **HITL (Display 4)** Accept/Refuse, **Ask LLM judge**
(auto accept/refuse + name), or **Father** promote. If the Game view is **not** on
Display 4, Ask LLM runs automatically after a short grace (`auto_llm_away_seconds`);
focusing Display 4 keeps the HITL freeze. Promoted skills join L1 as
dynamic candidates and play back via
`SkillExecutor` on the Architecture-B motor bus.
Place `HAS_LESSON.action` / `skill_name` can reference a promoted skill so
one-shot place reuse triggers playback. Config: `skill_cortex` in
`python/configs/mind.yaml`.
