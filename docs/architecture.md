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

## Learning description

The project combines:

1. Self-supervised representation learning (next-state prediction)
2. Intrinsic-motivation reinforcement learning (curiosity / learning progress)
3. Homeostatic objectives (energy, hydration, damage, effort, stability)

It is **not** purely unsupervised learning.

## Vision representation follow-up

The current 3×3 eye-patch representation is intentionally minimal. Each square
stores only five features: mean R/G/B, luminance standard deviation, and one
vertical-edge statistic. Across both eyes this produces the 90-dimensional STM
`patch_vec` (`2 eyes × 9 squares × 5 features`).

When revisiting the eye/vision architecture, increase the per-square feature
extraction before treating this representation as biologically or perceptually
rich. Candidate additions include:

- horizontal, vertical, and diagonal edge/orientation responses
- local contrast, gradients, and multi-scale spatial-frequency features
- hue, saturation, opponent-color, and chromatic-gradient features
- motion/optic-flow direction and magnitude
- texture statistics, temporal change, and flicker
- learned per-patch features trained with contrastive or predictive objectives

Preserve independent left/right and per-square identity when expanding the
vector, and migrate the pgvector `patch_vec` dimension and existing STM rows
together.
