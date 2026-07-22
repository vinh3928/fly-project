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
