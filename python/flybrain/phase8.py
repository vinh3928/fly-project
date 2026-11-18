"""Phase 8 exit-criteria constants for curiosity-driven exploration."""

# Gate: suppress explore when survival/hunger above these.
# Hunger/energy match composer food-seek (HUNGER_SEEK_MIN / ENERGY_SEEK_MAX) so
# curiosity does not keep boosting explore while the fly should forage.
SURVIVAL_RISK_GATE = 0.7
HUNGER_GATE = 0.45
ENERGY_MIN_FOR_EXPLORE = 0.55

# Novelty / progression
GRID_CELL_SIZE = 0.1
FIRST_VISIT_BONUS = 1.0
REPEAT_VISIT_DECAY = 0.85
CHEM_SIGNATURE_BINS = 8

# Experience novelty: distance of the full (normalized) embedding to remembered ones.
EXPERIENCE_DIMS = 256  # fused-sense latent; affect tail excluded
EXPERIENCE_MEMORY_SIZE = 1000
EXPERIENCE_KNN = 5
EXPERIENCE_RECENT_EXCLUDE = 8  # skip the last N memories (previous ticks look identical)
EXPERIENCE_HISTORY_SIZE = 500
EXPERIENCE_PERCENTILE = 95.0  # distance at this percentile maps to novelty 1.0
EXPERIENCE_BASELINE_PERCENTILE = 50.0  # and at this one to 0.0
EXPERIENCE_WARMUP = 20
EXPERIENCE_MIN_STD = 1e-3
EXPERIENCE_NEW_THRESHOLD = 0.5
EXPERIENCE_WEIGHT = 0.3

# Combined intrinsic score weights (then capped).
NOVELTY_WEIGHT = 0.55
PREDICTION_WEIGHT = 0.45
CURIOSITY_REWARD_CAP = 1.0

# Live-mind action bias when gated curiosity is high.
EXPLORE_ACTION_BOOST = 0.45
EXPLORE_CANDIDATE_BONUS = 0.22

# FakeExplorationEnv
EXPLORE_EPISODE_STEPS = 200
EXPLORE_ARENA_HALF = 0.45
EXPLORE_INITIAL_ENERGY = 0.85
EXPLORE_SPARSE_SUGAR_DISTANCE = 0.55
MIN_COVERAGE_CELLS_CURIOUS = 8
