"""Phase 8 exit-criteria constants for curiosity-driven exploration."""

# Gate: suppress explore when survival/hunger above these.
SURVIVAL_RISK_GATE = 0.7
HUNGER_GATE = 0.7
ENERGY_MIN_FOR_EXPLORE = 0.25

# Novelty / progression
GRID_CELL_SIZE = 0.1
FIRST_VISIT_BONUS = 1.0
REPEAT_VISIT_DECAY = 0.85
EMBEDDING_HASH_BINS = 12
CHEM_SIGNATURE_BINS = 8

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
