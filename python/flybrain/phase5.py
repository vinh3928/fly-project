"""Phase 5 exit-criteria constants for hybrid visual navigation."""

VISION_SIZE = 64
SUCCESS_RADIUS = 0.08
MAX_WALL_COLLISIONS_PER_EPISODE = 3
MIN_DISTANCE_REDUCTION = 0.05
EVAL_EPISODE_STEPS = 120
WALL_HAZARD_FORWARD_SCALE = 0.2

# Agent vector obs must not contain privileged target world XYZ.
# FakeVisualNavEnv uses DEFAULT_OBS_DIM (197) without appending target coords.
AGENT_OBS_HAS_TARGET_XYZ = False
