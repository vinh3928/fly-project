"""Phase 9 constants: goal-conditioned motor layer (architecture B)."""

# Mind goals and motor actions share the high-level 5-d interface.
MOTOR_GOAL_DIM = 5
MOTOR_ACTION_DIM = 5

# Vector obs for goal-conditioned motor training = body vector + goal.
BODY_OBS_DIM = 197
MOTOR_TRAINING_OBS_DIM = BODY_OBS_DIM + MOTOR_GOAL_DIM  # 202

# Live mind: think less often than the motor steps.
DEFAULT_MIND_INTERVAL = 40

# Locomotion reward weights (Python fake + Unity mirror).
FORWARD_PROGRESS_WEIGHT = 1.0
GOAL_TRACKING_WEIGHT = 0.8
UPRIGHT_WEIGHT = 0.3
FALL_PENALTY = 1.0
ACTION_EFFORT_WEIGHT = 0.02
ALIVE_BONUS = 0.001

# Fake locomotion / eval
LOCOMOTION_EPISODE_STEPS = 200
MIN_FORWARD_DISPLACEMENT = 0.15
MAX_TRACKING_ERROR = 0.45
