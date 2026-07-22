"""Phase 1 exit-criteria constants mirrored from Unity Phase1ExitCriteria."""

# Required simulated duration for full physics stability soak (10 minutes).
STABILITY_SECONDS = 600.0

# Automated EditMode soak length (Physics.Simulate subset of STABILITY_SECONDS).
AUTOMATED_STABILITY_FIXED_STEPS = 2000
AUTOMATED_FIXED_DELTA_TIME = 0.005  # 200 Hz

AUTOMATED_STABILITY_SECONDS = AUTOMATED_STABILITY_FIXED_STEPS * AUTOMATED_FIXED_DELTA_TIME
