namespace FlySimulation.Experiments
{
    /// <summary>
    /// Shared Phase 1 exit-criteria constants (single source of truth).
    /// </summary>
    public static class Phase1ExitCriteria
    {
        /// <summary>Required simulated duration for physics stability (10 minutes).</summary>
        public const float StabilitySeconds = 600f;

        /// <summary>
        /// Automated EditMode soak length (subset of StabilitySeconds) using Physics.Simulate.
        /// </summary>
        public const int AutomatedStabilityFixedSteps = 2000;

        public const float AutomatedFixedDeltaTime = 0.005f; // 200 Hz

        public static float AutomatedStabilitySeconds =>
            AutomatedStabilityFixedSteps * AutomatedFixedDeltaTime;
    }
}
