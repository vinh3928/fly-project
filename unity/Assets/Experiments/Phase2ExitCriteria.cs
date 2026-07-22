namespace FlySimulation.Experiments
{
    /// <summary>
    /// Shared Phase 2 exit-criteria constants (single source of truth).
    /// </summary>
    public static class Phase2ExitCriteria
    {
        /// <summary>Required standing duration without falling.</summary>
        public const float StandSeconds = 30f;

        /// <summary>Automated EditMode stand soak (subset of StandSeconds).</summary>
        public const int AutomatedStandFixedSteps = 2000;

        /// <summary>Automated walk / turn / recovery simulation steps.</summary>
        public const int AutomatedManeuverFixedSteps = 800;

        public const float AutomatedFixedDeltaTime = 0.005f; // 200 Hz

        /// <summary>Minimum forward displacement (meters) to count as walking.</summary>
        public const float WalkMinDisplacement = 0.01f;

        /// <summary>Minimum yaw change (degrees) to count as turning.</summary>
        public const float TurnMinYawDegrees = 20f;

        /// <summary>Impulse magnitude used for disturbance-recovery tests.</summary>
        public const float DisturbanceImpulse = 0.35f;

        /// <summary>Max tilt (degrees) after recovery window.</summary>
        public const float RecoveryMaxTiltDegrees = 30f;

        public static float AutomatedStandSeconds =>
            AutomatedStandFixedSteps * AutomatedFixedDeltaTime;

        public static float AutomatedManeuverSeconds =>
            AutomatedManeuverFixedSteps * AutomatedFixedDeltaTime;
    }
}
