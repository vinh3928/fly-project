namespace FlySimulation.Experiments
{
    /// <summary>
    /// Phase 5 visual-navigation exit-criteria constants.
    /// </summary>
    public static class Phase5ExitCriteria
    {
        public const string VisualTargetTag = "VisualTarget";
        public const int DefaultObstacleCount = 3;
        public const float DefaultObstacleRadius = 0.008f;
        public const float SuccessRadius = 0.08f;
        public const int MaxWallCollisionsPerEpisode = 3;

        /// <summary>
        /// Agent vector observations must not include privileged target world XYZ.
        /// </summary>
        public const bool AgentObsIncludesTargetXyz = false;

        public const int ExpectedAgentVectorSize = 197;
    }
}
