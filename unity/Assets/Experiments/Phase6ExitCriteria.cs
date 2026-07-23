namespace FlySimulation.Experiments
{
    /// <summary>
    /// Phase 6 taste / homeostasis exit-criteria constants.
    /// </summary>
    public static class Phase6ExitCriteria
    {
        public const float HungerSearchThreshold = 0.45f;
        public const float SugarConsumeRate = 0.08f;
        public const float BitterToxinRate = 0.04f;
        public const float MinProboscisExtensionToFeed = 0.35f;
        public const bool AgentObsIncludesFoodXyz = false;
        public const int ExpectedAgentVectorSize = 197;
    }
}
