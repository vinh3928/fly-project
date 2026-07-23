namespace FlySimulation.Experiments
{
    /// <summary>
    /// Shared Phase 3 exit-criteria constants (single source of truth).
    /// </summary>
    public static class Phase3ExitCriteria
    {
        public const string ObservationsFile = "observations.csv";
        public const string ActionsFile = "actions.csv";
        public const string InternalStateFile = "internal_state.csv";
        public const string JointsFile = "joints.csv";
        public const string ContactsFile = "contacts.csv";
        public const string TasteFile = "taste.csv";
        public const string OdorFile = "odor.csv";
        public const string PoseFile = "pose.csv";
        public const string EventsFile = "events.jsonl";
        public const string MetadataFile = "metadata.json";
        public const string EnvironmentFile = "environment.json";
        public const string VisionManifestFile = "vision/manifest.jsonl";
        public const string EpisodeSummaryFile = "episode_summary.json";

        /// <summary>Common leading columns on every CSV stream.</summary>
        public const string CommonCsvHeader = "experiment_id,episode_id,step,sim_time,seed,fly_id";

        public const int DefaultVisionWidth = 64;
        public const int DefaultVisionHeight = 64;
    }
}
