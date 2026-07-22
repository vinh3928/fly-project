using System;
using System.IO;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Recording;
using FlySimulation.Fly.Sensors;
using UnityEngine;

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Owns episode lifecycle, deterministic resets, and step timing.
    /// </summary>
    public sealed class FlyEpisodeManager : MonoBehaviour
    {
        [SerializeField] private ExperimentConfiguration configuration;
        [SerializeField] private ChamberController chamber;
        [SerializeField] private EnvironmentRandomizer randomizer;
        [SerializeField] private FlyBodyController body;
        [SerializeField] private FlyHomeostasis homeostasis;
        [SerializeField] private FlyMotorController motors;
        [SerializeField] private ChemicalFieldSystem chemicalField;
        [SerializeField] private ExperimentRecorder recorder;
        [SerializeField] private PhysicsStabilityMonitor stabilityMonitor;
        [SerializeField] private string dataRoot = "data/experiments";

        public ExperimentConfiguration Configuration => configuration;
        public long SimulationStep { get; private set; }
        public float EpisodeTime { get; private set; }
        public int EpisodeId { get; private set; }
        public bool EpisodeActive { get; private set; }
        public PhysicsStabilityMonitor StabilityMonitor => stabilityMonitor;

        public event Action EpisodeStarted;
        public event Action<string> EpisodeEnded;

        public void Configure(ExperimentConfiguration config)
        {
            configuration = config ?? ExperimentConfiguration.Default();
        }

        public void BindStabilityMonitor(PhysicsStabilityMonitor monitor)
        {
            stabilityMonitor = monitor;
        }

        public void StartEpisode()
        {
            if (configuration == null)
            {
                configuration = ExperimentConfiguration.Default();
            }

            configuration.ApplySeed();
            EpisodeId++;
            SimulationStep = 0;
            EpisodeTime = 0f;
            EpisodeActive = true;

            chamber?.ApplyConfiguration(configuration);
            // Clear cached layout so Randomize always rebuilds from RandomSeed.
            configuration.ObstaclePositions = Array.Empty<Vector3>();
            randomizer?.Randomize(configuration);
            chemicalField?.DiscoverSources();

            Vector3 spawn = configuration.FlySpawnPosition;
            Quaternion rotation = Quaternion.Euler(configuration.FlySpawnEuler);
            body?.Configure(configuration);
            body?.ResetPose(spawn, rotation);
            homeostasis?.ResetState(configuration);
            motors?.ApplyStanding();

            if (stabilityMonitor == null && body != null)
            {
                stabilityMonitor = body.GetComponent<PhysicsStabilityMonitor>();
            }

            if (stabilityMonitor != null && body != null)
            {
                stabilityMonitor.Bind(body.RootBody);
            }
            else
            {
                stabilityMonitor?.ResetMonitor();
            }

            string experimentDir = Path.Combine(
                GetProjectRoot(),
                dataRoot,
                $"{configuration.ExperimentId}_ep{EpisodeId:D4}");
            recorder?.BeginEpisode(configuration, EpisodeId, experimentDir);
            recorder?.LogEvent(SimulationStep, EpisodeTime, "episode_started");

            EpisodeStarted?.Invoke();
        }

        public void EndEpisode(string reason)
        {
            if (!EpisodeActive)
            {
                return;
            }

            EpisodeActive = false;
            recorder?.LogEvent(SimulationStep, EpisodeTime, reason);
            WriteTelemetrySummary(reason);
            recorder?.EndEpisode();
            EpisodeEnded?.Invoke(reason);
        }

        private void FixedUpdate()
        {
            if (!EpisodeActive)
            {
                return;
            }

            SimulationStep++;
            EpisodeTime += Time.fixedDeltaTime;

            if (body != null)
            {
                body.FixedSensorUpdate();
            }

            if (stabilityMonitor != null && !stabilityMonitor.Evaluate())
            {
                recorder?.LogEvent(SimulationStep, EpisodeTime, "physics_unstable:" + stabilityMonitor.FailureReason);
                EndEpisode("physics_unstable");
                return;
            }

            if (body != null && body.HasFallen)
            {
                recorder?.LogEvent(SimulationStep, EpisodeTime, "fly_fell");
                EndEpisode("fly_fell");
                return;
            }

            if (homeostasis != null && homeostasis.IsDepleted)
            {
                recorder?.LogEvent(SimulationStep, EpisodeTime, "energy_depleted");
                EndEpisode("energy_depleted");
                return;
            }

            if (EpisodeTime >= configuration.EpisodeDurationSeconds)
            {
                EndEpisode("episode_ended");
            }
        }

        private void WriteTelemetrySummary(string reason)
        {
            bool stable = stabilityMonitor == null || stabilityMonitor.IsStable;
            recorder?.WriteEpisodeSummary(
                SimulationStep,
                EpisodeTime,
                stable,
                reason,
                stabilityMonitor != null ? stabilityMonitor.MaxObservedLinearSpeed : 0f,
                stabilityMonitor != null ? stabilityMonitor.MaxObservedAngularSpeed : 0f);
        }

        private static string GetProjectRoot()
        {
            // Unity project is under unity/; repo root is one level up.
            var dataPath = Application.dataPath;
            var unityRoot = Directory.GetParent(dataPath)?.FullName;
            var repoRoot = Directory.GetParent(unityRoot ?? dataPath)?.FullName;
            return repoRoot ?? unityRoot ?? dataPath;
        }
    }
}
