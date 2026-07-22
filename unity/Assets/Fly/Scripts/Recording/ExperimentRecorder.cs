using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Motors;
using UnityEngine;

namespace FlySimulation.Fly.Recording
{
    /// <summary>
    /// Writes synchronized experiment metadata, CSV sensor tables, and JSONL events.
    /// Parquet export is handled post-hoc by the Python pipeline.
    /// </summary>
    public sealed class ExperimentRecorder : MonoBehaviour
    {
        [SerializeField] private bool enabledRecording = true;

        private string _episodeDirectory;
        private ExperimentConfiguration _config;
        private int _episodeId;
        private StreamWriter _observationsWriter;
        private StreamWriter _actionsWriter;
        private StreamWriter _internalWriter;
        private StreamWriter _eventsWriter;
        private bool _headerWritten;
        private readonly StringBuilder _line = new StringBuilder(512);

        public string EpisodeDirectory => _episodeDirectory;
        public bool IsRecording => enabledRecording && _observationsWriter != null;

        public void BeginEpisode(ExperimentConfiguration config, int episodeId, string directory)
        {
            if (!enabledRecording)
            {
                return;
            }

            EndEpisode();
            _config = config;
            _episodeId = episodeId;
            _episodeDirectory = directory;
            Directory.CreateDirectory(directory);

            File.WriteAllText(Path.Combine(directory, "metadata.json"), BuildMetadataJson(config, episodeId));
            File.WriteAllText(Path.Combine(directory, "environment.json"), config.ToJson());

            _observationsWriter = new StreamWriter(Path.Combine(directory, "observations.csv"), false, Encoding.UTF8);
            _actionsWriter = new StreamWriter(Path.Combine(directory, "actions.csv"), false, Encoding.UTF8);
            _internalWriter = new StreamWriter(Path.Combine(directory, "internal_state.csv"), false, Encoding.UTF8);
            _eventsWriter = new StreamWriter(Path.Combine(directory, "events.jsonl"), false, Encoding.UTF8);
            _headerWritten = false;
        }

        public void RecordStep(
            FlyEpisodeManager episode,
            FlyObservationCollector observations,
            FlyMotorController motors,
            FlyHomeostasis homeostasis,
            FlyBodyController body)
        {
            if (!IsRecording || episode == null)
            {
                return;
            }

            if (!_headerWritten)
            {
                WriteHeaders(observations != null ? observations.GetVectorLength() : 0);
                _headerWritten = true;
            }

            string common = string.Join(",",
                Escape(_config.ExperimentId),
                _episodeId.ToString(CultureInfo.InvariantCulture),
                episode.SimulationStep.ToString(CultureInfo.InvariantCulture),
                episode.EpisodeTime.ToString("F6", CultureInfo.InvariantCulture),
                _config.RandomSeed.ToString(CultureInfo.InvariantCulture),
                Escape(_config.FlyId));

            if (observations != null)
            {
                float[] vector = observations.GetVectorObservation();
                int length = observations.GetVectorLength();
                _line.Clear();
                _line.Append(common);
                for (int i = 0; i < length; i++)
                {
                    _line.Append(',').Append(vector[i].ToString("G9", CultureInfo.InvariantCulture));
                }

                _observationsWriter.WriteLine(_line);
            }

            _line.Clear();
            _line.Append(common);
            _line.Append(',').Append(motors != null ? motors.LastMotorEffort.ToString("G9", CultureInfo.InvariantCulture) : "0");
            _line.Append(',').Append(motors != null ? ((int)motors.Mode).ToString(CultureInfo.InvariantCulture) : "0");
            _actionsWriter.WriteLine(_line);

            _line.Clear();
            _line.Append(common);
            if (homeostasis != null)
            {
                _line.Append(',').Append(homeostasis.Energy.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.Hydration.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.Fatigue.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.Damage.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.BodyTemperature.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.ToxinExposure.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.Hunger.ToString("G9", CultureInfo.InvariantCulture));
                _line.Append(',').Append(homeostasis.Arousal.ToString("G9", CultureInfo.InvariantCulture));
            }

            if (body != null)
            {
                _line.Append(',').Append(body.HasFallen ? "1" : "0");
                _line.Append(',').Append(body.BodyInstability().ToString("G9", CultureInfo.InvariantCulture));
            }

            _internalWriter.WriteLine(_line);
        }

        public void LogEvent(long step, float timestamp, string eventName)
        {
            if (_eventsWriter == null)
            {
                return;
            }

            string json =
                "{\"experiment_id\":\"" + EscapeJson(_config != null ? _config.ExperimentId : "") +
                "\",\"episode_id\":" + _episodeId +
                ",\"step\":" + step +
                ",\"timestamp\":" + timestamp.ToString("F6", CultureInfo.InvariantCulture) +
                ",\"event\":\"" + EscapeJson(eventName) + "\"}";
            _eventsWriter.WriteLine(json);
        }

        /// <summary>
        /// Writes Phase 1 episode summary telemetry (steps, duration, physics health).
        /// Call while the episode directory is still open, before <see cref="EndEpisode"/>.
        /// </summary>
        public void WriteEpisodeSummary(
            long steps,
            float durationSeconds,
            bool physicsStable,
            string endReason,
            float maxLinearSpeed,
            float maxAngularSpeed)
        {
            if (!enabledRecording || string.IsNullOrEmpty(_episodeDirectory))
            {
                return;
            }

            string json =
                "{\n" +
                "  \"experiment_id\": \"" + EscapeJson(_config != null ? _config.ExperimentId : "") + "\",\n" +
                "  \"episode_id\": " + _episodeId + ",\n" +
                "  \"seed\": " + (_config != null ? _config.RandomSeed : 0) + ",\n" +
                "  \"steps\": " + steps + ",\n" +
                "  \"duration_seconds\": " + durationSeconds.ToString("F6", CultureInfo.InvariantCulture) + ",\n" +
                "  \"physics_stable\": " + (physicsStable ? "true" : "false") + ",\n" +
                "  \"end_reason\": \"" + EscapeJson(endReason) + "\",\n" +
                "  \"max_linear_speed\": " + maxLinearSpeed.ToString("G9", CultureInfo.InvariantCulture) + ",\n" +
                "  \"max_angular_speed\": " + maxAngularSpeed.ToString("G9", CultureInfo.InvariantCulture) + "\n" +
                "}\n";
            File.WriteAllText(Path.Combine(_episodeDirectory, "episode_summary.json"), json);
        }

        public void EndEpisode()
        {
            _observationsWriter?.Dispose();
            _actionsWriter?.Dispose();
            _internalWriter?.Dispose();
            _eventsWriter?.Dispose();
            _observationsWriter = null;
            _actionsWriter = null;
            _internalWriter = null;
            _eventsWriter = null;
            _headerWritten = false;
        }

        private void WriteHeaders(int observationLength)
        {
            var obsHeader = new List<string>
            {
                "experiment_id", "episode_id", "step", "sim_time", "seed", "fly_id"
            };
            for (int i = 0; i < observationLength; i++)
            {
                obsHeader.Add($"obs_{i}");
            }

            _observationsWriter.WriteLine(string.Join(",", obsHeader));
            _actionsWriter.WriteLine("experiment_id,episode_id,step,sim_time,seed,fly_id,motor_effort,control_mode");
            _internalWriter.WriteLine(
                "experiment_id,episode_id,step,sim_time,seed,fly_id,energy,hydration,fatigue,damage,temperature,toxin,hunger,arousal,fallen,instability");
        }

        private static string BuildMetadataJson(ExperimentConfiguration config, int episodeId)
        {
            return "{\n" +
                   "  \"experiment_id\": \"" + EscapeJson(config.ExperimentId) + "\",\n" +
                   "  \"episode_id\": " + episodeId + ",\n" +
                   "  \"seed\": " + config.RandomSeed + ",\n" +
                   "  \"fly_id\": \"" + EscapeJson(config.FlyId) + "\",\n" +
                   "  \"created_utc\": \"" + DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture) + "\"\n" +
                   "}\n";
        }

        private static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return "";
            }

            if (value.Contains(",") || value.Contains("\""))
            {
                return "\"" + value.Replace("\"", "\"\"") + "\"";
            }

            return value;
        }

        private static string EscapeJson(string value)
        {
            return (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"");
        }

        private void OnDestroy()
        {
            EndEpisode();
        }
    }
}
