using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Sensors;
using UnityEngine;

namespace FlySimulation.Fly.Recording
{
    /// <summary>
    /// Writes synchronized Phase 3 multimodal experiment logs (CSV/JSONL/PNG).
    /// Parquet export is handled post-hoc by the Python pipeline.
    /// </summary>
    public sealed class ExperimentRecorder : MonoBehaviour
    {
        [SerializeField] private bool enabledRecording = true;
        [SerializeField] private bool recordVision = true;

        private string _episodeDirectory;
        private ExperimentConfiguration _config;
        private int _episodeId;
        private StreamWriter _observationsWriter;
        private StreamWriter _actionsWriter;
        private StreamWriter _internalWriter;
        private StreamWriter _jointsWriter;
        private StreamWriter _contactsWriter;
        private StreamWriter _tasteWriter;
        private StreamWriter _odorWriter;
        private StreamWriter _poseWriter;
        private StreamWriter _eventsWriter;
        private StreamWriter _visionManifestWriter;
        private bool _headerWritten;
        private long _visionFrameIndex;
        private readonly StringBuilder _line = new StringBuilder(1024);

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
            if (recordVision)
            {
                Directory.CreateDirectory(Path.Combine(directory, "vision"));
            }

            File.WriteAllText(
                Path.Combine(directory, Phase3ExitCriteria.MetadataFile),
                BuildMetadataJson(config, episodeId));
            File.WriteAllText(
                Path.Combine(directory, Phase3ExitCriteria.EnvironmentFile),
                config.ToJson());

            _observationsWriter = NewWriter(Phase3ExitCriteria.ObservationsFile);
            _actionsWriter = NewWriter(Phase3ExitCriteria.ActionsFile);
            _internalWriter = NewWriter(Phase3ExitCriteria.InternalStateFile);
            _jointsWriter = NewWriter(Phase3ExitCriteria.JointsFile);
            _contactsWriter = NewWriter(Phase3ExitCriteria.ContactsFile);
            _tasteWriter = NewWriter(Phase3ExitCriteria.TasteFile);
            _odorWriter = NewWriter(Phase3ExitCriteria.OdorFile);
            _poseWriter = NewWriter(Phase3ExitCriteria.PoseFile);
            _eventsWriter = NewWriter(Phase3ExitCriteria.EventsFile);
            if (recordVision)
            {
                _visionManifestWriter = NewWriter(Phase3ExitCriteria.VisionManifestFile);
            }

            _headerWritten = false;
            _visionFrameIndex = 0;
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

            SensorManager sensors = observations != null ? observations.Sensors : null;
            if (!_headerWritten)
            {
                WriteHeaders(observations != null ? observations.GetVectorLength() : 0);
                _headerWritten = true;
            }

            string common = BuildCommon(episode);

            WriteObservationRow(common, observations);
            WriteActionRow(common, motors);
            WriteInternalRow(common, homeostasis, body);
            WriteJointsRow(common, sensors);
            WriteContactsRow(common, sensors);
            WriteTasteRow(common, sensors);
            WriteOdorRow(common, sensors);
            WritePoseRow(common, body, sensors);

            MaybeRecordVision(episode, body, sensors);
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
                "  \"max_angular_speed\": " + maxAngularSpeed.ToString("G9", CultureInfo.InvariantCulture) + ",\n" +
                "  \"vision_frames\": " + _visionFrameIndex + "\n" +
                "}\n";
            File.WriteAllText(Path.Combine(_episodeDirectory, Phase3ExitCriteria.EpisodeSummaryFile), json);
        }

        public void EndEpisode()
        {
            DisposeWriter(ref _observationsWriter);
            DisposeWriter(ref _actionsWriter);
            DisposeWriter(ref _internalWriter);
            DisposeWriter(ref _jointsWriter);
            DisposeWriter(ref _contactsWriter);
            DisposeWriter(ref _tasteWriter);
            DisposeWriter(ref _odorWriter);
            DisposeWriter(ref _poseWriter);
            DisposeWriter(ref _eventsWriter);
            DisposeWriter(ref _visionManifestWriter);
            _headerWritten = false;
        }

        private StreamWriter NewWriter(string relativePath)
        {
            return new StreamWriter(Path.Combine(_episodeDirectory, relativePath), false, Encoding.UTF8);
        }

        private static void DisposeWriter(ref StreamWriter writer)
        {
            writer?.Dispose();
            writer = null;
        }

        private string BuildCommon(FlyEpisodeManager episode)
        {
            return string.Join(",",
                Escape(_config.ExperimentId),
                _episodeId.ToString(CultureInfo.InvariantCulture),
                episode.SimulationStep.ToString(CultureInfo.InvariantCulture),
                episode.EpisodeTime.ToString("F6", CultureInfo.InvariantCulture),
                _config.RandomSeed.ToString(CultureInfo.InvariantCulture),
                Escape(_config.FlyId));
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
            _actionsWriter.WriteLine(Phase3ExitCriteria.CommonCsvHeader + ",motor_effort,control_mode");
            _internalWriter.WriteLine(
                Phase3ExitCriteria.CommonCsvHeader +
                ",energy,hydration,fatigue,damage,temperature,toxin,hunger,arousal,fallen,instability");

            var jointCols = new List<string> { Phase3ExitCriteria.CommonCsvHeader };
            for (int i = 0; i < SensorManager.JointCount; i++)
            {
                jointCols.Add($"j{i}_pos");
                jointCols.Add($"j{i}_vel");
                jointCols.Add($"j{i}_tgt");
                jointCols.Add($"j{i}_force");
            }

            _jointsWriter.WriteLine(string.Join(",", jointCols));

            var contactCols = new List<string> { Phase3ExitCriteria.CommonCsvHeader };
            for (int i = 0; i < SensorManager.FootCount; i++)
            {
                contactCols.Add($"f{i}_touch");
                contactCols.Add($"f{i}_force");
                contactCols.Add($"f{i}_nx");
                contactCols.Add($"f{i}_ny");
                contactCols.Add($"f{i}_nz");
            }

            _contactsWriter.WriteLine(string.Join(",", contactCols));

            var tasteCols = new List<string> { Phase3ExitCriteria.CommonCsvHeader };
            for (int i = 0; i < SensorManager.FootCount; i++)
            {
                AppendChemCols(tasteCols, $"foot{i}");
            }

            tasteCols.Add("proboscis_contact");
            AppendChemCols(tasteCols, "proboscis");
            _tasteWriter.WriteLine(string.Join(",", tasteCols));

            var odorCols = new List<string> { Phase3ExitCriteria.CommonCsvHeader };
            AppendChemCols(odorCols, "left");
            AppendChemCols(odorCols, "right");
            _odorWriter.WriteLine(string.Join(",", odorCols));

            _poseWriter.WriteLine(
                Phase3ExitCriteria.CommonCsvHeader +
                ",pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,fwd_x,fwd_y,fwd_z," +
                "left_eye_fwd_x,left_eye_fwd_y,left_eye_fwd_z," +
                "right_eye_fwd_x,right_eye_fwd_y,right_eye_fwd_z");
        }

        private static void AppendChemCols(List<string> cols, string prefix)
        {
            cols.Add($"{prefix}_sugar");
            cols.Add($"{prefix}_bitter");
            cols.Add($"{prefix}_salt");
            cols.Add($"{prefix}_acid");
            cols.Add($"{prefix}_water");
            cols.Add($"{prefix}_protein");
            cols.Add($"{prefix}_toxin");
            cols.Add($"{prefix}_odor");
        }

        private void WriteObservationRow(string common, FlyObservationCollector observations)
        {
            if (observations == null)
            {
                return;
            }

            float[] vector = observations.GetVectorObservation();
            int length = observations.GetVectorLength();
            _line.Clear();
            _line.Append(common);
            for (int i = 0; i < length; i++)
            {
                _line.Append(',').Append(Fmt(vector[i]));
            }

            _observationsWriter.WriteLine(_line);
        }

        private void WriteActionRow(string common, FlyMotorController motors)
        {
            _line.Clear();
            _line.Append(common);
            _line.Append(',').Append(motors != null ? Fmt(motors.LastMotorEffort) : "0");
            _line.Append(',').Append(motors != null ? ((int)motors.Mode).ToString(CultureInfo.InvariantCulture) : "0");
            _actionsWriter.WriteLine(_line);
        }

        private void WriteInternalRow(string common, FlyHomeostasis homeostasis, FlyBodyController body)
        {
            _line.Clear();
            _line.Append(common);
            if (homeostasis != null)
            {
                _line.Append(',').Append(Fmt(homeostasis.Energy));
                _line.Append(',').Append(Fmt(homeostasis.Hydration));
                _line.Append(',').Append(Fmt(homeostasis.Fatigue));
                _line.Append(',').Append(Fmt(homeostasis.Damage));
                _line.Append(',').Append(Fmt(homeostasis.BodyTemperature));
                _line.Append(',').Append(Fmt(homeostasis.ToxinExposure));
                _line.Append(',').Append(Fmt(homeostasis.Hunger));
                _line.Append(',').Append(Fmt(homeostasis.Arousal));
            }
            else
            {
                _line.Append(",0,0,0,0,0,0,0,0");
            }

            _line.Append(',').Append(body != null && body.HasFallen ? "1" : "0");
            _line.Append(',').Append(body != null ? Fmt(body.BodyInstability()) : "0");
            _internalWriter.WriteLine(_line);
        }

        private void WriteJointsRow(string common, SensorManager sensors)
        {
            _line.Clear();
            _line.Append(common);
            var joints = sensors != null ? sensors.Joints : null;
            for (int i = 0; i < SensorManager.JointCount; i++)
            {
                JointObservation obs = joints != null && i < joints.Length ? joints[i] : default;
                _line.Append(',').Append(Fmt(obs.Position));
                _line.Append(',').Append(Fmt(obs.Velocity));
                _line.Append(',').Append(Fmt(obs.Target));
                _line.Append(',').Append(Fmt(obs.AppliedForce));
            }

            _jointsWriter.WriteLine(_line);
        }

        private void WriteContactsRow(string common, SensorManager sensors)
        {
            _line.Clear();
            _line.Append(common);
            var feet = sensors != null ? sensors.Feet : null;
            for (int i = 0; i < SensorManager.FootCount; i++)
            {
                FootContactSensor foot = feet != null && i < feet.Length ? feet[i] : null;
                if (foot != null)
                {
                    _line.Append(',').Append(foot.IsTouching ? "1" : "0");
                    _line.Append(',').Append(Fmt(foot.NormalForce));
                    _line.Append(',').Append(Fmt(foot.SurfaceNormal.x));
                    _line.Append(',').Append(Fmt(foot.SurfaceNormal.y));
                    _line.Append(',').Append(Fmt(foot.SurfaceNormal.z));
                }
                else
                {
                    _line.Append(",0,0,0,1,0");
                }
            }

            _contactsWriter.WriteLine(_line);
        }

        private void WriteTasteRow(string common, SensorManager sensors)
        {
            _line.Clear();
            _line.Append(common);
            var feet = sensors != null ? sensors.Feet : null;
            for (int i = 0; i < SensorManager.FootCount; i++)
            {
                FootContactSensor foot = feet != null && i < feet.Length ? feet[i] : null;
                ChemicalChannels gated = foot != null
                    ? TasteGating.Gate(foot.IsTouching, foot.Taste)
                    : ChemicalChannels.Zero;
                AppendChem(_line, gated);
            }

            ProboscisSensor proboscis = sensors != null ? sensors.Proboscis : null;
            _line.Append(',').Append(proboscis != null && proboscis.IsContacting ? "1" : "0");
            AppendChem(_line, proboscis != null ? proboscis.GatedTaste : ChemicalChannels.Zero);
            _tasteWriter.WriteLine(_line);
        }

        private void WriteOdorRow(string common, SensorManager sensors)
        {
            _line.Clear();
            _line.Append(common);
            AntennaOdorSensor left = sensors != null ? sensors.LeftAntenna : null;
            AntennaOdorSensor right = sensors != null ? sensors.RightAntenna : null;
            AppendChem(_line, left != null ? left.Odor : ChemicalChannels.Zero);
            AppendChem(_line, right != null ? right.Odor : ChemicalChannels.Zero);
            _odorWriter.WriteLine(_line);
        }

        private void WritePoseRow(string common, FlyBodyController body, SensorManager sensors)
        {
            _line.Clear();
            _line.Append(common);
            Vector3 pos = body != null ? body.transform.position : Vector3.zero;
            Quaternion rot = body != null ? body.transform.rotation : Quaternion.identity;
            Vector3 fwd = body != null ? body.transform.forward : Vector3.forward;
            _line.Append(',').Append(Fmt(pos.x)).Append(',').Append(Fmt(pos.y)).Append(',').Append(Fmt(pos.z));
            _line.Append(',').Append(Fmt(rot.x)).Append(',').Append(Fmt(rot.y)).Append(',').Append(Fmt(rot.z)).Append(',').Append(Fmt(rot.w));
            _line.Append(',').Append(Fmt(fwd.x)).Append(',').Append(Fmt(fwd.y)).Append(',').Append(Fmt(fwd.z));

            FlyEyeCamera left = sensors != null && sensors.Vision != null ? sensors.Vision.LeftEye : null;
            FlyEyeCamera right = sensors != null && sensors.Vision != null ? sensors.Vision.RightEye : null;
            Vector3 lf = left != null ? left.Forward : fwd;
            Vector3 rf = right != null ? right.Forward : fwd;
            _line.Append(',').Append(Fmt(lf.x)).Append(',').Append(Fmt(lf.y)).Append(',').Append(Fmt(lf.z));
            _line.Append(',').Append(Fmt(rf.x)).Append(',').Append(Fmt(rf.y)).Append(',').Append(Fmt(rf.z));
            _poseWriter.WriteLine(_line);
        }

        private void MaybeRecordVision(FlyEpisodeManager episode, FlyBodyController body, SensorManager sensors)
        {
            if (!recordVision || _visionManifestWriter == null || _config == null)
            {
                return;
            }

            int visionPeriod = Mathf.Max(
                1,
                Mathf.RoundToInt(_config.PhysicsHz / Mathf.Max(1f, _config.VisionHz)));
            if (episode.SimulationStep % visionPeriod != 0)
            {
                return;
            }

            VisionFrameProvider vision = sensors != null ? sensors.Vision : null;
            vision?.CaptureNow();

            FlyEyeCamera left = vision != null ? vision.LeftEye : null;
            FlyEyeCamera right = vision != null ? vision.RightEye : null;
            string leftName = $"left_{_visionFrameIndex:D6}.png";
            string rightName = $"right_{_visionFrameIndex:D6}.png";

            if (left != null)
            {
                byte[] png = left.CapturePng();
                if (png != null && png.Length > 0)
                {
                    File.WriteAllBytes(Path.Combine(_episodeDirectory, "vision", leftName), png);
                }
            }

            if (right != null)
            {
                byte[] png = right.CapturePng();
                if (png != null && png.Length > 0)
                {
                    File.WriteAllBytes(Path.Combine(_episodeDirectory, "vision", rightName), png);
                }
            }

            Vector3 bodyFwd = body != null ? body.transform.forward : Vector3.forward;
            Vector3 leftFwd = left != null ? left.Forward : bodyFwd;
            Vector3 rightFwd = right != null ? right.Forward : bodyFwd;
            string json =
                "{\"step\":" + episode.SimulationStep +
                ",\"timestamp\":" + episode.EpisodeTime.ToString("F6", CultureInfo.InvariantCulture) +
                ",\"frame_index\":" + _visionFrameIndex +
                ",\"left_file\":\"vision/" + leftName + "\"" +
                ",\"right_file\":\"vision/" + rightName + "\"" +
                ",\"body_forward\":[" + Fmt(bodyFwd.x) + "," + Fmt(bodyFwd.y) + "," + Fmt(bodyFwd.z) + "]" +
                ",\"left_eye_forward\":[" + Fmt(leftFwd.x) + "," + Fmt(leftFwd.y) + "," + Fmt(leftFwd.z) + "]" +
                ",\"right_eye_forward\":[" + Fmt(rightFwd.x) + "," + Fmt(rightFwd.y) + "," + Fmt(rightFwd.z) + "]" +
                "}";
            _visionManifestWriter.WriteLine(json);
            _visionFrameIndex++;
        }

        private static void AppendChem(StringBuilder line, ChemicalChannels channels)
        {
            line.Append(',').Append(Fmt(channels.Sugar));
            line.Append(',').Append(Fmt(channels.Bitter));
            line.Append(',').Append(Fmt(channels.Salt));
            line.Append(',').Append(Fmt(channels.Acid));
            line.Append(',').Append(Fmt(channels.Water));
            line.Append(',').Append(Fmt(channels.Protein));
            line.Append(',').Append(Fmt(channels.Toxin));
            line.Append(',').Append(Fmt(channels.Odor));
        }

        private static string Fmt(float value)
        {
            return value.ToString("G9", CultureInfo.InvariantCulture);
        }

        private static string BuildMetadataJson(ExperimentConfiguration config, int episodeId)
        {
            return "{\n" +
                   "  \"experiment_id\": \"" + EscapeJson(config.ExperimentId) + "\",\n" +
                   "  \"episode_id\": " + episodeId + ",\n" +
                   "  \"seed\": " + config.RandomSeed + ",\n" +
                   "  \"fly_id\": \"" + EscapeJson(config.FlyId) + "\",\n" +
                   "  \"recording_hz\": " + config.RecordingHz.ToString("G9", CultureInfo.InvariantCulture) + ",\n" +
                   "  \"vision_hz\": " + config.VisionHz.ToString("G9", CultureInfo.InvariantCulture) + ",\n" +
                   "  \"schema_version\": 3,\n" +
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
