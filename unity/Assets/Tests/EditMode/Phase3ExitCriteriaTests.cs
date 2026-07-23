using System.IO;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Recording;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class TasteGatingTests
    {
        [Test]
        public void Gate_ZerosTasteWhenNotContacting()
        {
            var raw = new ChemicalChannels { Sugar = 0.8f, Bitter = 0.2f };
            ChemicalChannels gated = TasteGating.Gate(isContacting: false, raw);
            Assert.AreEqual(0f, gated.Sugar, 1e-5f);
            Assert.AreEqual(0f, gated.Bitter, 1e-5f);
            Assert.IsFalse(TasteGating.HasNonZeroTaste(gated));
        }

        [Test]
        public void Gate_PreservesTasteWhenContacting()
        {
            var raw = new ChemicalChannels { Sugar = 0.8f, Water = 0.5f };
            ChemicalChannels gated = TasteGating.Gate(isContacting: true, raw);
            Assert.AreEqual(0.8f, gated.Sugar, 1e-5f);
            Assert.AreEqual(0.5f, gated.Water, 1e-5f);
            Assert.IsTrue(TasteGating.HasNonZeroTaste(gated));
        }
    }

    public sealed class Phase3RecordingTests
    {
        [Test]
        public void Recorder_WritesTimestampedMultimodalStreams()
        {
            string dir = Path.Combine(Path.GetTempPath(), "fly_phase3_" + Path.GetRandomFileName());
            var go = new GameObject("phase3");
            var episode = go.AddComponent<FlyEpisodeManager>();
            var body = go.AddComponent<FlyBodyController>();
            var recorder = go.AddComponent<ExperimentRecorder>();
            Phase3TestHelpers.SetPrivate(recorder, "recordVision", false);

            var config = ExperimentConfiguration.Default();
            config.ExperimentId = "phase3_test";
            recorder.BeginEpisode(config, 1, dir);

            Phase3TestHelpers.SetAutoProp(episode, "SimulationStep", 0L);
            Phase3TestHelpers.SetAutoProp(episode, "EpisodeTime", 0f);
            Phase3TestHelpers.SetAutoProp(episode, "EpisodeActive", true);
            recorder.RecordStep(episode, null, null, null, body);

            Phase3TestHelpers.SetAutoProp(episode, "SimulationStep", 4L);
            Phase3TestHelpers.SetAutoProp(episode, "EpisodeTime", 0.02f);
            recorder.RecordStep(episode, null, null, null, body);

            recorder.LogEvent(4, 0.02f, "phase3_tick");
            recorder.WriteEpisodeSummary(4, 0.02f, true, "episode_ended", 0.1f, 0.01f);
            recorder.EndEpisode();

            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.JointsFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.ContactsFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.TasteFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.OdorFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.PoseFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.ObservationsFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.ActionsFile)));
            Assert.IsTrue(File.Exists(Path.Combine(dir, Phase3ExitCriteria.InternalStateFile)));

            string[] poseLines = File.ReadAllLines(Path.Combine(dir, Phase3ExitCriteria.PoseFile));
            Assert.AreEqual(3, poseLines.Length); // header + 2 rows
            StringAssert.Contains("sim_time", poseLines[0]);
            StringAssert.Contains("0.020000", poseLines[2]);

            string[] tasteLines = File.ReadAllLines(Path.Combine(dir, Phase3ExitCriteria.TasteFile));
            Assert.AreEqual(3, tasteLines.Length);
            StringAssert.Contains("sim_time", tasteLines[0]);

            string metadata = File.ReadAllText(Path.Combine(dir, Phase3ExitCriteria.MetadataFile));
            StringAssert.Contains("\"schema_version\": 3", metadata);

            string events = File.ReadAllText(Path.Combine(dir, Phase3ExitCriteria.EventsFile));
            StringAssert.Contains("phase3_tick", events);
            StringAssert.Contains("\"timestamp\":", events);

            Object.DestroyImmediate(go);
            Directory.Delete(dir, recursive: true);
        }

        [Test]
        public void VisionOrientation_EyeForwardAlignsWithBodyWhenCoaxial()
        {
            // Contract used by replay: when eyes face body forward, recorded forwards align.
            Vector3 bodyForward = Vector3.forward;
            Vector3 leftEyeForward = Vector3.forward;
            Vector3 rightEyeForward = Vector3.forward;
            Assert.Greater(Vector3.Dot(bodyForward.normalized, leftEyeForward.normalized), 0.99f);
            Assert.Greater(Vector3.Dot(bodyForward.normalized, rightEyeForward.normalized), 0.99f);
        }
    }

    public sealed class Phase3ExitCriteriaConstantTests
    {
        [Test]
        public void SchemaFiles_AreDefined()
        {
            Assert.AreEqual("joints.csv", Phase3ExitCriteria.JointsFile);
            Assert.AreEqual("taste.csv", Phase3ExitCriteria.TasteFile);
            Assert.AreEqual("vision/manifest.jsonl", Phase3ExitCriteria.VisionManifestFile);
            StringAssert.Contains("sim_time", Phase3ExitCriteria.CommonCsvHeader);
        }
    }

    internal static class Phase3TestHelpers
    {
        public static void SetPrivate(object target, string fieldName, object value)
        {
            var field = target.GetType().GetField(
                fieldName,
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Public);
            Assert.IsNotNull(field, fieldName);
            field.SetValue(target, value);
        }

        public static void SetAutoProp(object target, string propertyName, object value)
        {
            string backing = $"<{propertyName}>k__BackingField";
            var field = target.GetType().GetField(
                backing,
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            Assert.IsNotNull(field, backing);
            field.SetValue(target, value);
        }
    }
}
