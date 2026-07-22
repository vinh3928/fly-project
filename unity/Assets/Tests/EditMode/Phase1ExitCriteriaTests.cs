using System.IO;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Recording;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class DeterministicLayoutTests
    {
        [Test]
        public void SameSeed_ProducesIdenticalObstacleLayouts()
        {
            var size = new Vector3(1f, 0.5f, 1f);
            Vector3[] a = DeterministicLayout.GenerateObstaclePositions(42, size, 5, 0.15f, 0.05f);
            Vector3[] b = DeterministicLayout.GenerateObstaclePositions(42, size, 5, 0.15f, 0.05f);
            Vector3[] c = DeterministicLayout.GenerateObstaclePositions(99, size, 5, 0.15f, 0.05f);

            Assert.IsTrue(DeterministicLayout.AreEqual(a, b));
            Assert.IsFalse(DeterministicLayout.AreEqual(a, c));
            Assert.AreEqual(5, a.Length);
            Assert.Greater(a[0].y, 0f);
        }

        [Test]
        public void ObstaclePositions_StayInsideChamberMargins()
        {
            var size = new Vector3(1f, 0.5f, 1f);
            const float margin = 0.15f;
            const float radius = 0.05f;
            Vector3[] positions = DeterministicLayout.GenerateObstaclePositions(7, size, 20, margin, radius);

            float halfX = size.x * 0.5f - margin;
            float halfZ = size.z * 0.5f - margin;
            for (int i = 0; i < positions.Length; i++)
            {
                Assert.LessOrEqual(Mathf.Abs(positions[i].x), halfX + 1e-4f);
                Assert.LessOrEqual(Mathf.Abs(positions[i].z), halfZ + 1e-4f);
                Assert.AreEqual(radius, positions[i].y, 1e-5f);
            }
        }
    }

    public sealed class PhysicsStabilityMonitorTests
    {
        [Test]
        public void IsStateStable_AcceptsFiniteQuietMotion()
        {
            bool ok = PhysicsStabilityMonitor.IsStateStable(
                Vector3.zero,
                Vector3.one * 0.1f,
                Vector3.zero,
                maxLinearSpeed: 25f,
                maxAngularSpeed: 50f,
                out string reason);

            Assert.IsTrue(ok);
            Assert.AreEqual(string.Empty, reason);
        }

        [Test]
        public void IsStateStable_RejectsNaNAndRunawayVelocity()
        {
            Assert.IsFalse(PhysicsStabilityMonitor.IsStateStable(
                new Vector3(float.NaN, 0f, 0f),
                Vector3.zero,
                Vector3.zero,
                25f,
                50f,
                out string nanReason));
            Assert.AreEqual("non_finite_physics_state", nanReason);

            Assert.IsFalse(PhysicsStabilityMonitor.IsStateStable(
                Vector3.zero,
                Vector3.up * 100f,
                Vector3.zero,
                25f,
                50f,
                out string speedReason));
            Assert.AreEqual("linear_speed_exceeded", speedReason);
        }
    }

    public sealed class Phase1SpawnAndResetTests
    {
        [Test]
        public void ResetPose_PlacesBodyAtSpawnAndClearsVelocity()
        {
            var go = new GameObject("fly");
            var rb = go.AddComponent<Rigidbody>();
            rb.useGravity = false;
            var body = go.AddComponent<FlyBodyController>();
            Phase1TestHelpers.SetPrivate(body, "rootBody", rb);
            Phase1TestHelpers.SetPrivate(body, "legs", new LegController[0]);

            var config = ExperimentConfiguration.Default();
            config.LocomotionEnabled = false;
            body.Configure(config);

            Vector3 spawn = new Vector3(0.1f, 0.05f, -0.2f);
            body.ResetPose(spawn, Quaternion.identity);

            Assert.AreEqual(spawn.x, rb.position.x, 1e-5f);
            Assert.AreEqual(spawn.y, rb.position.y, 1e-5f);
            Assert.AreEqual(spawn.z, rb.position.z, 1e-5f);
            Assert.AreEqual(0f, rb.linearVelocity.magnitude, 1e-5f);
            Assert.AreEqual(0f, rb.angularVelocity.magnitude, 1e-5f);
            Assert.AreEqual(RigidbodyConstraints.FreezeRotation, rb.constraints);
            Assert.IsFalse(body.HasFallen);

            Object.DestroyImmediate(go);
        }
    }

    public sealed class ExperimentRecorderTelemetryTests
    {
        [Test]
        public void BeginEpisode_WritesMetadataAndSummary()
        {
            string dir = Path.Combine(Path.GetTempPath(), "fly_phase1_telemetry_" + Path.GetRandomFileName());
            var go = new GameObject("recorder");
            var recorder = go.AddComponent<ExperimentRecorder>();
            var config = ExperimentConfiguration.Default();
            config.ExperimentId = "phase1_test";
            config.RandomSeed = 42;

            recorder.BeginEpisode(config, episodeId: 1, directory: dir);
            recorder.LogEvent(0, 0f, "episode_started");
            recorder.WriteEpisodeSummary(
                steps: 100,
                durationSeconds: 0.5f,
                physicsStable: true,
                endReason: "episode_ended",
                maxLinearSpeed: 0.1f,
                maxAngularSpeed: 0.01f);
            recorder.EndEpisode();

            Assert.IsTrue(File.Exists(Path.Combine(dir, "metadata.json")));
            Assert.IsTrue(File.Exists(Path.Combine(dir, "environment.json")));
            Assert.IsTrue(File.Exists(Path.Combine(dir, "events.jsonl")));
            Assert.IsTrue(File.Exists(Path.Combine(dir, "episode_summary.json")));

            string summary = File.ReadAllText(Path.Combine(dir, "episode_summary.json"));
            StringAssert.Contains("\"physics_stable\": true", summary);
            StringAssert.Contains("\"seed\": 42", summary);
            StringAssert.Contains("episode_ended", summary);

            string events = File.ReadAllText(Path.Combine(dir, "events.jsonl"));
            StringAssert.Contains("episode_started", events);

            Object.DestroyImmediate(go);
            Directory.Delete(dir, recursive: true);
        }
    }

    public sealed class Phase1PhysicsSoakTests
    {
        [Test]
        public void PassiveRigidbody_RemainsStable_ForAutomatedSoak()
        {
            SimulationMode previous = Physics.simulationMode;
            Physics.simulationMode = SimulationMode.Script;

            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.transform.position = new Vector3(0f, -0.5f, 0f);
            floor.transform.localScale = new Vector3(4f, 1f, 4f);

            var fly = new GameObject("soak_fly");
            fly.transform.position = new Vector3(0f, 0.05f, 0f);
            var rb = fly.AddComponent<Rigidbody>();
            rb.mass = 0.2f;
            rb.linearDamping = 1f;
            rb.angularDamping = 5f;
            rb.constraints = RigidbodyConstraints.FreezeRotation;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            var capsule = fly.AddComponent<CapsuleCollider>();
            capsule.height = 0.02f;
            capsule.radius = 0.006f;
            capsule.direction = 2;

            var monitor = fly.AddComponent<PhysicsStabilityMonitor>();
            monitor.Bind(rb);

            Physics.SyncTransforms();

            float dt = Phase1ExitCriteria.AutomatedFixedDeltaTime;
            for (int i = 0; i < Phase1ExitCriteria.AutomatedStabilityFixedSteps; i++)
            {
                Physics.Simulate(dt);
                Assert.IsTrue(monitor.Evaluate(), monitor.FailureReason);
            }

            Assert.IsTrue(monitor.IsStable);
            Assert.Less(monitor.MaxObservedLinearSpeed, 25f);
            Assert.IsTrue(float.IsFinite(rb.position.x));
            Assert.IsTrue(float.IsFinite(rb.position.y));
            Assert.Greater(rb.position.y, -0.1f);

            Physics.simulationMode = previous;
            Object.DestroyImmediate(fly);
            Object.DestroyImmediate(floor);
        }
    }

    public sealed class Phase1ExitCriteriaTests
    {
        [Test]
        public void StabilityTargets_MatchPhase1Spec()
        {
            Assert.AreEqual(600f, Phase1ExitCriteria.StabilitySeconds, 1e-5f);
            Assert.AreEqual(10f, Phase1ExitCriteria.AutomatedStabilitySeconds, 1e-5f);
            Assert.Greater(Phase1ExitCriteria.AutomatedStabilityFixedSteps, 0);
        }
    }

    internal static class Phase1TestHelpers
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
    }
}
