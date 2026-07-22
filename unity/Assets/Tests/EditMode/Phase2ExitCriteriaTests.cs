using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class TripodGaitTests
    {
        [Test]
        public void OppositeTripods_ArePhaseShifted()
        {
            TripodGait.ComputeJointTargets(0, 0f, 1f, 0f, 0.5f, 25f, out float coxaA, out _, out _);
            TripodGait.ComputeJointTargets(1, 0f, 1f, 0f, 0.5f, 25f, out float coxaB, out _, out _);
            Assert.AreEqual(0f, coxaA, 1e-4f);
            Assert.AreEqual(0f, coxaB, 1e-4f);

            const float phase = Mathf.PI * 0.5f;
            TripodGait.ComputeJointTargets(0, phase, 1f, 0f, 0.5f, 25f, out coxaA, out _, out _);
            TripodGait.ComputeJointTargets(1, phase, 1f, 0f, 0.5f, 25f, out coxaB, out _, out _);
            Assert.AreEqual(25f, coxaA, 1e-3f);
            Assert.AreEqual(-25f, coxaB, 1e-3f);
        }

        [Test]
        public void TurnBias_MirrorsAcrossSides()
        {
            TripodGait.ComputeJointTargets(0, 0f, 0f, 1f, 0.5f, 25f, out float left, out _, out _);
            TripodGait.ComputeJointTargets(3, 0f, 0f, 1f, 0.5f, 25f, out float right, out _, out _);
            Assert.AreEqual(10f, left, 1e-4f);
            Assert.AreEqual(-10f, right, 1e-4f);
        }
    }

    public sealed class BalanceControllerTests
    {
        [Test]
        public void UprightTorque_PullsTiltedBodyTowardUp()
        {
            Quaternion tilted = Quaternion.Euler(20f, 0f, 0f);
            Vector3 torque = BalanceController.ComputeUprightTorque(tilted, Vector3.zero, 40f, 8f);
            Assert.Greater(torque.magnitude, 0f);
            // Torque axis should oppose the tilt about X.
            Assert.Greater(Mathf.Abs(torque.x), Mathf.Abs(torque.z));
        }

        [Test]
        public void HeightForce_PushesUpWhenBelowTarget()
        {
            Vector3 force = BalanceController.ComputeHeightForce(0.01f, 0f, 0.02f, 60f, 12f);
            Assert.Greater(force.y, 0f);
        }
    }

    public sealed class FootRaycastTests
    {
        [Test]
        public void ProbeGround_DetectsFloorBelow()
        {
            SimulationMode previous = Physics.simulationMode;
            Physics.simulationMode = SimulationMode.Script;

            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.transform.position = new Vector3(0f, -0.5f, 0f);
            floor.transform.localScale = new Vector3(4f, 1f, 4f);
            Physics.SyncTransforms();

            bool hit = FootContactSensor.ProbeGround(
                new Vector3(0f, 0.2f, 0f),
                1f,
                ~0,
                out float distance,
                out Vector3 normal);

            Assert.IsTrue(hit);
            Assert.Less(distance, 1f);
            Assert.Greater(normal.y, 0.5f);

            Physics.simulationMode = previous;
            Object.DestroyImmediate(floor);
        }
    }

    public sealed class Phase2LocomotionTests
    {
        [Test]
        public void Configure_UnlocksRotation_WhenLocomotionEnabled()
        {
            var go = new GameObject("fly");
            var rb = go.AddComponent<Rigidbody>();
            var body = go.AddComponent<FlyBodyController>();
            Phase2TestHelpers.SetPrivate(body, "rootBody", rb);
            Phase2TestHelpers.SetPrivate(body, "legs", new LegController[0]);

            var config = ExperimentConfiguration.Default();
            config.LocomotionEnabled = true;
            body.Configure(config);

            Assert.IsTrue(body.LocomotionEnabled);
            Assert.AreEqual(
                RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ,
                rb.constraints);
            Assert.IsFalse(rb.useGravity);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Stand_RemainsUpright_ForAutomatedSoak()
        {
            using var fixture = new LocomotionFixture();
            fixture.Motors.ApplyStanding();

            float dt = Phase2ExitCriteria.AutomatedFixedDeltaTime;
            for (int i = 0; i < Phase2ExitCriteria.AutomatedStandFixedSteps; i++)
            {
                fixture.Body.FixedSensorUpdate();
                fixture.Motors.Tick(dt);
                Physics.Simulate(dt);
                Assert.IsFalse(fixture.Body.HasFallen, $"fell at step {i}");
            }

            Assert.Less(fixture.Body.TiltDegrees, Phase2ExitCriteria.RecoveryMaxTiltDegrees);
            Assert.Greater(fixture.Body.RootBody.position.y, 0.02f);
        }

        [Test]
        public void WalkForward_DisplacesAlongForwardAxis()
        {
            using var fixture = new LocomotionFixture();
            Vector3 start = fixture.Body.RootBody.position;
            fixture.Motors.SetHighLevelCommands(1f, 0f, 0.5f, 0.8f);

            float dt = Phase2ExitCriteria.AutomatedFixedDeltaTime;
            for (int i = 0; i < Phase2ExitCriteria.AutomatedManeuverFixedSteps; i++)
            {
                fixture.Body.FixedSensorUpdate();
                fixture.Motors.Tick(dt);
                Physics.Simulate(dt);
            }

            float forwardDelta = Vector3.Dot(
                fixture.Body.RootBody.position - start,
                Vector3.forward);
            Assert.Greater(forwardDelta, Phase2ExitCriteria.WalkMinDisplacement);
            Assert.IsFalse(fixture.Body.HasFallen);
        }

        [Test]
        public void Turn_ChangesYaw()
        {
            using var fixture = new LocomotionFixture();
            float startYaw = fixture.Body.RootBody.rotation.eulerAngles.y;
            fixture.Motors.SetHighLevelCommands(0.2f, 1f, 0.5f, 0.8f);

            float dt = Phase2ExitCriteria.AutomatedFixedDeltaTime;
            for (int i = 0; i < Phase2ExitCriteria.AutomatedManeuverFixedSteps; i++)
            {
                fixture.Body.FixedSensorUpdate();
                fixture.Motors.Tick(dt);
                Physics.Simulate(dt);
            }

            float endYaw = fixture.Body.RootBody.rotation.eulerAngles.y;
            float delta = Mathf.Abs(Mathf.DeltaAngle(startYaw, endYaw));
            Assert.Greater(delta, Phase2ExitCriteria.TurnMinYawDegrees);
            Assert.IsFalse(fixture.Body.HasFallen);
        }

        [Test]
        public void Disturbance_RecoversTowardUpright()
        {
            using var fixture = new LocomotionFixture();
            fixture.Motors.ApplyStanding();

            float dt = Phase2ExitCriteria.AutomatedFixedDeltaTime;
            for (int i = 0; i < 100; i++)
            {
                fixture.Body.FixedSensorUpdate();
                fixture.Motors.Tick(dt);
                Physics.Simulate(dt);
            }

            fixture.Body.ApplyDisturbanceImpulse(Vector3.right * Phase2ExitCriteria.DisturbanceImpulse);

            for (int i = 0; i < Phase2ExitCriteria.AutomatedManeuverFixedSteps; i++)
            {
                fixture.Body.FixedSensorUpdate();
                fixture.Motors.Tick(dt);
                Physics.Simulate(dt);
            }

            Assert.Less(fixture.Body.TiltDegrees, Phase2ExitCriteria.RecoveryMaxTiltDegrees);
            Assert.IsFalse(fixture.Body.HasFallen);
        }
    }

    public sealed class Phase2ExitCriteriaConstantTests
    {
        [Test]
        public void Targets_MatchPhase2Spec()
        {
            Assert.AreEqual(30f, Phase2ExitCriteria.StandSeconds, 1e-5f);
            Assert.AreEqual(10f, Phase2ExitCriteria.AutomatedStandSeconds, 1e-5f);
            Assert.Greater(Phase2ExitCriteria.WalkMinDisplacement, 0f);
            Assert.Greater(Phase2ExitCriteria.TurnMinYawDegrees, 0f);
        }
    }

    internal sealed class LocomotionFixture : System.IDisposable
    {
        private readonly SimulationMode _previousMode;

        public GameObject Floor { get; }
        public GameObject Fly { get; }
        public FlyBodyController Body { get; }
        public FlyMotorController Motors { get; }

        public LocomotionFixture()
        {
            _previousMode = Physics.simulationMode;
            Physics.simulationMode = SimulationMode.Script;

            Floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Floor.name = "floor";
            Floor.transform.position = new Vector3(0f, -0.5f, 0f);
            Floor.transform.localScale = new Vector3(8f, 1f, 8f);

            Fly = new GameObject("phase2_fly");
            Fly.transform.position = new Vector3(0f, 0.02f, 0f);
            var rb = Fly.AddComponent<Rigidbody>();
            rb.mass = 0.02f;
            rb.linearDamping = 0.2f;
            rb.angularDamping = 4f;
            rb.useGravity = false;
            rb.constraints =
                RigidbodyConstraints.FreezeRotationX |
                RigidbodyConstraints.FreezeRotationZ;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            var sphere = Fly.AddComponent<SphereCollider>();
            sphere.radius = 0.005f;

            Body = Fly.AddComponent<FlyBodyController>();
            Phase2TestHelpers.SetPrivate(Body, "rootBody", rb);
            Phase2TestHelpers.SetPrivate(Body, "legs", new LegController[0]);

            var balance = Fly.AddComponent<BalanceController>();
            balance.Bind(rb);
            balance.TargetHeight = 0.02f;
            Body.BindBalance(balance);

            var config = ExperimentConfiguration.Default();
            config.LocomotionEnabled = true;
            config.FlySpawnPosition = new Vector3(0f, 0.02f, 0f);
            Body.Configure(config);
            Body.ResetPose(config.FlySpawnPosition, Quaternion.identity);

            Motors = Fly.AddComponent<FlyMotorController>();
            Motors.Bind(Body, null);

            Physics.SyncTransforms();
        }

        public void Dispose()
        {
            Physics.simulationMode = _previousMode;
            if (Fly != null)
            {
                Object.DestroyImmediate(Fly);
            }

            if (Floor != null)
            {
                Object.DestroyImmediate(Floor);
            }
        }
    }

    internal static class Phase2TestHelpers
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
