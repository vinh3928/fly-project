using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class SensorManagerTests
    {
        [Test]
        public void ExpectedVectorSize_MatchesLayoutFormula()
        {
            int expected =
                SensorManager.JointCount * 4 +
                SensorManager.FootCount * 5 +
                SensorManager.FootCount * SensorManager.ChemicalChannels +
                SensorManager.ChemicalChannels * 2 +
                (2 + SensorManager.ChemicalChannels) +
                SensorManager.InternalStateCount +
                SensorManager.BodyStateCount;

            Assert.AreEqual(197, SensorManager.ExpectedVectorSize);
            Assert.AreEqual(expected, SensorManager.ExpectedVectorSize);
            Assert.AreEqual(18, SensorManager.JointCount);
            Assert.AreEqual(6, SensorManager.FootCount);
            Assert.AreEqual(8, SensorManager.ChemicalChannels);
            Assert.AreEqual(8, SensorManager.InternalStateCount);
            Assert.AreEqual(13, SensorManager.BodyStateCount);
        }

        [Test]
        public void CollectVectorObservation_Unbound_ReturnsExpectedLength()
        {
            var go = new GameObject("sensors");
            var manager = go.AddComponent<SensorManager>();

            int length = manager.CollectVectorObservation(0.005f);

            Assert.AreEqual(SensorManager.ExpectedVectorSize, length);
            Assert.AreEqual(SensorManager.ExpectedVectorSize, manager.VectorObservationLength);
            Assert.GreaterOrEqual(manager.VectorObservation.Length, length);
            Object.DestroyImmediate(go);
        }
    }

    public sealed class ChemicalChannelsAlgebraTests
    {
        [Test]
        public void AddAndScale_ThenClamp()
        {
            var a = new ChemicalChannels { Sugar = 0.6f, Bitter = 0.5f };
            var b = new ChemicalChannels { Sugar = 0.7f, Water = 0.2f };
            ChemicalChannels summed = a + b;
            ChemicalChannels scaled = summed * 0.5f;
            ChemicalChannels clamped = ChemicalChannels.Clamp01(summed);

            Assert.AreEqual(1.3f, summed.Sugar, 1e-5f);
            Assert.AreEqual(0.65f, scaled.Sugar, 1e-5f);
            Assert.AreEqual(1f, clamped.Sugar, 1e-5f);
            Assert.AreEqual(0.5f, clamped.Bitter, 1e-5f);
            Assert.AreEqual(0.2f, clamped.Water, 1e-5f);
        }

        [Test]
        public void CopyTo_AndToArray_PreserveChannelOrder()
        {
            var channels = new ChemicalChannels
            {
                Sugar = 0.1f,
                Bitter = 0.2f,
                Salt = 0.3f,
                Acid = 0.4f,
                Water = 0.5f,
                Protein = 0.6f,
                Toxin = 0.7f,
                Odor = 0.8f
            };

            float[] array = channels.ToArray();
            var buffer = new float[10];
            channels.CopyTo(buffer, 2);

            CollectionAssert.AreEqual(
                new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f },
                array);
            Assert.AreEqual(0.1f, buffer[2], 1e-5f);
            Assert.AreEqual(0.8f, buffer[9], 1e-5f);
        }
    }

    public sealed class ChemicalFieldSystemTests
    {
        [Test]
        public void Sample_SumsRegisteredSourcesAndClamps()
        {
            var fieldGo = new GameObject("field");
            var field = fieldGo.AddComponent<ChemicalFieldSystem>();

            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = Vector3.zero;
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, 0.25f);

            var bitterGo = new GameObject("bitter");
            bitterGo.transform.position = Vector3.zero;
            var bitter = bitterGo.AddComponent<ChemicalSource>();
            bitter.Configure(ChemicalSource.SourceType.Bitter, 1f, 0.25f);

            field.Register(sugar);
            field.Register(bitter);
            field.Register(sugar); // duplicate should be ignored

            ChemicalChannels sample = field.Sample(Vector3.zero);
            Assert.AreEqual(1f, sample.Sugar, 1e-5f);
            Assert.AreEqual(1f, sample.Bitter, 1e-5f);

            field.Unregister(bitter);
            sample = field.Sample(Vector3.zero);
            Assert.AreEqual(1f, sample.Sugar, 1e-5f);
            Assert.AreEqual(0f, sample.Bitter, 1e-5f);

            field.Clear();
            sample = field.Sample(Vector3.zero);
            Assert.AreEqual(0f, sample.Sugar, 1e-5f);

            Object.DestroyImmediate(fieldGo);
            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(bitterGo);
        }
    }

    public sealed class HomeostasisExtendedTests
    {
        [Test]
        public void CopyNormalized_WritesEightValues()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 0.8f;
            config.InitialHydration = 0.6f;
            config.TemperatureCelsius = 25f;
            homeo.Configure(config);

            var buffer = new float[8];
            homeo.CopyNormalized(buffer);

            Assert.AreEqual(0.8f, buffer[0], 1e-5f);
            Assert.AreEqual(0.6f, buffer[1], 1e-5f);
            Assert.AreEqual(Mathf.Clamp01((25f - 10f) / 30f), buffer[4], 1e-5f);
            Assert.AreEqual(SensorManager.InternalStateCount, buffer.Length);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void DamageWaterToxin_AndIsDepleted()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 1f;
            config.InitialHydration = 1f;
            homeo.Configure(config);

            Assert.IsFalse(homeo.IsDepleted);

            homeo.ConsumeWater(0.2f);
            Assert.AreEqual(1f, homeo.Hydration, 1e-5f); // already full, clamped

            homeo.ApplyCollisionDamage(0.4f);
            Assert.AreEqual(0.4f, homeo.Damage, 1e-5f);

            homeo.ApplyBitterOrToxin(0.7f);
            Assert.GreaterOrEqual(homeo.Damage, 1f - 1e-5f);
            Assert.IsTrue(homeo.IsDepleted);

            Assert.AreEqual(0.1f, FlyHomeostasis.ComputeEnergyDelta(0.5f, 0.6f), 1e-5f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Tick_WithWings_BurnsExtraEnergy()
        {
            var go = new GameObject("homeostasis");
            var walking = go.AddComponent<FlyHomeostasis>();
            var flying = new GameObject("flying").AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.MetabolismRate = 0.01f;
            config.WingEnergyCost = 0.5f;
            config.MovementEnergyCost = 0f;
            walking.Configure(config);
            flying.Configure(config);

            walking.Tick(1f, motorEffort: 0f, usingWings: false);
            flying.Tick(1f, motorEffort: 0f, usingWings: true);

            Assert.Less(flying.Energy, walking.Energy);
            Object.DestroyImmediate(go);
            Object.DestroyImmediate(flying.gameObject);
        }
    }

    public sealed class RewardCalculatorExtendedTests
    {
        [Test]
        public void Compute_TracksEnergyDelta_AndAppliesFellPenalty()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 0f;
            config.EnergyWeight = 1f;
            config.HydrationWeight = 0f;
            config.DamageWeight = 0f;
            config.EffortWeight = 0f;
            config.InstabilityWeight = 0f;
            config.InitialEnergy = 0.5f;
            config.MetabolismRate = 0f;
            config.MovementEnergyCost = 0f;
            homeo.Configure(config);

            var calculator = new RewardCalculator(config);
            calculator.Reset(homeo);

            homeo.ConsumeSugar(0.2f);
            float reward = calculator.Compute(homeo, curiosity: 0f, motorEffort: 0f, instability: 0f, fell: false);
            Assert.AreEqual(0.2f, reward, 1e-4f);

            float fellReward = calculator.Compute(homeo, curiosity: 0f, motorEffort: 0f, instability: 0f, fell: true);
            Assert.AreEqual(-1f, fellReward, 1e-4f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void ComputeStatic_ClampsCuriosity()
        {
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 1f;
            config.EnergyWeight = 0f;
            config.HydrationWeight = 0f;
            config.DamageWeight = 0f;
            config.EffortWeight = 0f;
            config.InstabilityWeight = 0f;

            float reward = RewardCalculator.ComputeStatic(
                curiosity: 5f,
                energyImprovement: 0f,
                hydrationImprovement: 0f,
                damage: 0f,
                motorEffort: 0f,
                instability: 0f,
                config: config);

            Assert.AreEqual(1f, reward, 1e-5f);
        }
    }

    public sealed class LegJointEdgeTests
    {
        [Test]
        public void DistanceFromLimits_AtBoundaryIsZero()
        {
            Assert.AreEqual(0f, LegJoint.DistanceFromLimitsNormalized(-60f, -60f, 60f), 1e-4f);
            Assert.AreEqual(0f, LegJoint.DistanceFromLimitsNormalized(60f, -60f, 60f), 1e-4f);
            Assert.AreEqual(0f, LegJoint.DistanceFromLimitsNormalized(100f, -60f, 60f), 1e-4f);
        }

        [Test]
        public void SetNormalizedTarget_MapsMidAndExtremes()
        {
            var go = new GameObject("joint");
            var joint = go.AddComponent<LegJoint>();
            joint.SetNormalizedTarget(0f);
            Assert.AreEqual(0f, joint.TargetDegrees, 1e-3f);
            joint.SetNormalizedTarget(1f);
            Assert.AreEqual(60f, joint.TargetDegrees, 1e-3f);
            joint.SetNormalizedTarget(-1f);
            Assert.AreEqual(-60f, joint.TargetDegrees, 1e-3f);
            Object.DestroyImmediate(go);
        }
    }

    public sealed class ProboscisSensorTests
    {
        [Test]
        public void SetExtension_ClampsAndCopyObservationLayout()
        {
            var go = new GameObject("proboscis");
            var sensor = go.AddComponent<ProboscisSensor>();
            sensor.SetExtension(2f);

            var buffer = new float[10];
            sensor.CopyObservation(buffer, 0);

            Assert.AreEqual(1f, buffer[0], 1e-5f);
            Assert.AreEqual(0f, buffer[1], 1e-5f); // not contacting
            for (int i = 2; i < 10; i++)
            {
                Assert.AreEqual(0f, buffer[i], 1e-5f);
            }

            Object.DestroyImmediate(go);
        }
    }

    public sealed class ExperimentConfigurationExtendedTests
    {
        [Test]
        public void Default_RewardWeights_MatchPythonDefaults()
        {
            var config = ExperimentConfiguration.Default();
            Assert.AreEqual(0.1f, config.CuriosityWeight, 1e-5f);
            Assert.AreEqual(1f, config.EnergyWeight, 1e-5f);
            Assert.AreEqual(1f, config.HydrationWeight, 1e-5f);
            Assert.AreEqual(2f, config.DamageWeight, 1e-5f);
            Assert.AreEqual(0.05f, config.EffortWeight, 1e-5f);
            Assert.AreEqual(0.5f, config.InstabilityWeight, 1e-5f);
            Assert.AreEqual(new Vector3(0.5f, 0.25f, 0.5f), config.ChamberSize);
            Assert.AreEqual(1f, config.BodyScale, 1e-5f);
            Assert.AreEqual(0f, config.FlySpawnPosition.x, 1e-5f);
            Assert.AreEqual(0f, config.FlySpawnPosition.z, 1e-5f);
            Assert.Greater(config.FlySpawnPosition.y, 0f);
            Assert.IsTrue(config.LocomotionEnabled);
        }

        [Test]
        public void Json_RoundTripsRewardWeights()
        {
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 0.25f;
            config.DamageWeight = 3.5f;
            var restored = ExperimentConfiguration.FromJson(config.ToJson());
            Assert.AreEqual(0.25f, restored.CuriosityWeight, 1e-5f);
            Assert.AreEqual(3.5f, restored.DamageWeight, 1e-5f);
        }
    }
}
