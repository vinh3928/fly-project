using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class ChemistryTests
    {
        [Test]
        public void Concentration_UsesExponentialFalloff()
        {
            float atZero = ChemicalSource.ComputeConcentration(1f, 0f, 0.25f);
            float atSpread = ChemicalSource.ComputeConcentration(1f, 0.25f, 0.25f);
            float far = ChemicalSource.ComputeConcentration(1f, 2f, 0.25f);

            Assert.AreEqual(1f, atZero, 1e-5f);
            Assert.AreEqual(Mathf.Exp(-1f), atSpread, 1e-5f);
            Assert.Less(far, atSpread);
        }

        [Test]
        public void ChemicalChannels_Clamp01()
        {
            var channels = new ChemicalChannels { Sugar = 2f, Bitter = -1f, Water = 0.5f };
            channels = ChemicalChannels.Clamp01(channels);
            Assert.AreEqual(1f, channels.Sugar);
            Assert.AreEqual(0f, channels.Bitter);
            Assert.AreEqual(0.5f, channels.Water);
        }
    }

    public sealed class HomeostasisTests
    {
        [Test]
        public void Tick_ReducesEnergy()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.MetabolismRate = 0.1f;
            homeo.Configure(config);

            float before = homeo.Energy;
            homeo.Tick(1f, motorEffort: 0f, usingWings: false);
            Assert.Less(homeo.Energy, before);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void ConsumeSugar_IncreasesEnergy()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 0.2f;
            homeo.Configure(config);
            homeo.ConsumeSugar(0.3f);
            Assert.AreEqual(0.5f, homeo.Energy, 1e-5f);
            Object.DestroyImmediate(go);
        }
    }

    public sealed class JointTests
    {
        [Test]
        public void DistanceFromLimits_IsNormalized()
        {
            float mid = LegJoint.DistanceFromLimitsNormalized(0f, -60f, 60f);
            float near = LegJoint.DistanceFromLimitsNormalized(55f, -60f, 60f);
            Assert.Greater(mid, near);
            Assert.AreEqual(0.5f, mid, 1e-4f);
        }

        [Test]
        public void NormalizeAngle_WrapsAround()
        {
            Assert.AreEqual(-10f, LegJoint.NormalizeAngle(350f), 1e-4f);
            Assert.AreEqual(10f, LegJoint.NormalizeAngle(-350f), 1e-4f);
        }
    }

    public sealed class RewardTests
    {
        [Test]
        public void Reward_PenalizesDamageAndEffort()
        {
            var config = ExperimentConfiguration.Default();
            config.CuriosityWeight = 0f;
            config.EnergyWeight = 0f;
            config.HydrationWeight = 0f;
            config.DamageWeight = 2f;
            config.EffortWeight = 1f;
            config.InstabilityWeight = 0f;

            float reward = RewardCalculator.ComputeStatic(
                curiosity: 0f,
                energyImprovement: 0f,
                hydrationImprovement: 0f,
                damage: 0.5f,
                motorEffort: 0.2f,
                instability: 0f,
                config: config);

            Assert.AreEqual(-1.2f, reward, 1e-5f);
        }
    }

    public sealed class ExperimentConfigurationTests
    {
        [Test]
        public void Json_RoundTripsSeed()
        {
            var config = ExperimentConfiguration.Default();
            config.RandomSeed = 12345;
            config.ExperimentId = "test_exp";
            string json = config.ToJson();
            var restored = ExperimentConfiguration.FromJson(json);
            Assert.AreEqual(12345, restored.RandomSeed);
            Assert.AreEqual("test_exp", restored.ExperimentId);
        }
    }
}
