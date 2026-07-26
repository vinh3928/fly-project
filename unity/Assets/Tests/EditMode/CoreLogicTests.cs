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
        public void Concentration_UsesHardCutoffWithInnerFalloff()
        {
            const float radius = 0.100f;
            float atZero = ChemicalSource.ComputeConcentration(1f, 0f, radius);
            float atEdge = ChemicalSource.ComputeConcentration(1f, radius, radius);
            float justOutside = ChemicalSource.ComputeConcentration(1f, radius + 1e-4f, radius);
            float far = ChemicalSource.ComputeConcentration(1f, 2f, radius);

            Assert.AreEqual(1f, atZero, 1e-5f);
            Assert.AreEqual(Mathf.Exp(-1f), atEdge, 1e-5f);
            Assert.AreEqual(0f, justOutside, 1e-5f);
            Assert.AreEqual(0f, far, 1e-5f);
        }

        [Test]
        public void DefaultChemicalSpread_IsHardCutoffAtOneTenthMeter()
        {
            var config = ExperimentConfiguration.Default();
            float radius = config.ResolvedChemicalSpread;

            Assert.AreEqual(
                ExperimentConfiguration.DefaultChemicalSpread,
                radius,
                1e-5f);
            Assert.AreEqual(0.100f, radius, 1e-5f);
            Assert.AreEqual(
                Mathf.Exp(-1f),
                ChemicalSource.ComputeConcentration(1f, radius, radius),
                1e-5f);
            Assert.AreEqual(
                0f,
                ChemicalSource.ComputeConcentration(1f, radius + 1e-4f, radius),
                1e-5f);

            config.ChemicalSpread = 0.001f;
            Assert.AreEqual(
                ExperimentConfiguration.DefaultChemicalSpread,
                config.ResolvedChemicalSpread,
                1e-5f);
        }

        [Test]
        public void SugarAndBitter_AlsoWriteOdorChannel()
        {
            const float radius = 0.100f;
            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = Vector3.zero;
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, radius);

            var bitterGo = new GameObject("bitter");
            bitterGo.transform.position = new Vector3(radius, 0f, 0f);
            var bitter = bitterGo.AddComponent<ChemicalSource>();
            bitter.Configure(ChemicalSource.SourceType.Bitter, 1f, radius);

            ChemicalChannels atSugar = sugar.SampleAt(Vector3.zero);
            Assert.AreEqual(1f, atSugar.Sugar, 1e-5f);
            Assert.AreEqual(0f, atSugar.Bitter, 1e-5f);
            Assert.AreEqual(1f, atSugar.Odor, 1e-5f);

            float expected = Mathf.Exp(-1f);
            ChemicalChannels atEdge = sugar.SampleAt(new Vector3(radius, 0f, 0f));
            Assert.AreEqual(expected, atEdge.Sugar, 1e-5f);
            Assert.AreEqual(expected, atEdge.Odor, 1e-5f);

            ChemicalChannels outside = sugar.SampleAt(new Vector3(radius + 0.001f, 0f, 0f));
            Assert.AreEqual(0f, outside.Sugar, 1e-5f);
            Assert.AreEqual(0f, outside.Odor, 1e-5f);

            ChemicalChannels atBitter = bitter.SampleAt(bitterGo.transform.position);
            Assert.AreEqual(1f, atBitter.Bitter, 1e-5f);
            Assert.AreEqual(0f, atBitter.Sugar, 1e-5f);
            Assert.AreEqual(1f, atBitter.Odor, 1e-5f);

            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(bitterGo);
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
