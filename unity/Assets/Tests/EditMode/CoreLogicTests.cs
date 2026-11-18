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
        public void SugarPlume_IsOneAtCenterAndHundredthAtOneMeterRim()
        {
            float radius = ExperimentConfiguration.SugarPlumeRadius;
            float edge = ExperimentConfiguration.SugarPlumeEdgeFraction;
            Assert.AreEqual(1f, radius, 1e-6f);
            Assert.AreEqual(0.01f, edge, 1e-6f);

            Assert.AreEqual(1f, ChemicalSource.ComputeConcentration(1f, 0f, radius, edge), 1e-5f);
            Assert.AreEqual(0.1f, ChemicalSource.ComputeConcentration(1f, 0.5f, radius, edge), 1e-5f);
            Assert.AreEqual(0.01f, ChemicalSource.ComputeConcentration(1f, 1f, radius, edge), 1e-5f);
            Assert.AreEqual(0f, ChemicalSource.ComputeConcentration(1f, 1.001f, radius, edge), 1e-5f);

            var go = new GameObject("sugar");
            var sugar = go.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, radius, edge);
            ChemicalChannels farCorner = sugar.SampleAt(new Vector3(0.3f, 0f, 0.3f));
            Assert.Greater(farCorner.Sugar, 0f);
            Assert.AreEqual(farCorner.Sugar, farCorner.Odor, 1e-6f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void ThreeArgConfigure_KeepsExpMinusOneRim()
        {
            var go = new GameObject("bitter");
            var bitter = go.AddComponent<ChemicalSource>();
            bitter.Configure(ChemicalSource.SourceType.Bitter, 1f, 0.1f);
            Assert.AreEqual(ChemicalSource.DefaultEdgeFraction, bitter.EdgeFraction, 1e-6f);
            Assert.AreEqual(Mathf.Exp(-1f), bitter.ConcentrationAt(new Vector3(0.1f, 0f, 0f)), 1e-5f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Concentration_IsZeroWhenWallBlocksLineOfSight()
        {
            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = new Vector3(0f, 0f, 0f);
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, 1f, 0.01f);

            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = "Wall";
            wall.layer = ChamberWallHider.WallLayer;
            wall.transform.position = new Vector3(0f, 0f, 0.05f);
            wall.transform.localScale = new Vector3(0.2f, 0.1f, 0.01f);

            Vector3 beyond = new Vector3(0f, 0f, 0.12f);
            Vector3 clear = new Vector3(0.15f, 0f, 0f); // same side as source, no wall between

            Assert.IsTrue(ChemicalSource.IsOccludedByWall(sugarGo.transform.position, beyond));
            Assert.IsFalse(ChemicalSource.IsOccludedByWall(sugarGo.transform.position, clear));
            Assert.AreEqual(0f, sugar.ConcentrationAt(beyond), 1e-6f);
            Assert.AreEqual(0f, sugar.SampleAt(beyond).Odor, 1e-6f);
            Assert.Greater(sugar.ConcentrationAt(clear), 0f);
            Assert.Greater(sugar.SampleAt(clear).Odor, 0f);

            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(wall);
        }

        [Test]
        public void Concentration_PassesThroughOpenGapInWall()
        {
            var sugarGo = new GameObject("sugar");
            sugarGo.transform.position = new Vector3(0f, 0f, 0f);
            var sugar = sugarGo.AddComponent<ChemicalSource>();
            sugar.Configure(ChemicalSource.SourceType.Sugar, 1f, 1f, 0.01f);

            // Two wall slabs with a gap on the +Z axis (tunnel-style opening).
            var left = GameObject.CreatePrimitive(PrimitiveType.Cube);
            left.layer = ChamberWallHider.WallLayer;
            left.transform.position = new Vector3(-0.08f, 0f, 0.05f);
            left.transform.localScale = new Vector3(0.1f, 0.1f, 0.01f);
            var right = GameObject.CreatePrimitive(PrimitiveType.Cube);
            right.layer = ChamberWallHider.WallLayer;
            right.transform.position = new Vector3(0.08f, 0f, 0.05f);
            right.transform.localScale = new Vector3(0.1f, 0.1f, 0.01f);

            Vector3 throughGap = new Vector3(0f, 0f, 0.12f);
            Vector3 blocked = new Vector3(0.08f, 0f, 0.12f);

            Assert.IsFalse(ChemicalSource.IsOccludedByWall(sugarGo.transform.position, throughGap));
            Assert.IsTrue(ChemicalSource.IsOccludedByWall(sugarGo.transform.position, blocked));
            Assert.Greater(sugar.SampleAt(throughGap).Odor, 0f);
            Assert.AreEqual(0f, sugar.SampleAt(blocked).Odor, 1e-6f);

            Object.DestroyImmediate(sugarGo);
            Object.DestroyImmediate(left);
            Object.DestroyImmediate(right);
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

        [Test]
        public void Tick_NeverDropsEnergyBelowFloor()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 0.05f;
            config.MetabolismRate = 1f;
            homeo.Configure(config);
            for (int i = 0; i < 20; i++)
            {
                homeo.Tick(1f, motorEffort: 1f, usingWings: true);
            }

            Assert.AreEqual(FlyHomeostasis.MinEnergy, homeo.Energy, 1e-6f);
            Assert.IsFalse(homeo.IsDepleted);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void Configure_ClampsInitialEnergyToFloor()
        {
            var go = new GameObject("homeostasis");
            var homeo = go.AddComponent<FlyHomeostasis>();
            var config = ExperimentConfiguration.Default();
            config.InitialEnergy = 0f;
            homeo.Configure(config);
            Assert.AreEqual(FlyHomeostasis.MinEnergy, homeo.Energy, 1e-6f);
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
