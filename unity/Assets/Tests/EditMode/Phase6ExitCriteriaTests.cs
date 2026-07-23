using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class Phase6ExitCriteriaTests
    {
        [Test]
        public void Constants_MatchVectorContract()
        {
            Assert.AreEqual(197, Phase6ExitCriteria.ExpectedAgentVectorSize);
            Assert.AreEqual(SensorManager.ExpectedVectorSize, Phase6ExitCriteria.ExpectedAgentVectorSize);
            Assert.IsFalse(Phase6ExitCriteria.AgentObsIncludesFoodXyz);
        }

        [Test]
        public void ConsumeSugar_RaisesEnergyAndLowersHunger()
        {
            var go = new GameObject("homeo");
            var homeo = go.AddComponent<FlyHomeostasis>();
            homeo.Configure(ExperimentConfiguration.Default());
            // Drive to low energy
            homeo.Configure(ExperimentConfiguration.Default());
            typeof(FlyHomeostasis).GetField(
                "energy",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                ?.SetValue(homeo, 0.3f);
            typeof(FlyHomeostasis).GetField(
                "hunger",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                ?.SetValue(homeo, 0.7f);

            float before = homeo.Energy;
            homeo.ConsumeSugar(0.2f);
            Assert.Greater(homeo.Energy, before);
            Assert.Less(homeo.Hunger, 0.7f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void ApplyBitterOrToxin_RaisesDamageAndToxin()
        {
            var go = new GameObject("homeo");
            var homeo = go.AddComponent<FlyHomeostasis>();
            homeo.Configure(ExperimentConfiguration.Default());
            Assert.AreEqual(0f, homeo.Damage, 1e-5f);
            homeo.ApplyBitterOrToxin(0.2f);
            Assert.Greater(homeo.Damage, 0f);
            Assert.Greater(homeo.ToxinExposure, 0f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void FoodSearchMetrics_TracksConsumeAndBitter()
        {
            var go = new GameObject("metrics");
            var metrics = go.AddComponent<FoodSearchMetrics>();
            metrics.ResetMetrics(0.4f);
            metrics.RegisterSugarConsume(0.1f);
            metrics.RegisterBitterContact();
            metrics.UpdateEnergyDelta(0.55f);
            Assert.AreEqual(0.1f, metrics.SugarConsumed, 1e-5f);
            Assert.AreEqual(1, metrics.BitterContacts);
            Assert.AreEqual(0.15f, metrics.EnergyDelta, 1e-5f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void GatedTaste_ZerosWhenNotContacting()
        {
            var raw = new ChemicalChannels { Sugar = 0.9f, Bitter = 0.4f };
            ChemicalChannels gated = TasteGating.Gate(false, raw);
            Assert.AreEqual(0f, gated.Sugar, 1e-5f);
            Assert.AreEqual(0f, gated.Bitter, 1e-5f);
        }
    }
}
