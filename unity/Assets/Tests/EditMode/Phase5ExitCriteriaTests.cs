using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class Phase5ExitCriteriaTests
    {
        [Test]
        public void Constants_MatchPythonParity()
        {
            Assert.AreEqual(197, Phase5ExitCriteria.ExpectedAgentVectorSize);
            Assert.AreEqual(SensorManager.ExpectedVectorSize, Phase5ExitCriteria.ExpectedAgentVectorSize);
            Assert.IsFalse(Phase5ExitCriteria.AgentObsIncludesTargetXyz);
            Assert.AreEqual(3, Phase5ExitCriteria.DefaultObstacleCount);
            Assert.AreEqual(0.08f, Phase5ExitCriteria.SuccessRadius, 1e-5f);
        }

        [Test]
        public void VisualTargetSpawn_IsDeterministicForSeed()
        {
            Vector3 chamber = new Vector3(0.1f, 0.05f, 0.1f);
            Vector3 a = DeterministicLayout.GenerateVisualTargetPosition(42, chamber, 0.02f, 0.006f);
            Vector3 b = DeterministicLayout.GenerateVisualTargetPosition(42, chamber, 0.02f, 0.006f);
            Vector3 c = DeterministicLayout.GenerateVisualTargetPosition(43, chamber, 0.02f, 0.006f);
            Assert.AreEqual(a.x, b.x, 1e-6f);
            Assert.AreEqual(a.z, b.z, 1e-6f);
            Assert.IsFalse(Mathf.Approximately(a.x, c.x) && Mathf.Approximately(a.z, c.z));
        }

        [Test]
        public void VisualNavMetrics_DistanceHelpers()
        {
            float d = VisualNavMetrics.DistanceXZ(new Vector3(0f, 1f, 0f), new Vector3(0.3f, 9f, 0.4f));
            Assert.AreEqual(0.5f, d, 1e-5f);
            float wall = VisualNavMetrics.NearestAxisAlignedWallDistance(new Vector3(0.01f, 0f, 0f), 0.05f, 0.05f);
            Assert.AreEqual(0.04f, wall, 1e-5f);
        }

        [Test]
        public void AgentVectorSize_DoesNotReserveTargetXyzSlots()
        {
            // Contract: vector obs layout is joints/contacts/chem/internal/body only.
            Assert.AreEqual(197, SensorManager.ExpectedVectorSize);
            Assert.IsFalse(Phase5ExitCriteria.AgentObsIncludesTargetXyz);
        }
    }
}
