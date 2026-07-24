using FlySimulation.Fly.Brain;
using NUnit.Framework;

namespace FlySimulation.Tests.EditMode
{
    public sealed class Phase9TurnConventionTests
    {
        [Test]
        public void MindTurnLeft_MapsToPositiveMotorBias()
        {
            // Mind turn_left candidate uses negative turn (−0.7).
            Assert.Greater(LocomotionReward.MindTurnToMotorBias(-0.7f), 0f);
        }

        [Test]
        public void MindTurnRight_MapsToNegativeMotorBias()
        {
            Assert.Less(LocomotionReward.MindTurnToMotorBias(0.7f), 0f);
        }

        [Test]
        public void MindTurnToMotorBias_IsNegation()
        {
            Assert.AreEqual(-0.38f, LocomotionReward.MindTurnToMotorBias(0.38f), 1e-5f);
            Assert.AreEqual(0.38f, LocomotionReward.MindTurnToMotorBias(-0.38f), 1e-5f);
        }
    }
}
