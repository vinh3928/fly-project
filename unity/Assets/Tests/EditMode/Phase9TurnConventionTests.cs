using FlySimulation.Fly.Brain;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class Phase9TurnConventionTests
    {
        [Test]
        public void MindTurnLeft_MapsToNegativeMotorBias()
        {
            // Mind turn_left candidate uses negative turn (−0.7).
            Assert.Less(LocomotionReward.MindTurnToMotorBias(-0.7f), 0f);
        }

        [Test]
        public void MindTurnRight_MapsToPositiveMotorBias()
        {
            Assert.Greater(LocomotionReward.MindTurnToMotorBias(0.7f), 0f);
        }

        [Test]
        public void MindTurnToMotorBias_PreservesSign()
        {
            Assert.AreEqual(0.38f, LocomotionReward.MindTurnToMotorBias(0.38f), 1e-5f);
            Assert.AreEqual(-0.38f, LocomotionReward.MindTurnToMotorBias(-0.38f), 1e-5f);
        }

        [Test]
        public void PositiveMotorBiasYaw_SwingsForwardTowardRight()
        {
            // FlyMotorController applies Quaternion.Euler(0, +yaw, 0) for positive turnBias.
            float motorBias = LocomotionReward.MindTurnToMotorBias(0.5f);
            Quaternion yaw = Quaternion.Euler(0f, motorBias * 30f, 0f);
            Vector3 newForward = yaw * Vector3.forward;
            Assert.Greater(Vector3.Dot(newForward, Vector3.right), 0f);
        }
    }
}
