using FlySimulation.Fly.Brain;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class Phase9LocomotionRewardTests
    {
        [Test]
        public void TrackingError_IsPenalized()
        {
            float[] goal = { 1f, 0f, 0.5f, 0.5f, 0f };
            float[] good = { 1f, 0f, 0.5f, 0.5f, 0f };
            float[] bad = { -1f, 1f, 0f, 0f, 1f };
            float rGood = LocomotionReward.Compute(goal, good, forwardProgress: 0.05f, yawProgress: 0f, upright: 1f, fell: false);
            float rBad = LocomotionReward.Compute(goal, bad, forwardProgress: 0.05f, yawProgress: 0f, upright: 1f, fell: false);
            Assert.Greater(rGood, rBad);
        }

        [Test]
        public void Fall_AppliesPenalty()
        {
            float[] goal = { 0.5f, 0f, 0.5f, 0.5f, 0f };
            float[] action = { 0.5f, 0f, 0.5f, 0.5f, 0f };
            float ok = LocomotionReward.Compute(goal, action, 0.02f, 0f, 1f, fell: false);
            float fell = LocomotionReward.Compute(goal, action, 0.02f, 0f, 0f, fell: true);
            Assert.Less(fell, ok);
        }

        [Test]
        public void ForwardProgress_TowardGoal_IsRewarded()
        {
            float[] goal = { 1f, 0f, 0.5f, 0.5f, 0f };
            float[] action = { 1f, 0f, 0.5f, 0.5f, 0f };
            float withProgress = LocomotionReward.Compute(goal, action, forwardProgress: 0.1f, yawProgress: 0f, upright: 1f, fell: false);
            float without = LocomotionReward.Compute(goal, action, forwardProgress: 0f, yawProgress: 0f, upright: 1f, fell: false);
            Assert.Greater(withProgress, without);
        }
    }
}
