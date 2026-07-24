using UnityEngine;

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Phase 9 locomotion / goal-tracking reward (architecture B motor layer).
    /// </summary>
    public static class LocomotionReward
    {
        public const int GoalDim = 5;

        /// <summary>
        /// Mind/Python turn: + = right. <see cref="Motors.FlyMotorController"/> turnBias: + = +Y yaw = left.
        /// </summary>
        public static float MindTurnToMotorBias(float mindTurn) => -mindTurn;

        public static float Compute(
            float[] goal,
            float[] action,
            float forwardProgress,
            float yawProgress,
            float upright,
            bool fell,
            float forwardProgressWeight = 1f,
            float goalTrackingWeight = 0.8f,
            float uprightWeight = 0.3f,
            float fallPenalty = 1f,
            float effortWeight = 0.02f,
            float aliveBonus = 0.001f)
        {
            if (goal == null || goal.Length == 0 || action == null || action.Length == 0)
            {
                return fell ? -fallPenalty : aliveBonus;
            }

            int n = Mathf.Min(Mathf.Min(goal.Length, action.Length), GoalDim);
            float trackingErr = 0f;
            for (int i = 0; i < n; i++)
            {
                trackingErr += Mathf.Abs(goal[i] - action[i]);
            }

            trackingErr = n > 0 ? trackingErr / n : 0f;

            float gFwd = goal.Length > 0 ? goal[0] : 0f;
            float gTurn = goal.Length > 1 ? goal[1] : 0f;
            float progress = forwardProgress * Mathf.Abs(gFwd) + yawProgress * Mathf.Abs(gTurn);

            float effort = 0f;
            int m = Mathf.Min(action.Length, GoalDim);
            for (int i = 0; i < m; i++)
            {
                effort += Mathf.Abs(action[i]);
            }

            effort = m > 0 ? effort / m : 0f;

            float reward =
                forwardProgressWeight * progress
                - goalTrackingWeight * trackingErr
                + uprightWeight * Mathf.Clamp01(upright)
                - effortWeight * effort
                + aliveBonus;

            if (fell)
            {
                reward -= fallPenalty;
            }

            return reward;
        }
    }
}
