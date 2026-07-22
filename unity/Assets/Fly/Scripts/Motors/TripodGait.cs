using UnityEngine;

namespace FlySimulation.Fly.Motors
{
    /// <summary>
    /// Pure tripod-gait helpers shared by the motor controller and EditMode tests.
    /// </summary>
    public static class TripodGait
    {
        /// <summary>
        /// Returns swing angle (degrees) for leg index 0..5 given gait phase and commands.
        /// Even indices form one tripod; odd indices the other (π phase offset).
        /// </summary>
        public static void ComputeJointTargets(
            int legIndex,
            float phaseRadians,
            float walkForward,
            float turnBias,
            float bodyHeightCommand,
            float stepAmplitudeDegrees,
            out float coxaDeg,
            out float femurDeg,
            out float tibiaDeg)
        {
            bool left = legIndex < 3;
            float side = left ? 1f : -1f;
            float tripodePhase = (legIndex % 2 == 0) ? phaseRadians : phaseRadians + Mathf.PI;
            float swing = Mathf.Sin(tripodePhase) * stepAmplitudeDegrees * walkForward;
            float turn = turnBias * side * 10f;
            float heightLift = Mathf.Lerp(8f, -4f, Mathf.Clamp01(bodyHeightCommand));
            float lift = Mathf.Max(0f, Mathf.Sin(tripodePhase)) * 10f;

            coxaDeg = swing + turn;
            femurDeg = -40f + heightLift + lift;
            tibiaDeg = 70f - heightLift;
        }

        public static float AdvancePhase(float phaseRadians, float dt, float stepFrequency, float walkForward)
        {
            return phaseRadians + dt * stepFrequency * Mathf.Max(0.1f, Mathf.Abs(walkForward)) * Mathf.PI * 2f;
        }
    }
}
