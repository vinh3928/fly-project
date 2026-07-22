using UnityEngine;
#if UNITY_MLAGENTS
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
#endif

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Minimal ML-Agents agent used by the standalone training player so Python can
    /// launch Unity and exchange observations/actions (Phase 0 exit criterion).
    /// The full chamber fly remains in the editor Chamber scene.
    /// </summary>
#if UNITY_MLAGENTS
    public sealed class MinimalLaunchAgent : Agent
#else
    public sealed class MinimalLaunchAgent : MonoBehaviour
#endif
    {
        public const int ObservationSize = 197;
        public const int ActionSize = 5;

#if UNITY_MLAGENTS
        public override void CollectObservations(VectorSensor sensor)
        {
            for (int i = 0; i < ObservationSize; i++)
            {
                sensor.AddObservation(0f);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // No-op for Phase 0 launch verification.
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuous = actionsOut.ContinuousActions;
            for (int i = 0; i < continuous.Length; i++)
            {
                continuous[i] = 0f;
            }
        }
#endif
    }
}
