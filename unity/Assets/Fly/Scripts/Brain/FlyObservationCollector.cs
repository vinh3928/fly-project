using FlySimulation.Fly.Sensors;
using UnityEngine;

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Thin facade used by the agent to gather the latest observation payload.
    /// </summary>
    public sealed class FlyObservationCollector : MonoBehaviour
    {
        [SerializeField] private SensorManager sensorManager;

        public SensorManager Sensors => sensorManager;

        public void Bind(SensorManager manager)
        {
            sensorManager = manager;
        }

        public int Collect(float deltaTime)
        {
            return sensorManager != null ? sensorManager.CollectVectorObservation(deltaTime) : 0;
        }

        public float[] GetVectorObservation()
        {
            return sensorManager != null ? sensorManager.VectorObservation : System.Array.Empty<float>();
        }

        public int GetVectorLength()
        {
            return sensorManager != null ? sensorManager.VectorObservationLength : 0;
        }
    }
}
