using UnityEngine;

namespace FlySimulation.Fly.Motors
{
    /// <summary>
    /// Thin wrapper around proboscis extension for motor side.
    /// Must live in its own file: Unity only serializes MonoBehaviours whose class matches the file name.
    /// </summary>
    public sealed class ProboscisActuator : MonoBehaviour
    {
        [SerializeField] private Sensors.ProboscisSensor sensor;

        public Sensors.ProboscisSensor Sensor => sensor;

        public void Bind(Sensors.ProboscisSensor proboscisSensor)
        {
            sensor = proboscisSensor;
        }

        public void SetExtension(float normalized)
        {
            if (sensor != null)
            {
                sensor.SetExtension(normalized);
            }
        }
    }
}
