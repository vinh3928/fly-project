using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Lives on the proboscis tip (trigger + kinematic Rigidbody). Unity only sends
    /// trigger messages to that GameObject, so forward them to the sensor.
    /// </summary>
    public sealed class ProboscisTipContact : MonoBehaviour
    {
        [SerializeField] private ProboscisSensor sensor;

        public void Bind(ProboscisSensor proboscis)
        {
            sensor = proboscis;
        }

        private void OnTriggerStay(Collider other)
        {
            if (sensor != null)
            {
                sensor.ReportContact(other);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (sensor != null)
            {
                sensor.ReportExit();
            }
        }
    }
}
