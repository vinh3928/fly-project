using FlySimulation.Environment;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Antenna odor sampling from the chemical field.
    /// </summary>
    public sealed class AntennaOdorSensor : MonoBehaviour
    {
        [SerializeField] private ChemicalFieldSystem chemicalField;
        [SerializeField] private bool isLeft = true;

        public bool IsLeft => isLeft;
        public ChemicalChannels Odor { get; private set; }

        public void BindField(ChemicalFieldSystem field)
        {
            chemicalField = field;
        }

        /// <summary>
        /// Sample the bound field at the antenna pose (also called from FixedUpdate).
        /// Exposed so EditMode tests can measure odor without waiting on physics ticks.
        /// </summary>
        public ChemicalChannels SampleNow()
        {
            if (chemicalField == null)
            {
                Odor = ChemicalChannels.Zero;
                return Odor;
            }

            Odor = chemicalField.Sample(transform.position);
            return Odor;
        }

        private void FixedUpdate()
        {
            SampleNow();
        }

        public void CopyObservation(float[] buffer, int offset)
        {
            Odor.CopyTo(buffer, offset);
        }
    }
}
