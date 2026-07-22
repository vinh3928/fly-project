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

        private void FixedUpdate()
        {
            if (chemicalField == null)
            {
                Odor = ChemicalChannels.Zero;
                return;
            }

            Odor = chemicalField.Sample(transform.position);
        }

        public void CopyObservation(float[] buffer, int offset)
        {
            Odor.CopyTo(buffer, offset);
        }
    }
}
