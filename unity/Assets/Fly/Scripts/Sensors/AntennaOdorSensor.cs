using FlySimulation.Environment;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Antenna odor sampling from the chemical field.
    /// </summary>
    public sealed class AntennaOdorSensor : MonoBehaviour
    {
        /// <summary>Half the left/right spacing where the stalks leave the head.</summary>
        public const float BaseHalfSpacing = 0.001f;

        /// <summary>Stalk vector of the original 4.4 mm antenna (right side); scaled by <see cref="StalkScale"/>.</summary>
        public const float UnitStalkX = 0.003f;
        public const float UnitStalkY = 0.003f;
        public const float UnitStalkZ = 0.001f;
        public const float StalkScale = 6f;

        /// <summary>Half the left/right spacing of the sensing tips in meters (38 mm apart).</summary>
        public const float HalfSpacing = BaseHalfSpacing + UnitStalkX * StalkScale;

        public const float StalkRadius = 0.0007f;
        public const float ClubRadius = 0.0015f;

        [SerializeField] private ChemicalFieldSystem chemicalField;
        [SerializeField] private bool isLeft = true;

        public bool IsLeft => isLeft;
        public ChemicalChannels Odor { get; private set; }

        /// <summary>
        /// Sensing tip on an unscaled mount at the head center; + X is the fly's right.
        /// Antenna meshes sit on <see cref="FlyEyeCamera.SelfLayer"/> so the eyes do not see them.
        /// </summary>
        public static Vector3 MountOffset(bool isLeft)
        {
            var stalk = new Vector3(isLeft ? -UnitStalkX : UnitStalkX, UnitStalkY, UnitStalkZ) * StalkScale;
            return BaseOffset(isLeft) + stalk;
        }

        /// <summary>Where the stalk leaves the head (same frame as <see cref="MountOffset"/>).</summary>
        public static Vector3 BaseOffset(bool isLeft)
        {
            return new Vector3(isLeft ? -BaseHalfSpacing : BaseHalfSpacing, 0.0035f, 0.003f);
        }

        public static float StalkLength => Vector3.Distance(BaseOffset(true), MountOffset(true));

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
