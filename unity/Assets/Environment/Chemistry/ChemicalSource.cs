using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Point chemical source with hard-radius plume.
    /// Inside <see cref="Spread"/>: concentration = strength * edge^(distance / spread),
    /// so it is <c>strength</c> at the center and <c>strength * edge</c> at the rim.
    /// Default edge is exp(-1), i.e. strength * exp(-distance / spread).
    /// Outside that radius: concentration = 0 (hard cutoff).
    /// </summary>
    public sealed class ChemicalSource : MonoBehaviour
    {
        public enum SourceType
        {
            Sugar,
            Bitter,
            Salt,
            Acid,
            Water,
            Protein,
            Toxin,
            Odor
        }

        public static readonly float DefaultEdgeFraction = Mathf.Exp(-1f);

        [SerializeField] private SourceType sourceType = SourceType.Sugar;
        [SerializeField] private float strength = 1f;
        [SerializeField] private float spread = 0.100f;
        [Tooltip("Concentration fraction of strength at the plume rim (distance == spread).")]
        [SerializeField] private float edgeFraction = DefaultEdgeFraction;

        public SourceType Type => sourceType;
        public float Strength => strength;
        public float Spread => spread;
        public float EdgeFraction => edgeFraction;

        public void Configure(SourceType type, float sourceStrength, float sourceSpread)
        {
            Configure(type, sourceStrength, sourceSpread, DefaultEdgeFraction);
        }

        public void Configure(SourceType type, float sourceStrength, float sourceSpread, float sourceEdgeFraction)
        {
            sourceType = type;
            strength = sourceStrength;
            spread = Mathf.Max(1e-4f, sourceSpread);
            edgeFraction = ClampEdgeFraction(sourceEdgeFraction);
        }

        public float ConcentrationAt(Vector3 worldPosition)
        {
            float distance = Vector3.Distance(transform.position, worldPosition);
            return ComputeConcentration(strength, distance, spread, edgeFraction);
        }

        public ChemicalChannels SampleAt(Vector3 worldPosition)
        {
            float c = ConcentrationAt(worldPosition);
            var channels = ChemicalChannels.Zero;
            switch (sourceType)
            {
                case SourceType.Sugar:
                    channels.Sugar = c;
                    // Antenna "odor" channel mirrors sugar/bitter plumes so smell
                    // is detectable at range (taste channels stay contact/summary).
                    channels.Odor = c;
                    break;
                case SourceType.Bitter:
                    channels.Bitter = c;
                    channels.Odor = c;
                    break;
                case SourceType.Salt: channels.Salt = c; break;
                case SourceType.Acid: channels.Acid = c; break;
                case SourceType.Water: channels.Water = c; break;
                case SourceType.Protein: channels.Protein = c; break;
                case SourceType.Toxin: channels.Toxin = c; break;
                case SourceType.Odor: channels.Odor = c; break;
            }

            return channels;
        }

        public static float ComputeConcentration(float sourceStrength, float distance, float sourceSpread)
        {
            return ComputeConcentration(sourceStrength, distance, sourceSpread, DefaultEdgeFraction);
        }

        public static float ComputeConcentration(
            float sourceStrength,
            float distance,
            float sourceSpread,
            float sourceEdgeFraction)
        {
            float radius = Mathf.Max(1e-4f, sourceSpread);
            if (distance > radius)
            {
                return 0f;
            }

            float edge = ClampEdgeFraction(sourceEdgeFraction);
            return sourceStrength * Mathf.Pow(edge, distance / radius);
        }

        private static float ClampEdgeFraction(float value)
        {
            return Mathf.Clamp(value, 1e-6f, 1f);
        }
    }
}
