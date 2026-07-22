using System;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Eight-channel chemical concentration schema shared by surfaces and taste sensors.
    /// </summary>
    [Serializable]
    public struct ChemicalChannels
    {
        public float Sugar;
        public float Bitter;
        public float Salt;
        public float Acid;
        public float Water;
        public float Protein;
        public float Toxin;
        public float Odor;

        public static ChemicalChannels Zero => default;

        public float[] ToArray()
        {
            return new[] { Sugar, Bitter, Salt, Acid, Water, Protein, Toxin, Odor };
        }

        public void CopyTo(float[] buffer, int offset = 0)
        {
            buffer[offset] = Sugar;
            buffer[offset + 1] = Bitter;
            buffer[offset + 2] = Salt;
            buffer[offset + 3] = Acid;
            buffer[offset + 4] = Water;
            buffer[offset + 5] = Protein;
            buffer[offset + 6] = Toxin;
            buffer[offset + 7] = Odor;
        }

        public static ChemicalChannels Clamp01(ChemicalChannels c)
        {
            c.Sugar = Mathf.Clamp01(c.Sugar);
            c.Bitter = Mathf.Clamp01(c.Bitter);
            c.Salt = Mathf.Clamp01(c.Salt);
            c.Acid = Mathf.Clamp01(c.Acid);
            c.Water = Mathf.Clamp01(c.Water);
            c.Protein = Mathf.Clamp01(c.Protein);
            c.Toxin = Mathf.Clamp01(c.Toxin);
            c.Odor = Mathf.Clamp01(c.Odor);
            return c;
        }

        public static ChemicalChannels operator +(ChemicalChannels a, ChemicalChannels b)
        {
            return new ChemicalChannels
            {
                Sugar = a.Sugar + b.Sugar,
                Bitter = a.Bitter + b.Bitter,
                Salt = a.Salt + b.Salt,
                Acid = a.Acid + b.Acid,
                Water = a.Water + b.Water,
                Protein = a.Protein + b.Protein,
                Toxin = a.Toxin + b.Toxin,
                Odor = a.Odor + b.Odor
            };
        }

        public static ChemicalChannels operator *(ChemicalChannels a, float scale)
        {
            return new ChemicalChannels
            {
                Sugar = a.Sugar * scale,
                Bitter = a.Bitter * scale,
                Salt = a.Salt * scale,
                Acid = a.Acid * scale,
                Water = a.Water * scale,
                Protein = a.Protein * scale,
                Toxin = a.Toxin * scale,
                Odor = a.Odor * scale
            };
        }
    }

    /// <summary>
    /// Surface-mounted chemical concentrations for taste sensing.
    /// </summary>
    public sealed class SurfaceChemistry : MonoBehaviour
    {
        public float Sugar;
        public float Bitter;
        public float Salt;
        public float Acid;
        public float Water;
        public float Protein;
        public float Toxin;
        public float Odor;

        public ChemicalChannels Channels =>
            new ChemicalChannels
            {
                Sugar = Sugar,
                Bitter = Bitter,
                Salt = Salt,
                Acid = Acid,
                Water = Water,
                Protein = Protein,
                Toxin = Toxin,
                Odor = Odor
            };

        public void SetChannels(ChemicalChannels channels)
        {
            Sugar = channels.Sugar;
            Bitter = channels.Bitter;
            Salt = channels.Salt;
            Acid = channels.Acid;
            Water = channels.Water;
            Protein = channels.Protein;
            Toxin = channels.Toxin;
            Odor = channels.Odor;
        }
    }
}
