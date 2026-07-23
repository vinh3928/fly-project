using FlySimulation.Environment;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Phase 3: taste channels are valid only during contact.
    /// </summary>
    public static class TasteGating
    {
        public static ChemicalChannels Gate(bool isContacting, ChemicalChannels raw)
        {
            return isContacting ? raw : ChemicalChannels.Zero;
        }

        public static void CopyGated(bool isContacting, ChemicalChannels raw, float[] buffer, int offset)
        {
            Gate(isContacting, raw).CopyTo(buffer, offset);
        }

        public static bool HasNonZeroTaste(ChemicalChannels channels)
        {
            return channels.Sugar > 0f ||
                   channels.Bitter > 0f ||
                   channels.Salt > 0f ||
                   channels.Acid > 0f ||
                   channels.Water > 0f ||
                   channels.Protein > 0f ||
                   channels.Toxin > 0f ||
                   channels.Odor > 0f;
        }
    }
}
