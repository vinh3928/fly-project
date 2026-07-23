using FlySimulation.Environment;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Proboscis taste and extension state.
    /// </summary>
    public sealed class ProboscisSensor : MonoBehaviour
    {
        [SerializeField] private float maxExtension = 0.02f;
        [SerializeField] private Transform tip;

        public float ExtensionNormalized { get; private set; }
        public bool IsContacting { get; private set; }
        public ChemicalChannels Taste { get; private set; }

        public void SetExtension(float normalized)
        {
            ExtensionNormalized = Mathf.Clamp01(normalized);
            if (tip != null)
            {
                tip.localPosition = new Vector3(0f, 0f, ExtensionNormalized * maxExtension);
            }
        }

        private void OnTriggerStay(Collider other)
        {
            var surface = other.GetComponent<SurfaceChemistry>() ??
                          other.GetComponentInParent<SurfaceChemistry>();
            if (surface == null)
            {
                IsContacting = false;
                Taste = ChemicalChannels.Zero;
                return;
            }

            IsContacting = true;
            Taste = surface.Channels;
        }

        private void OnTriggerExit(Collider other)
        {
            IsContacting = false;
            Taste = ChemicalChannels.Zero;
        }

        public void CopyObservation(float[] buffer, int offset)
        {
            buffer[offset] = ExtensionNormalized;
            buffer[offset + 1] = IsContacting ? 1f : 0f;
            TasteGating.CopyGated(IsContacting, Taste, buffer, offset + 2);
        }

        /// <summary>Gated taste used by the recorder (zero unless contacting).</summary>
        public ChemicalChannels GatedTaste => TasteGating.Gate(IsContacting, Taste);
    }
}
