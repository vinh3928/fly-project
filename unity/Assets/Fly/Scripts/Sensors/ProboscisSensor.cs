using FlySimulation.Environment;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Proboscis taste and extension state.
    /// </summary>
    public sealed class ProboscisSensor : MonoBehaviour
    {
        /// <summary>
        /// Extension at which the tip reaches the floor. Full extension also fully lowers the
        /// head, so the scene builder sizes reach from the lowered pose; any overshoot would
        /// push the tip through the thin puddle and out of contact.
        /// </summary>
        public const float FloorContactExtension = 1f;

        public const float TipRadius = 0.002f;

        /// <summary>Pitch (deg) of the proboscis relative to the fly body: local +Z → down.</summary>
        public const float BodyPitchDegrees = 90f;

        [SerializeField] private float maxExtension = 0.02f;
        [SerializeField] private Transform tip;

        private float _contactStamp = -1f;

        public float ExtensionNormalized { get; private set; }
        public float MaxExtension => maxExtension;
        public bool IsContacting { get; private set; }
        public ChemicalChannels Taste { get; private set; }

        /// <summary>Reach (m) so the tip touches the floor at <see cref="FloorContactExtension"/>.</summary>
        public static float MaxExtensionForDrop(float dropToFloor)
        {
            return Mathf.Max(0.001f, dropToFloor) / FloorContactExtension;
        }

        public void Configure(Transform tipTransform, float maxReach)
        {
            tip = tipTransform;
            maxExtension = Mathf.Max(0.001f, maxReach);
            SetExtension(ExtensionNormalized);
        }

        public void SetExtension(float normalized)
        {
            ExtensionNormalized = Mathf.Clamp01(normalized);
            if (tip != null)
            {
                tip.localPosition = new Vector3(0f, 0f, ExtensionNormalized * maxExtension);
            }
        }

        /// <summary>
        /// Trigger stay from the tip collider (relayed by <see cref="ProboscisTipContact"/>).
        /// Surfaces touched in the same physics step are merged so the floor does not mask food.
        /// </summary>
        public void ReportContact(Collider other)
        {
            if (other == null)
            {
                return;
            }

            var surface = other.GetComponent<SurfaceChemistry>() ??
                          other.GetComponentInParent<SurfaceChemistry>();
            if (surface == null)
            {
                return;
            }

            float now = Time.fixedTime;
            if (!Mathf.Approximately(now, _contactStamp) || !IsContacting)
            {
                _contactStamp = now;
                Taste = ChemicalChannels.Zero;
            }

            IsContacting = true;
            Taste = ChemicalChannels.Clamp01(Taste + surface.Channels);
        }

        public void ReportExit()
        {
            IsContacting = false;
            Taste = ChemicalChannels.Zero;
        }

        private void OnTriggerStay(Collider other) => ReportContact(other);

        private void OnTriggerExit(Collider other) => ReportExit();

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
