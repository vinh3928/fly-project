using FlySimulation.Environment;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Foot contact, force, surface normal, and taste chemistry.
    /// Uses collision contacts when available, with a downward raycast fallback
    /// for Phase 2 kinematic legs (trigger feet / no per-foot rigidbodies).
    /// </summary>
    public sealed class FootContactSensor : MonoBehaviour
    {
        [SerializeField] private float forceScale = 10f;
        [SerializeField] private float raycastLength = 0.03f;
        [SerializeField] private LayerMask groundMask = ~0;
        [SerializeField] private bool enableRaycastFallback = true;

        public bool IsTouching { get; private set; }
        public float NormalForce { get; private set; }
        public Vector3 SurfaceNormal { get; private set; } = Vector3.up;
        public Vector3 RelativeVelocity { get; private set; }
        public string SurfaceType { get; private set; } = "none";
        public SurfaceChemistry CurrentSurface { get; private set; }
        public ChemicalChannels Taste { get; private set; }

        private int _contactCount;
        private float _accumulatedForce;
        private Vector3 _accumulatedNormal;
        private SurfaceChemistry _pendingSurface;
        private string _pendingSurfaceType = "none";

        private void FixedUpdate()
        {
            if (_contactCount > 0)
            {
                IsTouching = true;
                NormalForce = Mathf.Clamp01(_accumulatedForce / forceScale);
                SurfaceNormal = _accumulatedNormal.sqrMagnitude > 1e-6f
                    ? _accumulatedNormal.normalized
                    : Vector3.up;
                CurrentSurface = _pendingSurface;
                SurfaceType = _pendingSurfaceType;
                Taste = CurrentSurface != null ? CurrentSurface.Channels : ChemicalChannels.Zero;
            }
            else if (enableRaycastFallback && TryRaycastContact(out RaycastHit hit, out SurfaceChemistry surface))
            {
                IsTouching = true;
                NormalForce = Mathf.Clamp01(1f - hit.distance / Mathf.Max(1e-4f, raycastLength));
                SurfaceNormal = hit.normal;
                RelativeVelocity = Vector3.zero;
                CurrentSurface = surface;
                SurfaceType = hit.collider != null ? hit.collider.tag : "none";
                Taste = surface != null ? surface.Channels : ChemicalChannels.Zero;
            }
            else
            {
                IsTouching = false;
                NormalForce = 0f;
                SurfaceNormal = Vector3.up;
                RelativeVelocity = Vector3.zero;
                CurrentSurface = null;
                SurfaceType = "none";
                Taste = ChemicalChannels.Zero;
            }

            _contactCount = 0;
            _accumulatedForce = 0f;
            _accumulatedNormal = Vector3.zero;
            _pendingSurface = null;
            _pendingSurfaceType = "none";
        }

        /// <summary>Pure raycast probe used by EditMode tests.</summary>
        public static bool ProbeGround(
            Vector3 origin,
            float maxDistance,
            LayerMask mask,
            out float hitDistance,
            out Vector3 hitNormal)
        {
            if (Physics.Raycast(origin, Vector3.down, out RaycastHit hit, maxDistance, mask))
            {
                hitDistance = hit.distance;
                hitNormal = hit.normal;
                return true;
            }

            hitDistance = maxDistance;
            hitNormal = Vector3.up;
            return false;
        }

        private bool TryRaycastContact(out RaycastHit hit, out SurfaceChemistry surface)
        {
            surface = null;
            if (!Physics.Raycast(transform.position, Vector3.down, out hit, raycastLength, groundMask))
            {
                return false;
            }

            surface = hit.collider.GetComponent<SurfaceChemistry>();
            if (surface == null)
            {
                surface = hit.collider.GetComponentInParent<SurfaceChemistry>();
            }

            return true;
        }

        private void OnCollisionStay(Collision collision)
        {
            _contactCount++;
            _accumulatedForce += collision.impulse.magnitude / Mathf.Max(Time.fixedDeltaTime, 1e-4f);
            if (collision.contactCount > 0)
            {
                ContactPoint contact = collision.GetContact(0);
                _accumulatedNormal += contact.normal;
                RelativeVelocity = collision.relativeVelocity;
            }

            var surface = collision.collider.GetComponent<SurfaceChemistry>();
            if (surface == null)
            {
                surface = collision.collider.GetComponentInParent<SurfaceChemistry>();
            }

            if (surface != null)
            {
                _pendingSurface = surface;
                _pendingSurfaceType = collision.collider.tag;
            }
            else
            {
                _pendingSurfaceType = collision.collider.tag;
            }
        }

        public void CopyContactObservation(float[] buffer, int offset)
        {
            buffer[offset] = IsTouching ? 1f : 0f;
            buffer[offset + 1] = NormalForce;
            buffer[offset + 2] = SurfaceNormal.x;
            buffer[offset + 3] = SurfaceNormal.y;
            buffer[offset + 4] = SurfaceNormal.z;
        }

        public void CopyTasteObservation(float[] buffer, int offset)
        {
            TasteGating.CopyGated(IsTouching, Taste, buffer, offset);
        }
    }
}
