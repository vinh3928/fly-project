using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Eval-only visual navigation metrics. Not written into ML-Agents vector observations.
    /// </summary>
    public sealed class VisualNavMetrics : MonoBehaviour
    {
        [SerializeField] private Transform body;
        [SerializeField] private Transform visualTarget;
        [SerializeField] private float chamberHalfExtentX = 0.05f;
        [SerializeField] private float chamberHalfExtentZ = 0.05f;
        [SerializeField] private LayerMask obstacleMask = ~0;

        public int WallHits { get; private set; }
        public float DistanceToTarget { get; private set; }
        public float NearestWallDistance { get; private set; }
        public bool ReachedTarget { get; private set; }

        public void Bind(Transform bodyTransform, Transform target, Vector3 chamberSize)
        {
            body = bodyTransform;
            visualTarget = target;
            chamberHalfExtentX = chamberSize.x * 0.5f;
            chamberHalfExtentZ = chamberSize.z * 0.5f;
            WallHits = 0;
            ReachedTarget = false;
            Refresh();
        }

        public void ResetMetrics()
        {
            WallHits = 0;
            ReachedTarget = false;
            Refresh();
        }

        public void Refresh()
        {
            if (body == null)
            {
                DistanceToTarget = 0f;
                NearestWallDistance = 0f;
                return;
            }

            Vector3 pos = body.position;
            if (visualTarget != null)
            {
                DistanceToTarget = Vector3.Distance(
                    new Vector3(pos.x, 0f, pos.z),
                    new Vector3(visualTarget.position.x, 0f, visualTarget.position.z));
                if (DistanceToTarget <= Phase5ExitCriteria.SuccessRadius)
                {
                    ReachedTarget = true;
                }
            }

            float dx = chamberHalfExtentX - Mathf.Abs(pos.x);
            float dz = chamberHalfExtentZ - Mathf.Abs(pos.z);
            NearestWallDistance = Mathf.Min(dx, dz);
        }

        public void RegisterWallHit()
        {
            WallHits++;
        }

        /// <summary>Pure helper for EditMode tests / Python parity.</summary>
        public static float DistanceXZ(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        public static float NearestAxisAlignedWallDistance(Vector3 position, float halfX, float halfZ)
        {
            return Mathf.Min(halfX - Mathf.Abs(position.x), halfZ - Mathf.Abs(position.z));
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision == null)
            {
                return;
            }

            if (collision.collider != null &&
                (collision.collider.CompareTag("Chamber") || collision.collider.CompareTag("Obstacle")))
            {
                // Floor contacts are frequent; only count near-vertical wall-ish normals.
                foreach (ContactPoint contact in collision.contacts)
                {
                    if (Mathf.Abs(contact.normal.y) < 0.35f)
                    {
                        RegisterWallHit();
                        break;
                    }
                }
            }
        }

        private void FixedUpdate()
        {
            Refresh();
        }
    }
}
