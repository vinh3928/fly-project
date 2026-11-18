using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Root rigidbody, center of mass, and body segment references.
    /// </summary>
    public sealed class FlyBodyController : MonoBehaviour
    {
        /// <summary>Thorax ground-contact sphere radius (local, at the root origin).</summary>
        public const float GroundRadius = 0.005f;

        /// <summary>Head segment center (local); the eyes sit at its center.</summary>
        public static readonly Vector3 HeadCenter = new Vector3(0f, 0f, 0.01f);

        /// <summary>Head sphere radius; must exceed the eye near-plane corner distance.</summary>
        public const float HeadRadius = 0.0045f;

        /// <summary>Abdomen collision sphere (local); radius = ground radius so it clears the floor.</summary>
        public static readonly Vector3 AbdomenCenter = new Vector3(0f, 0f, -0.012f);

        [SerializeField] private Rigidbody rootBody;
        [SerializeField] private Transform head;
        [SerializeField] private Transform thorax;
        [SerializeField] private Transform abdomen;
        [SerializeField] private LegController[] legs = new LegController[6];
        [SerializeField] private BalanceController balance;
        [SerializeField] private Vector3 centerOfMassOffset = Vector3.zero;
        [SerializeField] private float fallTiltDegrees = 60f;

        public Rigidbody RootBody => rootBody;
        public LegController[] Legs => legs;
        public Transform Head => head;
        public BalanceController Balance => balance;
        public bool HasFallen { get; private set; }
        public bool LocomotionEnabled { get; private set; }

        public Vector3 LinearVelocity => rootBody != null ? rootBody.linearVelocity : Vector3.zero;
        public Vector3 AngularVelocity => rootBody != null ? rootBody.angularVelocity : Vector3.zero;

        /// <summary>
        /// Velocity from position change between physics steps. Locomotion moves the root with
        /// MovePosition and zeroes <see cref="LinearVelocity"/>, so this is the one that drops to
        /// zero when the fly is jammed against a wall.
        /// </summary>
        public Vector3 MeasuredVelocity { get; private set; }

        private Vector3 _lastSampledPosition;
        private bool _hasSampledPosition;

        public float TiltDegrees => Vector3.Angle(transform.up, Vector3.up);

        public void BindBalance(BalanceController balanceController)
        {
            balance = balanceController;
            if (balance != null && rootBody != null)
            {
                balance.Bind(rootBody);
            }
        }

        public void Configure(ExperimentConfiguration config)
        {
            transform.localScale = Vector3.one * config.BodyScale;
            LocomotionEnabled = config.LocomotionEnabled;
            EnsureSolidShell();
            if (rootBody != null)
            {
                rootBody.centerOfMass = centerOfMassOffset;
                rootBody.interpolation = RigidbodyInterpolation.Interpolate;
                rootBody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

                if (config.LocomotionEnabled)
                {
                    // Phase 2 hybrid: yaw free, pitch/roll locked so the body cannot tip through the floor.
                    rootBody.linearDamping = 0.2f;
                    rootBody.angularDamping = 4f;
                    rootBody.constraints =
                        RigidbodyConstraints.FreezeRotationX |
                        RigidbodyConstraints.FreezeRotationZ;
                    rootBody.useGravity = false; // height is hard-locked by BalanceController
                }
                else
                {
                    // Phase 1: passive body — freeze rotation under gravity alone.
                    rootBody.linearDamping = 1f;
                    rootBody.angularDamping = 5f;
                    rootBody.constraints = RigidbodyConstraints.FreezeRotation;
                    rootBody.useGravity = true;
                }
            }

            if (balance != null)
            {
                balance.Bind(rootBody);
                balance.EnabledBalance = config.LocomotionEnabled;
                balance.TargetHeight = Mathf.Max(0.005f, config.FlySpawnPosition.y);
            }
        }

        public void ResetPose(Vector3 worldPosition, Quaternion worldRotation)
        {
            HasFallen = false;
            if (rootBody != null)
            {
                rootBody.position = worldPosition;
                rootBody.rotation = worldRotation;
                rootBody.linearVelocity = Vector3.zero;
                rootBody.angularVelocity = Vector3.zero;
                rootBody.Sleep();
            }
            else
            {
                transform.SetPositionAndRotation(worldPosition, worldRotation);
            }

            Physics.SyncTransforms();
            _hasSampledPosition = false;
            MeasuredVelocity = Vector3.zero;

            for (int i = 0; i < legs.Length; i++)
            {
                legs[i]?.ApplyStandingPose();
            }
        }

        /// <summary>
        /// Solid head and abdomen spheres on the root, alongside the thorax ground sphere, so walls
        /// stop the whole body (and the eyes inside the head), not just the thorax. Idempotent.
        /// </summary>
        public void EnsureSolidShell()
        {
            EnsureSphere(HeadCenter, HeadRadius);
            EnsureSphere(AbdomenCenter, GroundRadius);
        }

        private void EnsureSphere(Vector3 center, float radius)
        {
            foreach (var existing in GetComponents<SphereCollider>())
            {
                if (!existing.isTrigger && (existing.center - center).sqrMagnitude < 1e-12f)
                {
                    existing.radius = radius;
                    return;
                }
            }

            var sphere = gameObject.AddComponent<SphereCollider>();
            sphere.center = center;
            sphere.radius = radius;
        }

        private void FixedUpdate()
        {
            SampleVelocity(Time.fixedDeltaTime);
        }

        /// <summary>Update <see cref="MeasuredVelocity"/>; call once per physics step.</summary>
        public void SampleVelocity(float dt)
        {
            Vector3 position = rootBody != null ? rootBody.position : transform.position;
            MeasuredVelocity = _hasSampledPosition && dt > 0f
                ? (position - _lastSampledPosition) / dt
                : Vector3.zero;
            _lastSampledPosition = position;
            _hasSampledPosition = true;
        }

        public void FixedSensorUpdate()
        {
            if (LocomotionEnabled)
            {
                balance?.FixedBalanceUpdate();
            }

            if (rootBody == null)
            {
                return;
            }

            if (TiltDegrees > fallTiltDegrees)
            {
                HasFallen = true;
            }
        }

        public float HeightAboveGround(LayerMask groundMask)
        {
            Vector3 origin = rootBody != null ? rootBody.position : transform.position;
            if (Physics.Raycast(origin, Vector3.down, out RaycastHit hit, 5f, groundMask))
            {
                return hit.distance;
            }

            return origin.y;
        }

        public float BodyInstability()
        {
            float tilt = TiltDegrees / 180f;
            float spin = AngularVelocity.magnitude / 10f;
            return Mathf.Clamp01(tilt + spin);
        }

        /// <summary>Applies an instantaneous world impulse (Phase 2 disturbance recovery).</summary>
        public void ApplyDisturbanceImpulse(Vector3 worldImpulse)
        {
            if (rootBody == null)
            {
                return;
            }

            rootBody.AddForce(worldImpulse, ForceMode.Impulse);
        }
    }
}
