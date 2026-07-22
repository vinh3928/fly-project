using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Root rigidbody, center of mass, and body segment references.
    /// </summary>
    public sealed class FlyBodyController : MonoBehaviour
    {
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

            for (int i = 0; i < legs.Length; i++)
            {
                legs[i]?.ApplyStandingPose();
            }
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
