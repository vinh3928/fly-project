using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Tracks physics health for Phase 1 exit criteria (no NaNs / explosions).
    /// </summary>
    public sealed class PhysicsStabilityMonitor : MonoBehaviour
    {
        [SerializeField] private Rigidbody target;
        [SerializeField] private float maxLinearSpeed = 25f;
        [SerializeField] private float maxAngularSpeed = 50f;

        public bool IsStable { get; private set; } = true;
        public string FailureReason { get; private set; } = string.Empty;
        public float MaxObservedLinearSpeed { get; private set; }
        public float MaxObservedAngularSpeed { get; private set; }

        public void Bind(Rigidbody body)
        {
            target = body;
            ResetMonitor();
        }

        public void ResetMonitor()
        {
            IsStable = true;
            FailureReason = string.Empty;
            MaxObservedLinearSpeed = 0f;
            MaxObservedAngularSpeed = 0f;
        }

        public bool Evaluate()
        {
            if (!IsStable)
            {
                return false;
            }

            if (target == null)
            {
                return true;
            }

            Vector3 linear = target.linearVelocity;
            Vector3 angular = target.angularVelocity;
            Vector3 position = target.position;

            if (!IsFinite(linear) || !IsFinite(angular) || !IsFinite(position))
            {
                Fail("non_finite_physics_state");
                return false;
            }

            float linearSpeed = linear.magnitude;
            float angularSpeed = angular.magnitude;
            MaxObservedLinearSpeed = Mathf.Max(MaxObservedLinearSpeed, linearSpeed);
            MaxObservedAngularSpeed = Mathf.Max(MaxObservedAngularSpeed, angularSpeed);

            if (linearSpeed > maxLinearSpeed)
            {
                Fail("linear_speed_exceeded");
                return false;
            }

            if (angularSpeed > maxAngularSpeed)
            {
                Fail("angular_speed_exceeded");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Pure helper used by EditMode tests without a live Rigidbody.
        /// </summary>
        public static bool IsStateStable(
            Vector3 position,
            Vector3 linearVelocity,
            Vector3 angularVelocity,
            float maxLinearSpeed,
            float maxAngularSpeed,
            out string reason)
        {
            if (!IsFinite(position) || !IsFinite(linearVelocity) || !IsFinite(angularVelocity))
            {
                reason = "non_finite_physics_state";
                return false;
            }

            if (linearVelocity.magnitude > maxLinearSpeed)
            {
                reason = "linear_speed_exceeded";
                return false;
            }

            if (angularVelocity.magnitude > maxAngularSpeed)
            {
                reason = "angular_speed_exceeded";
                return false;
            }

            reason = string.Empty;
            return true;
        }

        private void Fail(string reason)
        {
            IsStable = false;
            FailureReason = reason;
        }

        private static bool IsFinite(Vector3 v)
        {
            return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}
