using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Keeps the fly upright and planted on the floor for Phase 2 hybrid locomotion.
    /// Visual legs animate; root rigidbody carries the body (no foot ground-reaction yet).
    /// </summary>
    public sealed class BalanceController : MonoBehaviour
    {
        [SerializeField] private Rigidbody target;
        [SerializeField] private float uprightStrength = 80f;
        [SerializeField] private float uprightDamping = 12f;
        [SerializeField] private float targetHeight = 0.02f;
        [SerializeField] private bool enabledBalance = true;
        [SerializeField] private bool hardLockHeight = true;

        public bool EnabledBalance
        {
            get => enabledBalance;
            set => enabledBalance = value;
        }

        public float TargetHeight
        {
            get => targetHeight;
            set => targetHeight = Mathf.Max(0.001f, value);
        }

        public void Bind(Rigidbody body)
        {
            target = body;
        }

        public void FixedBalanceUpdate()
        {
            if (!enabledBalance || target == null)
            {
                return;
            }

            Vector3 torque = ComputeUprightTorque(
                target.rotation,
                target.angularVelocity,
                uprightStrength,
                uprightDamping);
            target.AddTorque(torque, ForceMode.Acceleration);

            if (hardLockHeight)
            {
                ClampToTargetHeight();
            }
            else
            {
                Vector3 force = ComputeHeightForce(
                    target.position.y,
                    target.linearVelocity.y,
                    targetHeight,
                    strength: 120f,
                    damping: 20f);
                target.AddForce(force, ForceMode.Acceleration);
            }
        }

        private void ClampToTargetHeight()
        {
            Vector3 pos = target.position;
            pos.y = targetHeight;
            target.position = pos;

            Vector3 vel = target.linearVelocity;
            vel.y = 0f;
            target.linearVelocity = vel;
        }

        /// <summary>Pure upright PD used by tests and the runtime controller.</summary>
        public static Vector3 ComputeUprightTorque(
            Quaternion rotation,
            Vector3 angularVelocity,
            float strength,
            float damping)
        {
            Vector3 up = rotation * Vector3.up;
            Vector3 axis = Vector3.Cross(up, Vector3.up);
            if (axis.sqrMagnitude < 1e-8f)
            {
                axis = Vector3.Cross(up, Vector3.forward);
                if (axis.sqrMagnitude < 1e-8f)
                {
                    axis = Vector3.right;
                }
            }

            axis.Normalize();
            float angle = Vector3.Angle(up, Vector3.up) * Mathf.Deg2Rad;
            return axis * (angle * strength) - angularVelocity * damping;
        }

        /// <summary>Pure vertical spring-damper used by tests and the runtime controller.</summary>
        public static Vector3 ComputeHeightForce(
            float positionY,
            float velocityY,
            float targetY,
            float strength,
            float damping)
        {
            float error = targetY - positionY;
            return Vector3.up * (error * strength - velocityY * damping);
        }
    }
}
