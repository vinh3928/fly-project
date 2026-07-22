using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Three-joint articulated leg: coxa, femur, tibia.
    /// </summary>
    public sealed class LegController : MonoBehaviour
    {
        public enum LegId
        {
            LeftFront,
            LeftMiddle,
            LeftHind,
            RightFront,
            RightMiddle,
            RightHind
        }

        [SerializeField] private LegId legId;
        [SerializeField] private LegJoint coxa;
        [SerializeField] private LegJoint femur;
        [SerializeField] private LegJoint tibia;
        [SerializeField] private Transform foot;

        public LegId Id => legId;
        public LegJoint Coxa => coxa;
        public LegJoint Femur => femur;
        public LegJoint Tibia => tibia;
        public Transform Foot => foot;

        public void SetJointTargets(float coxaDeg, float femurDeg, float tibiaDeg)
        {
            coxa?.SetTargetDegrees(coxaDeg);
            femur?.SetTargetDegrees(femurDeg);
            tibia?.SetTargetDegrees(tibiaDeg);
        }

        public void SetNormalizedTargets(float coxaN, float femurN, float tibiaN)
        {
            coxa?.SetNormalizedTarget(coxaN);
            femur?.SetNormalizedTarget(femurN);
            tibia?.SetNormalizedTarget(tibiaN);
        }

        public void CollectObservations(JointObservation[] buffer, int offset, float deltaTime)
        {
            if (buffer == null || buffer.Length < offset + 3)
            {
                return;
            }

            buffer[offset] = coxa != null ? coxa.Observe(deltaTime) : default;
            buffer[offset + 1] = femur != null ? femur.Observe(deltaTime) : default;
            buffer[offset + 2] = tibia != null ? tibia.Observe(deltaTime) : default;
        }

        public void ApplyStandingPose()
        {
            // Plantigrade stance: feet reach down toward the floor under the body.
            SetJointTargets(0f, -40f, 70f);
        }
    }
}
