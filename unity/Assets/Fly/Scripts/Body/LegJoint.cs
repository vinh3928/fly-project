using System;
using UnityEngine;

namespace FlySimulation.Fly.Body
{
    [Serializable]
    public struct JointObservation
    {
        public float Position;
        public float Velocity;
        public float Target;
        public float AppliedForce;
        public float DistanceFromLimits;
    }

    [Serializable]
    public struct JointCommand
    {
        public float TargetPosition;
        public float Stiffness;
        public float Damping;
    }

    /// <summary>
    /// Single controlled leg joint (coxa / femur / tibia).
    /// Works with ArticulationBody when present, otherwise tracks local euler for prototyping.
    /// </summary>
    public sealed class LegJoint : MonoBehaviour
    {
        public enum JointAxis
        {
            X,
            Y,
            Z
        }

        [SerializeField] private JointAxis axis = JointAxis.X;
        [SerializeField] private float lowerLimit = -60f;
        [SerializeField] private float upperLimit = 60f;
        [SerializeField] private float stiffness = 200f;
        [SerializeField] private float damping = 20f;

        private ArticulationBody _articulation;
        private float _targetDegrees;
        private float _appliedForce;
        private float _previousPosition;

        public float LowerLimit => lowerLimit;
        public float UpperLimit => upperLimit;
        public float TargetDegrees => _targetDegrees;

        private void Awake()
        {
            _articulation = GetComponent<ArticulationBody>();
            _targetDegrees = GetCurrentPositionDegrees();
            _previousPosition = _targetDegrees;
        }

        public void SetTargetDegrees(float degrees)
        {
            _targetDegrees = Mathf.Clamp(degrees, lowerLimit, upperLimit);
            ApplyDrive();
        }

        public void SetNormalizedTarget(float normalized)
        {
            float t = Mathf.Clamp(normalized, -1f, 1f);
            float mid = 0.5f * (lowerLimit + upperLimit);
            float half = 0.5f * (upperLimit - lowerLimit);
            SetTargetDegrees(mid + t * half);
        }

        public JointObservation Observe(float deltaTime)
        {
            float position = GetCurrentPositionDegrees();
            float velocity = deltaTime > 1e-6f ? (position - _previousPosition) / deltaTime : 0f;
            _previousPosition = position;

            float limitSpan = Mathf.Max(1e-3f, upperLimit - lowerLimit);
            float distanceFromLimits = Mathf.Min(position - lowerLimit, upperLimit - position) / limitSpan;

            return new JointObservation
            {
                Position = NormalizePosition(position),
                Velocity = Mathf.Clamp(velocity / 180f, -1f, 1f),
                Target = NormalizePosition(_targetDegrees),
                AppliedForce = Mathf.Clamp(_appliedForce / 10f, -1f, 1f),
                DistanceFromLimits = Mathf.Clamp01(distanceFromLimits)
            };
        }

        private void ApplyDrive()
        {
            if (_articulation == null)
            {
                Vector3 euler = transform.localEulerAngles;
                float value = _targetDegrees;
                switch (axis)
                {
                    case JointAxis.X: euler.x = value; break;
                    case JointAxis.Y: euler.y = value; break;
                    case JointAxis.Z: euler.z = value; break;
                }

                transform.localEulerAngles = euler;
                _appliedForce = 0f;
                return;
            }

            var drive = _articulation.xDrive;
            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = 1e5f;
            drive.target = _targetDegrees;
            _articulation.xDrive = drive;
            _appliedForce = stiffness * (_targetDegrees - GetCurrentPositionDegrees()) * 0.01f;
        }

        private float GetCurrentPositionDegrees()
        {
            if (_articulation != null && _articulation.jointPosition.dofCount > 0)
            {
                return _articulation.jointPosition[0] * Mathf.Rad2Deg;
            }

            Vector3 euler = transform.localEulerAngles;
            float raw = axis switch
            {
                JointAxis.Y => euler.y,
                JointAxis.Z => euler.z,
                _ => euler.x
            };
            return NormalizeAngle(raw);
        }

        private float NormalizePosition(float degrees)
        {
            float mid = 0.5f * (lowerLimit + upperLimit);
            float half = 0.5f * (upperLimit - lowerLimit);
            return half > 1e-6f ? Mathf.Clamp((degrees - mid) / half, -1f, 1f) : 0f;
        }

        public static float NormalizeAngle(float angle)
        {
            while (angle > 180f) angle -= 360f;
            while (angle < -180f) angle += 360f;
            return angle;
        }

        public static float DistanceFromLimitsNormalized(float position, float lower, float upper)
        {
            float span = Mathf.Max(1e-3f, upper - lower);
            return Mathf.Clamp01(Mathf.Min(position - lower, upper - position) / span);
        }
    }
}
