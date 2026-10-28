using FlySimulation.Fly.Body;
using UnityEngine;

namespace FlySimulation.Fly.Motors
{
    /// <summary>
    /// Hierarchical motor interface: high-level gait commands or direct joint targets,
    /// plus root-body XZ velocity / yaw for Phase 2 walking / turning.
    /// </summary>
    public sealed class FlyMotorController : MonoBehaviour
    {
        public enum ControlMode
        {
            HighLevelGait,
            DirectJoints
        }

        [SerializeField] private FlyBodyController body;
        [SerializeField] private ControlMode mode = ControlMode.HighLevelGait;
        [SerializeField] private ProboscisActuator proboscis;
        [SerializeField] private float stepFrequency = 2f;
        [SerializeField] private float stepAmplitude = 25f;
        [SerializeField] private float turnBias;
        [SerializeField] private float walkForward;
        [SerializeField] private float bodyHeightCommand = 0.5f;
        [SerializeField] private float walkSpeed = 0.18f;
        [SerializeField] private float turnSpeed = 4.0f;

        private float _phase;
        private readonly float[] _jointTargets = new float[18];
        private float _proboscisExtension;
        private float _lastEffort;

        public float LastMotorEffort => _lastEffort;
        public ControlMode Mode => mode;
        public float WalkForward => walkForward;
        public float TurnBias => turnBias;
        public float GaitPhase => _phase;

        public void Bind(FlyBodyController bodyController, ProboscisActuator proboscisActuator)
        {
            body = bodyController;
            proboscis = proboscisActuator;
        }

        public void SetHighLevelCommands(float forward, float turn, float height, float frequency)
        {
            mode = ControlMode.HighLevelGait;
            walkForward = Mathf.Clamp(forward, -1f, 1f);
            turnBias = Mathf.Clamp(turn, -1f, 1f);
            bodyHeightCommand = Mathf.Clamp01(height);
            stepFrequency = Mathf.Lerp(0.5f, 4f, Mathf.Clamp01(frequency));
        }

        public void SetDirectJointTargets(float[] targets18)
        {
            mode = ControlMode.DirectJoints;
            int count = Mathf.Min(18, targets18.Length);
            for (int i = 0; i < count; i++)
            {
                _jointTargets[i] = Mathf.Clamp(targets18[i], -1f, 1f);
            }
        }

        public void SetProboscisExtension(float normalized)
        {
            _proboscisExtension = Mathf.Clamp01(normalized);
            if (proboscis != null)
            {
                proboscis.SetExtension(_proboscisExtension);
            }
        }

        private void FixedUpdate()
        {
            Tick(Time.fixedDeltaTime);
        }

        /// <summary>Advance gait + locomotion (also callable from EditMode Physics.Simulate loops).</summary>
        public void Tick(float dt)
        {
            if (body == null)
            {
                return;
            }

            if (mode == ControlMode.HighLevelGait)
            {
                UpdateGait(dt);
                ApplyLocomotion(dt);
            }
            else
            {
                ApplyDirectTargets();
            }

            _lastEffort = Mathf.Clamp01(Mathf.Abs(walkForward) + Mathf.Abs(turnBias) + _proboscisExtension * 0.1f);
        }

        private void UpdateGait(float dt)
        {
            _phase = TripodGait.AdvancePhase(_phase, dt, stepFrequency, walkForward);
            var legs = body.Legs;
            if (legs == null)
            {
                return;
            }

            for (int i = 0; i < legs.Length; i++)
            {
                if (legs[i] == null)
                {
                    continue;
                }

                TripodGait.ComputeJointTargets(
                    i,
                    _phase,
                    walkForward,
                    turnBias,
                    bodyHeightCommand,
                    stepAmplitude,
                    out float coxa,
                    out float femur,
                    out float tibia);
                legs[i].SetJointTargets(coxa, femur, tibia);
            }
        }

        private void ApplyLocomotion(float dt)
        {
            if (!body.LocomotionEnabled || body.RootBody == null)
            {
                return;
            }

            Rigidbody rb = body.RootBody;
            rb.WakeUp();

            Vector3 forward = body.transform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 1e-6f)
            {
                forward = Vector3.forward;
            }
            else
            {
                forward.Normalize();
            }

            // Translate / yaw the root so directional motion matches the animated gait.
            Vector3 delta = forward * (walkForward * walkSpeed * dt);
            float standY = body.Balance != null ? body.Balance.TargetHeight : rb.position.y;
            Vector3 next = rb.position + delta;
            next.y = standY;
            rb.MovePosition(next);

            if (Mathf.Abs(turnBias) > 1e-4f)
            {
                // turnBias in [-1,1]; turnSpeed is rad/s at full command.
                float yawDegrees = turnBias * turnSpeed * dt * Mathf.Rad2Deg;
                rb.MoveRotation(rb.rotation * Quaternion.Euler(0f, yawDegrees, 0f));
            }

            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        private void ApplyDirectTargets()
        {
            var legs = body.Legs;
            if (legs == null)
            {
                return;
            }

            for (int i = 0; i < legs.Length && i < 6; i++)
            {
                int o = i * 3;
                legs[i]?.SetNormalizedTargets(_jointTargets[o], _jointTargets[o + 1], _jointTargets[o + 2]);
            }
        }

        public void ApplyStanding()
        {
            walkForward = 0f;
            turnBias = 0f;
            _phase = 0f;
            var legs = body != null ? body.Legs : null;
            if (legs == null)
            {
                return;
            }

            for (int i = 0; i < legs.Length; i++)
            {
                legs[i]?.ApplyStandingPose();
            }
        }
    }
}
