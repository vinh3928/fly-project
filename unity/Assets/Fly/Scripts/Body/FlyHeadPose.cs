using FlySimulation.Fly.Sensors;
using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Lowers and pitches the head down while probing (proboscis extended), bringing the
    /// antenna tips toward the food. The proboscis is counter-rotated so it stays vertical.
    /// </summary>
    public sealed class FlyHeadPose : MonoBehaviour
    {
        public const float MaxPitchDegrees = 45f;
        public const float MaxDrop = 0.01f;
        /// <summary>Max left/right look yaw while scanning for landmarks (degrees).</summary>
        public const float MaxLookYawDegrees = 180f;

        /// <summary>Proboscis extension at which the head starts lowering (probe mode extends to 1).</summary>
        public const float ProbeExtensionStart = 0.5f;

        [SerializeField] private Transform mount;
        [SerializeField] private Transform head;
        [SerializeField] private ProboscisSensor proboscis;
        [SerializeField] private Vector3 mountRest;
        [SerializeField] private Vector3 headRest;
        [Tooltip("Lowering speed (fraction of full pose per second).")]
        [SerializeField] private float response = 4f;

        public float Lower { get; private set; }
        public float LookYaw { get; private set; }

        public void Bind(Transform headMount, Transform headVisual, ProboscisSensor proboscisSensor)
        {
            mount = headMount;
            head = headVisual;
            proboscis = proboscisSensor;
            mountRest = mount != null ? mount.localPosition : Vector3.zero;
            headRest = head != null ? head.localPosition : Vector3.zero;
            Apply(0f);
        }

        /// <summary>Yaw the head/eyes left-right (landmark scan); clamped to ±MaxLookYawDegrees.</summary>
        public void SetLookYaw(float degrees)
        {
            LookYaw = Mathf.Clamp(degrees, -MaxLookYawDegrees, MaxLookYawDegrees);
            Apply(Lower);
        }

        public static float TargetLower(float proboscisExtension)
        {
            return Mathf.Clamp01(
                (proboscisExtension - ProbeExtensionStart) / (1f - ProbeExtensionStart));
        }

        public static float PitchDegrees(float lower) => MaxPitchDegrees * Mathf.Clamp01(lower);

        public static Vector3 DropOffset(float lower) => Vector3.down * (MaxDrop * Mathf.Clamp01(lower));

        /// <summary>
        /// How much lower (m) a point at <paramref name="localOnMount"/> sits when the head is
        /// fully lowered (drop plus pitch) than at rest.
        /// </summary>
        public static float FullLowerExtraDrop(Vector3 localOnMount)
        {
            Vector3 pitched = Quaternion.Euler(MaxPitchDegrees, 0f, 0f) * localOnMount;
            return MaxDrop + (localOnMount.y - pitched.y);
        }

        public void Apply(float lower)
        {
            Lower = Mathf.Clamp01(lower);
            float pitch = PitchDegrees(Lower);
            Vector3 drop = DropOffset(Lower);
            if (mount != null)
            {
                mount.localPosition = mountRest + drop;
                mount.localRotation = Quaternion.Euler(pitch, LookYaw, 0f);
            }

            if (head != null)
            {
                head.localPosition = headRest + drop;
            }

            if (proboscis != null)
            {
                proboscis.transform.localRotation =
                    Quaternion.Euler(ProboscisSensor.BodyPitchDegrees - pitch, 0f, 0f);
            }
        }

        private void FixedUpdate()
        {
            float target = proboscis != null ? TargetLower(proboscis.ExtensionNormalized) : 0f;
            Apply(Mathf.MoveTowards(Lower, target, response * Time.fixedDeltaTime));
        }
    }
}
