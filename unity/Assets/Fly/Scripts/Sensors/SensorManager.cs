using FlySimulation.Fly.Body;
using FlySimulation.Fly.Sensors;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Batches joint, contact, chemistry, vision, and internal-state observations.
    /// Avoids per-sensor Update allocations by writing into fixed buffers.
    /// </summary>
    public sealed class SensorManager : MonoBehaviour
    {
        public const int JointCount = 18;
        public const int FootCount = 6;
        public const int ChemicalChannels = 8;
        public const int InternalStateCount = 8;
        public const int BodyStateCount = 13;

        /// <summary>Body-state velocity channels are measured velocity / this (m/s); Python BODY_LINEAR_VELOCITY_SCALE.</summary>
        public const float LinearVelocityScale = 2f;

        /// <summary>
        /// Fixed vector observation length written by <see cref="CollectVectorObservation"/>.
        /// Must match ML-Agents BehaviorParameters.VectorObservationSize.
        /// </summary>
        public const int ExpectedVectorSize =
            JointCount * 4 +
            FootCount * 5 +
            FootCount * ChemicalChannels +
            ChemicalChannels * 2 +
            (2 + ChemicalChannels) +
            InternalStateCount +
            BodyStateCount;

        [SerializeField] private FlyBodyController body;
        [SerializeField] private FlyHomeostasis homeostasis;
        [SerializeField] private FootContactSensor[] feet = new FootContactSensor[6];
        [SerializeField] private AntennaOdorSensor leftAntenna;
        [SerializeField] private AntennaOdorSensor rightAntenna;
        [SerializeField] private ProboscisSensor proboscis;
        [SerializeField] private VisionFrameProvider vision;
        [SerializeField] private LayerMask groundMask = ~0;

        private readonly JointObservation[] _jointObservations = new JointObservation[JointCount];
        private readonly float[] _vectorObservation = new float[512];
        private int _vectorLength;

        public float[] VectorObservation => _vectorObservation;
        public int VectorObservationLength => _vectorLength;
        public float[] LeftVision => vision != null ? vision.LeftFrame : System.Array.Empty<float>();
        public float[] RightVision => vision != null ? vision.RightFrame : System.Array.Empty<float>();
        public JointObservation[] Joints => _jointObservations;
        public FootContactSensor[] Feet => feet;
        public ProboscisSensor Proboscis => proboscis;
        public AntennaOdorSensor LeftAntenna => leftAntenna;
        public AntennaOdorSensor RightAntenna => rightAntenna;
        public VisionFrameProvider Vision => vision;
        public FlyBodyController Body => body;
        public FlyHomeostasis Homeostasis => homeostasis;

        public void Bind(
            FlyBodyController bodyController,
            FlyHomeostasis homeo,
            FootContactSensor[] footSensors,
            AntennaOdorSensor left,
            AntennaOdorSensor right,
            ProboscisSensor proboscisSensor,
            VisionFrameProvider visionProvider)
        {
            body = bodyController;
            homeostasis = homeo;
            feet = footSensors;
            leftAntenna = left;
            rightAntenna = right;
            proboscis = proboscisSensor;
            vision = visionProvider;
        }

        public int CollectVectorObservation(float deltaTime)
        {
            int offset = 0;
            offset = WriteJoints(offset, deltaTime);
            offset = WriteContacts(offset);
            offset = WriteTaste(offset);
            offset = WriteAntennae(offset);
            offset = WriteProboscis(offset);
            offset = WriteInternal(offset);
            offset = WriteBodyState(offset);
            _vectorLength = offset;
            return offset;
        }

        private int WriteJoints(int offset, float deltaTime)
        {
            var legs = body != null ? body.Legs : null;
            for (int i = 0; i < FootCount; i++)
            {
                int jointOffset = i * 3;
                if (legs != null && i < legs.Length && legs[i] != null)
                {
                    legs[i].CollectObservations(_jointObservations, jointOffset, deltaTime);
                }

                for (int j = 0; j < 3; j++)
                {
                    JointObservation obs = _jointObservations[jointOffset + j];
                    _vectorObservation[offset++] = obs.Position;
                    _vectorObservation[offset++] = obs.Velocity;
                    _vectorObservation[offset++] = obs.Target;
                    _vectorObservation[offset++] = obs.AppliedForce;
                }
            }

            return offset;
        }

        private int WriteContacts(int offset)
        {
            for (int i = 0; i < FootCount; i++)
            {
                if (feet != null && i < feet.Length && feet[i] != null)
                {
                    feet[i].CopyContactObservation(_vectorObservation, offset);
                }
                else
                {
                    for (int k = 0; k < 5; k++)
                    {
                        _vectorObservation[offset + k] = 0f;
                    }
                }

                offset += 5;
            }

            return offset;
        }

        private int WriteTaste(int offset)
        {
            for (int i = 0; i < FootCount; i++)
            {
                if (feet != null && i < feet.Length && feet[i] != null)
                {
                    feet[i].CopyTasteObservation(_vectorObservation, offset);
                }
                else
                {
                    for (int k = 0; k < ChemicalChannels; k++)
                    {
                        _vectorObservation[offset + k] = 0f;
                    }
                }

                offset += ChemicalChannels;
            }

            return offset;
        }

        private int WriteAntennae(int offset)
        {
            if (leftAntenna != null)
            {
                leftAntenna.CopyObservation(_vectorObservation, offset);
            }

            offset += ChemicalChannels;
            if (rightAntenna != null)
            {
                rightAntenna.CopyObservation(_vectorObservation, offset);
            }

            offset += ChemicalChannels;
            return offset;
        }

        private int WriteProboscis(int offset)
        {
            if (proboscis != null)
            {
                proboscis.CopyObservation(_vectorObservation, offset);
            }

            return offset + 2 + ChemicalChannels;
        }

        private int WriteInternal(int offset)
        {
            FlyHomeostasis homeo = homeostasis != null ? homeostasis : GetComponent<FlyHomeostasis>();
            if (homeo != null)
            {
                homeo.CopyNormalized(_vectorObservation, offset);
            }
            else
            {
                for (int i = 0; i < InternalStateCount; i++)
                {
                    _vectorObservation[offset + i] = 0f;
                }
            }

            return offset + InternalStateCount;
        }

        private int WriteBodyState(int offset)
        {
            Vector3 linear = body != null ? body.MeasuredVelocity : Vector3.zero;
            Vector3 angular = body != null ? body.AngularVelocity : Vector3.zero;
            Vector3 gravityLocal = body != null ? body.transform.InverseTransformDirection(Physics.gravity).normalized : Vector3.down;
            Vector3 up = body != null ? body.transform.up : Vector3.up;
            float height = body != null ? body.HeightAboveGround(groundMask) : 0f;

            _vectorObservation[offset++] = Mathf.Clamp(linear.x / LinearVelocityScale, -1f, 1f);
            _vectorObservation[offset++] = Mathf.Clamp(linear.y / LinearVelocityScale, -1f, 1f);
            _vectorObservation[offset++] = Mathf.Clamp(linear.z / LinearVelocityScale, -1f, 1f);
            _vectorObservation[offset++] = Mathf.Clamp(angular.x / 10f, -1f, 1f);
            _vectorObservation[offset++] = Mathf.Clamp(angular.y / 10f, -1f, 1f);
            _vectorObservation[offset++] = Mathf.Clamp(angular.z / 10f, -1f, 1f);
            _vectorObservation[offset++] = gravityLocal.x;
            _vectorObservation[offset++] = gravityLocal.y;
            _vectorObservation[offset++] = gravityLocal.z;
            _vectorObservation[offset++] = up.x;
            _vectorObservation[offset++] = up.y;
            _vectorObservation[offset++] = up.z;
            _vectorObservation[offset++] = Mathf.Clamp01(height / 0.2f);
            return offset;
        }
    }
}
