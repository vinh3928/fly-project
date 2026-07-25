using FlySimulation.Fly.Sensors;
using UnityEngine;
#if UNITY_MLAGENTS
using Unity.MLAgents.Sensors;
#endif

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Ensures ML-Agents <see cref="CameraSensorComponent"/>s exist for left/right
    /// <see cref="FlyEyeCamera"/>s so Python live mind receives eye frames / wall_hazard.
    /// Must run before Agent sensor init (early Awake).
    /// </summary>
    [DefaultExecutionOrder(-200)]
    public sealed class FlyEyeMlAgentsBridge : MonoBehaviour
    {
        public const int EyeResolution = 64;

#if UNITY_MLAGENTS
        private void Awake()
        {
            EnsureCameraSensors();
        }

        public void EnsureCameraSensors()
        {
            VisionFrameProvider vision = GetComponent<SensorManager>() != null
                ? GetComponent<SensorManager>().Vision
                : null;
            if (vision == null)
            {
                vision = GetComponentInChildren<VisionFrameProvider>(true);
            }

            FlyEyeCamera left = vision != null ? vision.LeftEye : null;
            FlyEyeCamera right = vision != null ? vision.RightEye : null;
            if (left == null || right == null)
            {
                var eyes = GetComponentsInChildren<FlyEyeCamera>(true);
                foreach (var eye in eyes)
                {
                    if (eye.IsLeftEye)
                    {
                        left = left != null ? left : eye;
                    }
                    else
                    {
                        right = right != null ? right : eye;
                    }
                }

                // Fallback if both marked left (legacy scene): order by local X.
                if ((left == null || right == null) && eyes.Length >= 2)
                {
                    System.Array.Sort(
                        eyes,
                        (a, b) => a.transform.localPosition.x.CompareTo(b.transform.localPosition.x));
                    left = eyes[0];
                    right = eyes[eyes.Length - 1];
                }
            }

            if (left == null || right == null)
            {
                Debug.LogWarning("FlyEyeMlAgentsBridge: left/right FlyEyeCamera not found.");
                return;
            }

            // Prefer correct laterality flags when present.
            if (left.IsLeftEye == right.IsLeftEye)
            {
                // Leave spatial sort as-is; Capture still works for stereo bars.
            }

            BindOrCreate("LeftEye", left.GetComponent<Camera>(), grayscale: false);
            BindOrCreate("RightEye", right.GetComponent<Camera>(), grayscale: false);
        }

        private void BindOrCreate(string sensorName, Camera cam, bool grayscale)
        {
            if (cam == null)
            {
                return;
            }

            CameraSensorComponent[] existing = GetComponents<CameraSensorComponent>();
            CameraSensorComponent match = null;
            foreach (var c in existing)
            {
                if (c.SensorName == sensorName)
                {
                    match = c;
                    break;
                }
            }

            if (match == null)
            {
                match = gameObject.AddComponent<CameraSensorComponent>();
            }

            match.Camera = cam;
            match.SensorName = sensorName;
            match.Width = EyeResolution;
            match.Height = EyeResolution;
            match.Grayscale = grayscale;
            match.CompressionType = SensorCompressionType.PNG;
            match.ObservationStacks = 1;
            // Camera may be disabled for Game-view clarity; CameraSensor still calls Render().
            match.RuntimeCameraEnable = false;
        }
#else
        private void Awake()
        {
            // ML-Agents not present — no visual sensors to register.
        }
#endif
    }
}
