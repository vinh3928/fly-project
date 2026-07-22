using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Centralized vision frame provider with rate limiting.
    /// </summary>
    public sealed class VisionFrameProvider : MonoBehaviour
    {
        [SerializeField] private FlyEyeCamera leftEye;
        [SerializeField] private FlyEyeCamera rightEye;
        [SerializeField] private float visionHz = 30f;

        private float _accumulator;
        private float[] _leftFrame = System.Array.Empty<float>();
        private float[] _rightFrame = System.Array.Empty<float>();

        public float[] LeftFrame => _leftFrame;
        public float[] RightFrame => _rightFrame;
        public FlyEyeCamera LeftEye => leftEye;
        public FlyEyeCamera RightEye => rightEye;

        public void Bind(FlyEyeCamera left, FlyEyeCamera right, float hz)
        {
            leftEye = left;
            rightEye = right;
            visionHz = Mathf.Max(1f, hz);
        }

        private void Update()
        {
            _accumulator += Time.deltaTime;
            float interval = 1f / visionHz;
            if (_accumulator < interval)
            {
                return;
            }

            _accumulator %= interval;
            CaptureNow();
        }

        public void CaptureNow()
        {
            if (leftEye != null)
            {
                _leftFrame = leftEye.CaptureFrame();
            }

            if (rightEye != null)
            {
                _rightFrame = rightEye.CaptureFrame();
            }
        }
    }
}
