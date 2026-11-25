using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Sensors;
using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// Binds ears + mouth to the speech SideChannel and pushes audio each FixedUpdate.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SpeechAudioBridge : MonoBehaviour
    {
        [SerializeField] private FlyAudioCapture ears;
        [SerializeField] private FlyMouthActuator mouth;

#if UNITY_MLAGENTS
        private SpeechDebugChannel _channel;
#endif

        private void Awake()
        {
            if (ears == null)
            {
                ears = GetComponentInChildren<FlyAudioCapture>(true);
            }

            if (mouth == null)
            {
                mouth = GetComponentInChildren<FlyMouthActuator>(true);
            }
        }

        private void OnEnable()
        {
#if UNITY_MLAGENTS
            if (ears != null && mouth != null)
            {
                _channel = SpeechDebugChannel.Register(ears, mouth);
            }
#endif
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (_channel != null)
            {
                _channel.Unregister();
                _channel = null;
            }
#endif
        }

        private void FixedUpdate()
        {
#if UNITY_MLAGENTS
            _channel?.SendAudioFeatures();
#endif
        }

        /// <summary>EditMode: apply mouth without SideChannel.</summary>
        public void ApplyMouthForTests(float[] ctrl)
        {
            if (mouth == null)
            {
                mouth = GetComponentInChildren<FlyMouthActuator>(true);
            }

            if (mouth != null)
            {
                mouth.SetArticulatory(ctrl);
            }
        }

        public FlyAudioCapture Ears => ears;
        public FlyMouthActuator Mouth => mouth;
    }
}
