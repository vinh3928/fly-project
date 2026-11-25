using UnityEngine;

namespace FlySimulation.Fly.Motors
{
    /// <summary>
    /// Mouth: 7-d articulatory command → procedural buzz/formant AudioSource.
    /// jaw, lip_open, tongue_front, tongue_height, voicing, pitch, amplitude.
    /// Streaming clip so babble is audible to the fly (and the Game view listener).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FlyMouthActuator : MonoBehaviour
    {
        public const int ArticulatoryDim = 7;

        [SerializeField] private AudioSource source;
        [SerializeField] private float baseFrequency = 180f;

        private readonly float[] _ctrl = new float[ArticulatoryDim];
        private double _phase;
        private int _sampleRate = 48000;
        private readonly object _lock = new object();

        public float[] Control
        {
            get
            {
                lock (_lock)
                {
                    return (float[])_ctrl.Clone();
                }
            }
        }

        private void Awake()
        {
            lock (_lock)
            {
                for (int i = 0; i < ArticulatoryDim; i++)
                {
                    _ctrl[i] = i == 5 ? 0f : (i == 6 ? 0f : 0.3f);
                }
            }

            EnsureSource();
        }

        private void OnEnable()
        {
            EnsureSource();
            if (source != null && !source.isPlaying)
            {
                source.Play();
            }
        }

        private void EnsureSource()
        {
            if (source == null)
            {
                source = GetComponent<AudioSource>();
                if (source == null)
                {
                    source = gameObject.AddComponent<AudioSource>();
                }
            }

            _sampleRate = Mathf.Max(8000, AudioSettings.outputSampleRate);
            source.playOnAwake = true;
            source.loop = true;
            source.spatialBlend = 0.55f;
            source.minDistance = 0.02f;
            source.maxDistance = 0.8f;
            source.rolloffMode = AudioRolloffMode.Linear;
            source.dopplerLevel = 0f;
            source.volume = 0.85f;
            source.mute = false;
            source.priority = 32;

            if (source.clip == null || !source.clip.name.StartsWith("MouthBuzzStream"))
            {
                var clip = AudioClip.Create(
                    "MouthBuzzStream",
                    _sampleRate,
                    1,
                    _sampleRate,
                    true,
                    OnAudioRead);
                source.clip = clip;
            }

            if (!source.isPlaying)
            {
                source.Play();
            }
        }

        /// <summary>Set articulatory command (Python speech babble / executor).</summary>
        public void SetArticulatory(
            float jaw,
            float lipOpen,
            float tongueFront,
            float tongueHeight,
            float voicing,
            float pitch,
            float amplitude)
        {
            lock (_lock)
            {
                _ctrl[0] = Mathf.Clamp01(jaw);
                _ctrl[1] = Mathf.Clamp01(lipOpen);
                _ctrl[2] = Mathf.Clamp01(tongueFront);
                _ctrl[3] = Mathf.Clamp01(tongueHeight);
                _ctrl[4] = Mathf.Clamp01(voicing);
                _ctrl[5] = Mathf.Clamp(pitch, -1f, 1f);
                _ctrl[6] = Mathf.Clamp01(amplitude);
            }
        }

        public void SetArticulatory(float[] values)
        {
            if (values == null)
            {
                return;
            }

            SetArticulatory(
                values.Length > 0 ? values[0] : 0f,
                values.Length > 1 ? values[1] : 0f,
                values.Length > 2 ? values[2] : 0f,
                values.Length > 3 ? values[3] : 0f,
                values.Length > 4 ? values[4] : 0f,
                values.Length > 5 ? values[5] : 0f,
                values.Length > 6 ? values[6] : 0f);
        }

        public void Silence()
        {
            SetArticulatory(0.2f, 0.2f, 0.3f, 0.3f, 0f, 0f, 0f);
        }

        private void OnAudioRead(float[] data)
        {
            if (data == null)
            {
                return;
            }

            float jaw, lip, tf, th, voice, pitch, amp;
            lock (_lock)
            {
                jaw = _ctrl[0];
                lip = _ctrl[1];
                tf = _ctrl[2];
                th = _ctrl[3];
                voice = _ctrl[4];
                pitch = _ctrl[5];
                amp = _ctrl[6];
            }

            float level = amp * voice;
            if (level < 1e-4f)
            {
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = 0f;
                }

                return;
            }

            float f0 = baseFrequency * Mathf.Pow(2f, pitch);
            float f1 = f0 * (1.2f + 1.8f * jaw);
            float f2 = f0 * (2.5f + 3.5f * tf + 1.5f * th);
            double dt = 1.0 / _sampleRate;
            float lipGain = Mathf.Lerp(0.4f, 1f, lip);
            for (int i = 0; i < data.Length; i++)
            {
                _phase += 2.0 * Mathf.PI * f0 * dt;
                if (_phase > 2.0 * Mathf.PI)
                {
                    _phase -= 2.0 * Mathf.PI;
                }

                float buzz = Mathf.Sin((float)_phase);
                float p1 = Mathf.Sin((float)(_phase * (f1 / (double)f0)));
                float p2 = Mathf.Sin((float)(_phase * (f2 / (double)f0)));
                data[i] = level * lipGain * (0.55f * buzz + 0.3f * p1 + 0.15f * p2);
            }
        }
    }
}
