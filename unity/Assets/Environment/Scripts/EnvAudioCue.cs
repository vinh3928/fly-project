using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Chamber/outside cue tone the fly can hear (no speech model — simple oscillator).
    /// Uses a streaming <see cref="AudioClip"/> so playback is reliably audible.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class EnvAudioCue : MonoBehaviour
    {
        [SerializeField] private AudioSource source;
        [SerializeField] private float frequency = 440f;
        [SerializeField] private float amplitude = 0.45f;
        [SerializeField] private float burstSeconds = 0.55f;
        [SerializeField] private float periodSeconds = 2f;

        private double _phase;
        private float _timer;
        private bool _bursting;
        private int _sampleRate = 48000;
        private readonly object _lock = new object();

        public void Configure(float freqHz, float amp = 0.45f, float period = 2f)
        {
            frequency = freqHz;
            amplitude = amp;
            periodSeconds = period;
            if (source != null)
            {
                source.volume = 1f;
            }
        }

        private void Awake()
        {
            EnsureSource();
            _timer = Random.Range(0f, Mathf.Max(0.1f, periodSeconds));
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
            // Mostly spatial for the fly listener, still audible from overview distance.
            source.spatialBlend = 0.7f;
            source.minDistance = 0.03f;
            source.maxDistance = 1.25f;
            source.rolloffMode = AudioRolloffMode.Linear;
            source.dopplerLevel = 0f;
            source.volume = 1f;
            source.mute = false;
            source.priority = 64;

            if (source.clip == null || !source.clip.name.StartsWith("EnvCueStream"))
            {
                var clip = AudioClip.Create(
                    "EnvCueStream",
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

        private void Update()
        {
            // Unscaled so HITL timeScale=0 still cycles bursts.
            _timer += Time.unscaledDeltaTime;
            lock (_lock)
            {
                if (!_bursting && _timer >= periodSeconds)
                {
                    _bursting = true;
                    _timer = 0f;
                }
                else if (_bursting && _timer >= burstSeconds)
                {
                    _bursting = false;
                    _timer = 0f;
                }
            }
        }

        private void OnAudioRead(float[] data)
        {
            if (data == null)
            {
                return;
            }

            bool bursting;
            float amp;
            float freq;
            lock (_lock)
            {
                bursting = _bursting;
                amp = amplitude;
                freq = frequency;
            }

            float level = bursting ? amp : 0f;
            double dt = 1.0 / _sampleRate;
            for (int i = 0; i < data.Length; i++)
            {
                _phase += 2.0 * Mathf.PI * freq * dt;
                if (_phase > 2.0 * Mathf.PI)
                {
                    _phase -= 2.0 * Mathf.PI;
                }

                data[i] = level * Mathf.Sin((float)_phase);
            }
        }

        /// <summary>EditMode: force a burst on.</summary>
        public void TriggerBurstForTests()
        {
            lock (_lock)
            {
                _bursting = true;
                _timer = 0f;
            }
        }
    }
}
