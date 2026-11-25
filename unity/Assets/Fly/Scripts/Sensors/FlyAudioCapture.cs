using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Ear: samples the scene mix (exclusive <see cref="AudioListener"/> on the fly)
    /// into band energies for the Python speech cortex (no ML ASR).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FlyAudioCapture : MonoBehaviour
    {
        public const int BandCount = 16;

        [SerializeField] private int ringSeconds = 1;
        [SerializeField] private float voicedThreshold = 0.04f;

        private float[] _ring;
        private int _write;
        private readonly float[] _bands = new float[BandCount];
        private float _rms;
        private float _voiced;
        private readonly object _lock = new object();
        private float[] _listenerScratch;

        public float Rms => _rms;
        public float Voiced => _voiced;
        public float[] Bands => (float[])_bands.Clone();

        private void Awake()
        {
            ClaimExclusiveListener();
            int samples = Mathf.Max(1024, AudioSettings.outputSampleRate * Mathf.Max(1, ringSeconds));
            _ring = new float[samples];
            _listenerScratch = new float[1024];
        }

        private void OnEnable()
        {
            ClaimExclusiveListener();
        }

        /// <summary>
        /// One listener only — overview/follow cameras keep video, fly owns hearing
        /// so pad/mouth spatial audio is near-field and audible in the Game view.
        /// </summary>
        private void ClaimExclusiveListener()
        {
            foreach (var other in FindObjectsByType<AudioListener>())
            {
                if (other != null && other.gameObject != gameObject)
                {
                    other.enabled = false;
                }
            }

            var mine = GetComponent<AudioListener>();
            if (mine == null)
            {
                mine = gameObject.AddComponent<AudioListener>();
            }

            mine.enabled = true;
        }

        private void OnAudioFilterRead(float[] data, int channels)
        {
            // Listener mix callback: record without muting the speakers.
            if (_ring == null || data == null || data.Length == 0)
            {
                return;
            }

            lock (_lock)
            {
                int ch = Mathf.Max(1, channels);
                for (int i = 0; i < data.Length; i += ch)
                {
                    float s = 0f;
                    for (int c = 0; c < ch; c++)
                    {
                        s += data[i + c];
                    }

                    s /= ch;
                    _ring[_write] = s;
                    _write = (_write + 1) % _ring.Length;
                }
            }
        }

        private void LateUpdate()
        {
            // Backup path if filter callback is sparse: pull listener PCM.
            if (_listenerScratch != null && _listenerScratch.Length > 0)
            {
                try
                {
                    AudioListener.GetOutputData(_listenerScratch, 0);
                    lock (_lock)
                    {
                        if (_ring != null)
                        {
                            for (int i = 0; i < _listenerScratch.Length; i++)
                            {
                                _ring[_write] = _listenerScratch[i];
                                _write = (_write + 1) % _ring.Length;
                            }
                        }
                    }
                }
                catch
                {
                    // GetOutputData can throw if no listener yet.
                }
            }

            Analyze();
        }

        /// <summary>EditMode / tests: inject samples into the ring buffer.</summary>
        public void InjectSamplesForTests(float[] mono)
        {
            if (mono == null || mono.Length == 0)
            {
                return;
            }

            if (_ring == null)
            {
                _ring = new float[Mathf.Max(1024, mono.Length * 2)];
            }

            lock (_lock)
            {
                for (int i = 0; i < mono.Length; i++)
                {
                    _ring[_write] = mono[i];
                    _write = (_write + 1) % _ring.Length;
                }
            }

            Analyze();
        }

        public void CopyObservation(float[] buffer, int offset)
        {
            if (buffer == null || offset + BandCount + 2 > buffer.Length)
            {
                return;
            }

            for (int i = 0; i < BandCount; i++)
            {
                buffer[offset + i] = _bands[i];
            }

            buffer[offset + BandCount] = _rms;
            buffer[offset + BandCount + 1] = _voiced;
        }

        private void Analyze()
        {
            if (_ring == null || _ring.Length < BandCount)
            {
                return;
            }

            float[] snap;
            lock (_lock)
            {
                snap = (float[])_ring.Clone();
            }

            int win = Mathf.Min(1024, snap.Length);
            int start = (_write - win + snap.Length) % snap.Length;
            double sumSq = 0.0;
            for (int b = 0; b < BandCount; b++)
            {
                _bands[b] = 0f;
            }

            for (int i = 0; i < win; i++)
            {
                float s = snap[(start + i) % snap.Length];
                sumSq += s * s;
                int band = (i * BandCount) / win;
                if (band >= BandCount)
                {
                    band = BandCount - 1;
                }

                _bands[band] += s * s;
            }

            _rms = Mathf.Sqrt((float)(sumSq / Mathf.Max(1, win)));
            float maxBand = 0f;
            for (int b = 0; b < BandCount; b++)
            {
                _bands[b] = Mathf.Sqrt(_bands[b] / Mathf.Max(1, win / BandCount));
                if (_bands[b] > maxBand)
                {
                    maxBand = _bands[b];
                }
            }

            if (maxBand > 1e-6f)
            {
                for (int b = 0; b < BandCount; b++)
                {
                    _bands[b] = Mathf.Clamp01(_bands[b] / maxBand);
                }
            }

            _voiced = _rms >= voicedThreshold ? 1f : 0f;
        }
    }
}
