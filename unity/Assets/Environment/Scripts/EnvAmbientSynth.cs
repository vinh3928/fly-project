using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Procedural ambient layers for the sim (no imported clips): wind, bees, chamber song.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class EnvAmbientSynth : MonoBehaviour
    {
        public enum Kind
        {
            Wind = 0,
            Bees = 1,
            Song = 2,
        }

        [SerializeField] private Kind kind = Kind.Wind;
        [SerializeField] private AudioSource source;
        [SerializeField] private float amplitude = 0.35f;
        [SerializeField] private float spatialBlend = 0.85f;
        [SerializeField] private float minDistance = 0.04f;
        [SerializeField] private float maxDistance = 1.5f;

        private int _sampleRate = 48000;
        private double _phaseA;
        private double _phaseB;
        private double _phaseC;
        private double _lfo;
        private uint _noiseState = 0xA341316Cu;
        private int _songNote;
        private int _songSamplesLeft;
        private int _songNoteLen;
        private float _songFreq;
        private readonly object _lock = new object();

        // Chamber "radio": We Wish You a Merry Christmas (C major, Hz + beats).
        // beat ≈ 0.28s → ~107 BPM feel.
        private const float SongBeatSeconds = 0.28f;
        private static readonly float[] SongFreq =
        {
            // We wish you a merry Christmas
            392.00f, 440.00f, 493.88f, 523.25f, 523.25f, 493.88f, 440.00f, 392.00f,
            // We wish you a merry Christmas
            392.00f, 440.00f, 493.88f, 523.25f, 523.25f, 493.88f, 440.00f, 392.00f,
            // We wish you a merry Christmas
            392.00f, 392.00f, 440.00f, 493.88f, 523.25f, 523.25f, 587.33f, 523.25f,
            // And a happy New Year
            493.88f, 440.00f, 392.00f, 440.00f, 392.00f,
        };
        private static readonly float[] SongBeats =
        {
            1f, 1f, 1f, 1f, 1f, 1f, 1f, 2f,
            1f, 1f, 1f, 1f, 1f, 1f, 1f, 2f,
            1f, 1f, 1f, 1f, 1f, 1f, 1f, 2f,
            1f, 1f, 1f, 1f, 3f,
        };

        public Kind SoundKind => kind;

        public void Configure(Kind sound, float amp = -1f, float blend = -1f)
        {
            kind = sound;
            if (amp >= 0f)
            {
                amplitude = amp;
            }
            else
            {
                amplitude = sound switch
                {
                    Kind.Wind => 0.22f,
                    Kind.Bees => 0.28f,
                    Kind.Song => 0.32f,
                    _ => 0.25f,
                };
            }

            if (blend >= 0f)
            {
                spatialBlend = blend;
            }
            else
            {
                spatialBlend = sound switch
                {
                    Kind.Wind => 0.55f,
                    Kind.Bees => 0.9f,
                    Kind.Song => 0.65f,
                    _ => 0.8f,
                };
            }

            ApplySourceSettings();
        }

        private void Awake()
        {
            if (_noiseState == 0)
            {
                _noiseState = 0xA341316Cu;
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
            ApplySourceSettings();

            string clipName = $"Ambient_{kind}";
            if (source.clip == null || source.clip.name != clipName)
            {
                source.clip = AudioClip.Create(
                    clipName,
                    _sampleRate,
                    1,
                    _sampleRate,
                    true,
                    OnAudioRead);
            }

            if (!source.isPlaying)
            {
                source.Play();
            }
        }

        private void ApplySourceSettings()
        {
            if (source == null)
            {
                return;
            }

            source.playOnAwake = true;
            source.loop = true;
            source.spatialBlend = Mathf.Clamp01(spatialBlend);
            source.minDistance = Mathf.Max(0.01f, minDistance);
            source.maxDistance = Mathf.Max(source.minDistance + 0.05f, maxDistance);
            source.rolloffMode = AudioRolloffMode.Linear;
            source.dopplerLevel = 0f;
            source.volume = 1f;
            source.mute = false;
            source.priority = kind == Kind.Song ? 48 : 96;
        }

        private void OnAudioRead(float[] data)
        {
            if (data == null)
            {
                return;
            }

            Kind mode;
            float amp;
            lock (_lock)
            {
                mode = kind;
                amp = amplitude;
            }

            switch (mode)
            {
                case Kind.Bees:
                    FillBees(data, amp);
                    break;
                case Kind.Song:
                    FillSong(data, amp);
                    break;
                default:
                    FillWind(data, amp);
                    break;
            }
        }

        private void FillWind(float[] data, float amp)
        {
            double dt = 1.0 / _sampleRate;
            float prev = 0f;
            for (int i = 0; i < data.Length; i++)
            {
                _lfo += dt * 0.18;
                float breath = 0.55f + 0.45f * Mathf.Sin((float)_lfo);
                float n = NextNoise();
                // Simple 1-pole low-pass → soft whoosh.
                prev = prev * 0.92f + n * 0.08f;
                float gust = prev * breath;
                data[i] = amp * gust * 1.6f;
            }
        }

        private void FillBees(float[] data, float amp)
        {
            double dt = 1.0 / _sampleRate;
            for (int i = 0; i < data.Length; i++)
            {
                _lfo += dt * 7.0;
                float buzzMod = 0.7f + 0.3f * Mathf.Sin((float)_lfo * 0.37f);
                // Wing ~220 Hz + rough AM + a little noise.
                _phaseA += 2.0 * Mathf.PI * 220.0 * dt;
                _phaseB += 2.0 * Mathf.PI * 440.0 * dt;
                if (_phaseA > 2.0 * Mathf.PI)
                {
                    _phaseA -= 2.0 * Mathf.PI;
                }

                if (_phaseB > 2.0 * Mathf.PI)
                {
                    _phaseB -= 2.0 * Mathf.PI;
                }

                float wing = Mathf.Sin((float)_phaseA);
                float harm = 0.35f * Mathf.Sin((float)_phaseB);
                float rasp = NextNoise() * 0.25f;
                data[i] = amp * buzzMod * (0.7f * wing + harm + rasp);
            }
        }

        private void FillSong(float[] data, float amp)
        {
            double dt = 1.0 / _sampleRate;
            for (int i = 0; i < data.Length; i++)
            {
                if (_songSamplesLeft <= 0)
                {
                    int idx = _songNote % SongFreq.Length;
                    _songFreq = SongFreq[idx];
                    float beats = SongBeats[idx % SongBeats.Length];
                    _songNoteLen = Mathf.Max(
                        1,
                        Mathf.RoundToInt(beats * SongBeatSeconds * _sampleRate));
                    _songSamplesLeft = _songNoteLen;
                    _songNote = (_songNote + 1) % SongFreq.Length;
                    _phaseC = 0.0;
                }

                _phaseC += 2.0 * Mathf.PI * _songFreq * dt;
                if (_phaseC > 2.0 * Mathf.PI)
                {
                    _phaseC -= 2.0 * Mathf.PI;
                }

                float env = 1f;
                int attack = Mathf.Max(1, _songNoteLen / 16);
                int release = Mathf.Max(1, _songNoteLen / 5);
                int into = _songNoteLen - _songSamplesLeft;
                if (into < attack)
                {
                    env = into / (float)attack;
                }
                else if (_songSamplesLeft < release)
                {
                    env = _songSamplesLeft / (float)release;
                }

                float fund = Mathf.Sin((float)_phaseC);
                float fifth = 0.22f * Mathf.Sin((float)(_phaseC * 1.5));
                data[i] = amp * env * (0.72f * fund + fifth);
                _songSamplesLeft--;
            }
        }

        private float NextNoise()
        {
            // xorshift
            _noiseState ^= _noiseState << 13;
            _noiseState ^= _noiseState >> 17;
            _noiseState ^= _noiseState << 5;
            return ((_noiseState & 0xFFFF) / 32768f) - 1f;
        }
    }
}
