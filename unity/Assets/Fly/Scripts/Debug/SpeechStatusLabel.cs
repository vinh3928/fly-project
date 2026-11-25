using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// White overhead text above the fly: speech_hear / speech_babble alphabets.
    /// Replaces the legacy place-graph plume/s label.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SpeechStatusLabel : MonoBehaviour
    {
        public const string ObjectName = "SpeechStatusLabel";
        public const float DefaultYLift = 0.06f;

        [SerializeField] private Transform flyRoot;
        [SerializeField] private float yLift = DefaultYLift;

        private GameObject _labelRoot;
        private TextMesh _label;
        private string _hear = "";
        private string _babble = "";

        public string HearAlphabet => _hear;
        public string BabbleAlphabet => _babble;
        public string LabelText => _label != null ? _label.text : "";
        public int LabelRootCount => CountNamedChildren(ObjectName);

        public void Bind(Transform fly)
        {
            flyRoot = fly;
            EnsureLabel();
            RefreshText();
        }

        private void Awake()
        {
            EnsureLabel();
            RefreshText();
        }

        private void OnEnable()
        {
            if (flyRoot == null)
            {
                var body = Object.FindAnyObjectByType<FlySimulation.Fly.Body.FlyBodyController>();
                if (body != null)
                {
                    flyRoot = body.transform;
                }
            }

#if UNITY_MLAGENTS
            if (SpeechDebugChannel.Active != null)
            {
                SpeechDebugChannel.Active.StatusUpdated += OnStatusUpdated;
                ApplyFromChannel(SpeechDebugChannel.Active);
            }
#endif
            EnsureLabel();
            RefreshText();
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (SpeechDebugChannel.Active != null)
            {
                SpeechDebugChannel.Active.StatusUpdated -= OnStatusUpdated;
            }
#endif
        }

        private void OnDestroy()
        {
            // Keep scene-serialized label; only clear runtime refs.
            _labelRoot = null;
            _label = null;
        }

        private void LateUpdate()
        {
            if (_labelRoot == null)
            {
                EnsureLabel();
                if (_labelRoot == null)
                {
                    return;
                }
            }

            Transform root = flyRoot != null ? flyRoot : transform;
            _labelRoot.transform.position = root.position + Vector3.up * yLift;

            Camera cam = Camera.main;
            if (cam != null)
            {
                _labelRoot.transform.rotation = Quaternion.LookRotation(
                    _labelRoot.transform.position - cam.transform.position);
            }

#if UNITY_MLAGENTS
            if (SpeechDebugChannel.Active != null)
            {
                ApplyFromChannel(SpeechDebugChannel.Active);
            }
#endif
        }

        /// <summary>EditMode / Python-free: set overhead alphabets.</summary>
        public void SetSpeech(string hear, string babble)
        {
            _hear = hear ?? "";
            _babble = babble ?? "";
            EnsureLabel();
            RefreshText();
        }

        public static string FormatLabel(string hear, string babble)
        {
            string h = string.IsNullOrEmpty(hear) ? "-" : hear;
            string b = string.IsNullOrEmpty(babble) ? "-" : babble;
            return $"speech_hear = {h}\nspeech_babble = {b}";
        }

#if UNITY_MLAGENTS
        private void OnStatusUpdated()
        {
            if (SpeechDebugChannel.Active != null)
            {
                ApplyFromChannel(SpeechDebugChannel.Active);
            }
        }

        private void ApplyFromChannel(SpeechDebugChannel channel)
        {
            if (channel == null)
            {
                return;
            }

            SetSpeech(channel.LastHearAlphabet, channel.LastBabbleAlphabet);
        }
#endif

        private void RefreshText()
        {
            if (_label == null)
            {
                return;
            }

            _label.text = FormatLabel(_hear, _babble);
            if (_labelRoot != null && !_labelRoot.activeSelf)
            {
                _labelRoot.SetActive(true);
            }
        }

        private void EnsureLabel()
        {
            if (_labelRoot != null && _label != null)
            {
                CullDuplicateLabelRoots(_labelRoot);
                ApplyDebugLayer(_labelRoot);
                return;
            }

            Transform existing = transform.Find(ObjectName);
            if (existing != null)
            {
                _labelRoot = existing.gameObject;
                _label = _labelRoot.GetComponentInChildren<TextMesh>(true);
                if (_label == null)
                {
                    var labelGo = new GameObject("Label");
                    labelGo.transform.SetParent(_labelRoot.transform, false);
                    _label = labelGo.AddComponent<TextMesh>();
                    ConfigureTextMesh(_label);
                }

                CullDuplicateLabelRoots(_labelRoot);
                ApplyDebugLayer(_labelRoot);
                _labelRoot.SetActive(true);
                return;
            }

            _labelRoot = new GameObject(ObjectName);
            _labelRoot.transform.SetParent(transform, false);
            // Don't persist play-mode clones into Chamber.unity beside the component.
            _labelRoot.hideFlags = HideFlags.DontSave;
            ApplyDebugLayer(_labelRoot);
            var fresh = new GameObject("Label");
            fresh.hideFlags = HideFlags.DontSave;
            fresh.transform.SetParent(_labelRoot.transform, false);
            ApplyDebugLayer(fresh);
            _label = fresh.AddComponent<TextMesh>();
            ConfigureTextMesh(_label);
            _labelRoot.SetActive(true);
        }

        private static void ConfigureTextMesh(TextMesh mesh)
        {
            mesh.characterSize = 0.03f;
            mesh.fontSize = 48;
            mesh.anchor = TextAnchor.LowerCenter;
            mesh.alignment = TextAlignment.Center;
            mesh.color = Color.white;
        }

        private void CullDuplicateLabelRoots(GameObject keep)
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                Transform child = transform.GetChild(i);
                if (child == null || child.name != ObjectName || child.gameObject == keep)
                {
                    continue;
                }

                if (Application.isPlaying)
                {
                    Object.Destroy(child.gameObject);
                }
                else
                {
                    Object.DestroyImmediate(child.gameObject);
                }
            }
        }

        private int CountNamedChildren(string name)
        {
            int count = 0;
            for (int i = 0; i < transform.childCount; i++)
            {
                if (transform.GetChild(i).name == name)
                {
                    count++;
                }
            }

            return count;
        }

        private void ApplyDebugLayer(GameObject go)
        {
            int layer = flyRoot != null
                ? flyRoot.gameObject.layer
                : FlySimulation.Fly.Sensors.FlyEyeCamera.SelfLayer;
            go.layer = layer;
            foreach (Transform child in go.transform)
            {
                child.gameObject.layer = layer;
            }
        }
    }
}
