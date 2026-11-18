using System.Text;
using UnityEngine;
using UnityEngine.UI;
#if UNITY_MLAGENTS
using FlySimulation.Fly.DebugViz;
#endif

namespace FlySimulation.Environment
{
    /// <summary>
    /// Display 3: scrolling human-readable mind / deep-thought log from Python.
    /// Camera clears a dark backdrop; Canvas overlays the text on targetDisplay 2.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public sealed class MindLogDisplay : MonoBehaviour
    {
        public const string ObjectName = "MindLogDisplay";
        public const int TargetDisplayIndex = 2; // Unity Display 3

        [SerializeField] private int maxEntries = 12;
        [SerializeField] private bool activateDisplay3 = true;
        [SerializeField] private Color background = new Color(0.06f, 0.07f, 0.09f, 1f);
        [SerializeField] private Color textColor = new Color(0.85f, 0.92f, 0.78f, 1f);
        [SerializeField] private int fontSize = 18;

        private Camera _camera;
        private Canvas _canvas;
        private Text _body;
        private Text _header;
        private bool _displayActivated;
        private readonly StringBuilder _builder = new StringBuilder(4096);

#if UNITY_MLAGENTS
        private MindLogDebugChannel _channel;
#endif

        public Camera Camera => _camera != null ? _camera : (_camera = GetComponent<Camera>());
        public Text Body => _body;

        private void Awake()
        {
            _camera = GetComponent<Camera>();
            Place(_camera);
            EnsureUi();
            TryActivateDisplay3();
            SetPlaceholder();
        }

        private void OnEnable()
        {
            Place(Camera);
            EnsureUi();
            TryActivateDisplay3();
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.LogUpdated += OnLogUpdated;
            }
#endif
            RefreshText();
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (_channel != null)
            {
                _channel.LogUpdated -= OnLogUpdated;
                _channel = null;
            }
#endif
        }

        private void TryActivateDisplay3()
        {
            if (!activateDisplay3 || _displayActivated)
            {
                return;
            }

            if (Display.displays.Length > TargetDisplayIndex)
            {
                Display.displays[TargetDisplayIndex].Activate();
                _displayActivated = true;
            }
        }

#if UNITY_MLAGENTS
        private void EnsureChannel()
        {
            if (_channel != null)
            {
                return;
            }

            try
            {
                _channel = MindLogDebugChannel.Register(maxEntries);
            }
            catch (System.Exception)
            {
                // EditMode / no Academy: keep a local buffer for ApplyEntry tests.
                _channel = new MindLogDebugChannel(maxEntries);
            }
        }

        private void OnLogUpdated()
        {
            RefreshText();
        }
#endif

        /// <summary>Push a panel from EditMode tests (no SideChannel required).</summary>
        public void ApplyEntry(int kind, int step, string text)
        {
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.ApplyEntry(kind, step, text);
                RefreshText();
                return;
            }
#endif
            SetBody(text ?? "");
        }

        private void RefreshText()
        {
#if UNITY_MLAGENTS
            if (_channel == null || _channel.Entries.Count == 0)
            {
                SetPlaceholder();
                return;
            }

            _builder.Length = 0;
            for (int i = 0; i < _channel.Entries.Count; i++)
            {
                if (i > 0)
                {
                    _builder.Append("\n\n——\n\n");
                }

                _builder.Append(_channel.Entries[i]);
            }

            SetBody(_builder.ToString());
#else
            SetPlaceholder();
#endif
        }

        private void SetPlaceholder()
        {
            SetBody(
                "Mind log (Display 3)\n" +
                "Waiting for Python live-mind thoughts…\n" +
                "Shows MIND* and DEEP* human-readable panels.");
        }

        private void SetBody(string text)
        {
            EnsureUi();
            if (_body != null)
            {
                _body.text = text;
            }

            if (_header != null)
            {
                _header.text = "Fly mind / deep thought";
            }
        }

        private void EnsureUi()
        {
            if (_canvas != null && _body != null)
            {
                _canvas.targetDisplay = TargetDisplayIndex;
                return;
            }

            var root = new GameObject("MindLogCanvas");
            root.transform.SetParent(transform, false);
            _canvas = root.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.targetDisplay = TargetDisplayIndex;
            _canvas.sortingOrder = 100;
            var scaler = root.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280f, 720f);
            root.AddComponent<GraphicRaycaster>();

            var bgGo = new GameObject("Backdrop");
            bgGo.transform.SetParent(root.transform, false);
            var bg = bgGo.AddComponent<Image>();
            bg.color = background;
            var bgRt = bg.rectTransform;
            bgRt.anchorMin = Vector2.zero;
            bgRt.anchorMax = Vector2.one;
            bgRt.offsetMin = Vector2.zero;
            bgRt.offsetMax = Vector2.zero;

            _header = CreateText(root.transform, "Header", 22, FontStyle.Bold);
            var headerRt = _header.rectTransform;
            headerRt.anchorMin = new Vector2(0f, 1f);
            headerRt.anchorMax = new Vector2(1f, 1f);
            headerRt.pivot = new Vector2(0.5f, 1f);
            headerRt.anchoredPosition = new Vector2(0f, -16f);
            headerRt.sizeDelta = new Vector2(-48f, 32f);
            _header.alignment = TextAnchor.UpperLeft;

            _body = CreateText(root.transform, "Body", fontSize, FontStyle.Normal);
            var bodyRt = _body.rectTransform;
            bodyRt.anchorMin = new Vector2(0f, 0f);
            bodyRt.anchorMax = new Vector2(1f, 1f);
            bodyRt.offsetMin = new Vector2(24f, 24f);
            bodyRt.offsetMax = new Vector2(-24f, -56f);
            _body.alignment = TextAnchor.UpperLeft;
            _body.horizontalOverflow = HorizontalWrapMode.Wrap;
            _body.verticalOverflow = VerticalWrapMode.Truncate;
        }

        private Text CreateText(Transform parent, string name, int size, FontStyle style)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var text = go.AddComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (text.font == null)
            {
                text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }

            text.fontSize = size;
            text.fontStyle = style;
            text.color = textColor;
            text.supportRichText = false;
            text.raycastTarget = false;
            return text;
        }

        public static void Place(Camera cam)
        {
            if (cam == null)
            {
                return;
            }

            cam.enabled = true;
            cam.tag = "Untagged";
            cam.targetDisplay = TargetDisplayIndex;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.06f, 0.07f, 0.09f, 1f);
            cam.cullingMask = 0; // UI overlay only
            cam.depth = 40f;
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 10f;
            cam.targetTexture = null;
        }

        public static bool IsLogCamera(Camera cam)
        {
            if (cam == null)
            {
                return false;
            }

            return cam.GetComponent<MindLogDisplay>() != null || cam.gameObject.name == ObjectName;
        }
    }
}
