using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
#if UNITY_MLAGENTS
using FlySimulation.Fly.DebugViz;
#endif

namespace FlySimulation.Environment
{
    /// <summary>
    /// Display 3 overlay: Father-advocate question with human text reply or Ask LLM.
    /// Lives beside <see cref="MindLogDisplay"/> (same target display).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FatherDialogDisplay : MonoBehaviour
    {
        public const string ObjectName = "FatherDialogDisplay";
        public const int TargetDisplayIndex = MindLogDisplay.TargetDisplayIndex;

        [SerializeField] private Color panelColor = new Color(0.08f, 0.1f, 0.14f, 0.96f);
        [SerializeField] private Color textColor = new Color(0.92f, 0.94f, 0.88f, 1f);
        [SerializeField] private Color accentColor = new Color(0.55f, 0.78f, 0.95f, 1f);
        [SerializeField] private int fontSize = 18;

        private Canvas _canvas;
        private GameObject _panelRoot;
        private Text _title;
        private Text _cue;
        private Text _question;
        private InputField _input;
        private Button _submit;
        private Button _askLlm;
        private Button _skip;
        private bool _visible;
        private bool _allowLlm = true;
        private int _pendingRequestId;
        private float _awaySeconds;
        private bool _lastViewingReported;
        private bool _hasViewingReport;
        // Match mind.yaml father_advocate.auto_llm_away_seconds.
        [SerializeField] private float autoLlmAwaySeconds = 5f;

#if UNITY_MLAGENTS
        private FatherDialogDebugChannel _channel;
#endif

        public bool IsVisible => _visible;
        public string QuestionText => _question != null ? _question.text : "";

        private void Awake()
        {
            EnsureEventSystem();
            EnsureUi();
            Hide();
        }

        private void Update()
        {
            if (!_visible)
            {
                _awaySeconds = 0f;
                return;
            }

            TickAwayAutoAskLlm(Time.unscaledDeltaTime);
        }

        private void OnEnable()
        {
            EnsureEventSystem();
            EnsureUi();
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.AskReceived += OnAsk;
                _channel.HideReceived += OnHide;
                if (_channel.IsVisible)
                {
                    OnAsk();
                }
            }
#endif
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (_channel != null)
            {
                _channel.AskReceived -= OnAsk;
                _channel.HideReceived -= OnHide;
                _channel = null;
            }
#endif
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
                _channel = FatherDialogDebugChannel.Register();
            }
            catch (System.Exception)
            {
                _channel = new FatherDialogDebugChannel();
            }
        }

        private void OnAsk()
        {
            if (_channel == null)
            {
                return;
            }

            Show(
                _channel.LastRequestId,
                _channel.LastCue,
                _channel.LastQuestion,
                _channel.AllowLlm);
        }

        private void OnHide()
        {
            Hide();
        }
#endif

        /// <summary>EditMode helper: show a dialog without SideChannel.</summary>
        public void ShowForTests(int requestId, string cue, string question, bool allowLlm = true)
        {
            Show(requestId, cue, question, allowLlm);
        }

        public void Show(int requestId, string cue, string question, bool allowLlm)
        {
            EnsureUi();
            _visible = true;
            _allowLlm = allowLlm;
            _awaySeconds = 0f;
            _hasViewingReport = false;
            if (_panelRoot != null)
            {
                _panelRoot.SetActive(true);
            }

            if (_title != null)
            {
                _title.text = "Father advocate";
            }

            if (_cue != null)
            {
                _cue.text = string.IsNullOrEmpty(cue) ? "cue: (unknown)" : "cue: " + cue;
            }

            if (_question != null)
            {
                _question.text = question ?? "";
            }

            if (_input != null)
            {
                _input.text = "";
                _input.ActivateInputField();
            }

            if (_askLlm != null)
            {
                _askLlm.gameObject.SetActive(allowLlm);
            }

            _pendingRequestId = requestId;
            ReportViewingIfChanged(IsUserViewingDisplay3());
        }

        public void Hide()
        {
            _visible = false;
            _awaySeconds = 0f;
            if (_panelRoot != null)
            {
                _panelRoot.SetActive(false);
            }
        }

        private bool IsUserViewingDisplay3()
        {
#if UNITY_EDITOR
            // 0 = Display 1 … 2 = Display 3 (Game view Display dropdown).
            return Display.activeEditorGameViewTarget == TargetDisplayIndex;
#else
            return Display.displays.Length > TargetDisplayIndex;
#endif
        }

        private void TickAwayAutoAskLlm(float dt)
        {
            if (!_visible || !_allowLlm)
            {
                _awaySeconds = 0f;
                return;
            }

            bool viewing = IsUserViewingDisplay3();
            ReportViewingIfChanged(viewing);
            if (viewing)
            {
                // User on Display 3 — unlimited time for the UI.
                _awaySeconds = 0f;
                return;
            }

            if (autoLlmAwaySeconds < 0f)
            {
                return;
            }

            _awaySeconds += Mathf.Max(0f, dt);
            if (_awaySeconds < autoLlmAwaySeconds)
            {
                return;
            }

            SubmitAskLlm();
        }

        private void ReportViewingIfChanged(bool viewing)
        {
            if (_hasViewingReport && viewing == _lastViewingReported)
            {
                return;
            }

            _hasViewingReport = true;
            _lastViewingReported = viewing;
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.SendViewing(_pendingRequestId, viewing);
            }
#endif
        }

        private void SubmitHuman()
        {
            string answer = _input != null ? _input.text : "";
            SendReply(FatherDialogDebugChannelSource.Human, answer);
        }

        private void SubmitAskLlm()
        {
            SendReply(FatherDialogDebugChannelSource.AskLlm, "");
        }

        private void SubmitSkip()
        {
            SendReply(FatherDialogDebugChannelSource.Skip, "");
        }

        private void SendReply(int source, string answer)
        {
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.SendReply(_pendingRequestId, source, answer ?? "");
            }
#endif
            Hide();
        }

        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null)
            {
                return;
            }

            var go = new GameObject("EventSystem");
            go.AddComponent<EventSystem>();
            go.AddComponent<StandaloneInputModule>();
        }

        private void EnsureUi()
        {
            if (_canvas != null && _panelRoot != null)
            {
                _canvas.targetDisplay = TargetDisplayIndex;
                return;
            }

            var root = new GameObject("FatherDialogCanvas");
            root.transform.SetParent(transform, false);
            _canvas = root.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.targetDisplay = TargetDisplayIndex;
            _canvas.sortingOrder = 200; // above MindLog
            var scaler = root.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280f, 720f);
            root.AddComponent<GraphicRaycaster>();

            _panelRoot = new GameObject("DialogPanel");
            _panelRoot.transform.SetParent(root.transform, false);
            var panelImage = _panelRoot.AddComponent<Image>();
            panelImage.color = panelColor;
            var panelRt = panelImage.rectTransform;
            panelRt.anchorMin = new Vector2(0.12f, 0.28f);
            panelRt.anchorMax = new Vector2(0.88f, 0.78f);
            panelRt.offsetMin = Vector2.zero;
            panelRt.offsetMax = Vector2.zero;

            _title = CreateText(_panelRoot.transform, "Title", 22, FontStyle.Bold, TextAnchor.UpperLeft);
            PlaceText(_title, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(16f, -12f), new Vector2(-16f, 36f));

            _cue = CreateText(_panelRoot.transform, "Cue", 16, FontStyle.Italic, TextAnchor.UpperLeft);
            _cue.color = accentColor;
            PlaceText(_cue, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(16f, -48f), new Vector2(-16f, 28f));

            _question = CreateText(_panelRoot.transform, "Question", fontSize, FontStyle.Normal, TextAnchor.UpperLeft);
            PlaceText(_question, new Vector2(0f, 0.42f), new Vector2(1f, 1f), new Vector2(16f, -84f), new Vector2(-16f, -8f));
            _question.horizontalOverflow = HorizontalWrapMode.Wrap;
            _question.verticalOverflow = VerticalWrapMode.Overflow;

            var inputGo = new GameObject("ReplyInput");
            inputGo.transform.SetParent(_panelRoot.transform, false);
            var inputBg = inputGo.AddComponent<Image>();
            inputBg.color = new Color(0.04f, 0.05f, 0.07f, 1f);
            var inputRt = inputBg.rectTransform;
            inputRt.anchorMin = new Vector2(0f, 0.18f);
            inputRt.anchorMax = new Vector2(1f, 0.4f);
            inputRt.offsetMin = new Vector2(16f, 0f);
            inputRt.offsetMax = new Vector2(-16f, 0f);

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(inputGo.transform, false);
            var inputText = textGo.AddComponent<Text>();
            StyleText(inputText, fontSize, FontStyle.Normal);
            inputText.supportRichText = false;
            inputText.alignment = TextAnchor.UpperLeft;
            var textRt = inputText.rectTransform;
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = new Vector2(10f, 8f);
            textRt.offsetMax = new Vector2(-10f, -8f);

            var placeholderGo = new GameObject("Placeholder");
            placeholderGo.transform.SetParent(inputGo.transform, false);
            var placeholder = placeholderGo.AddComponent<Text>();
            StyleText(placeholder, fontSize, FontStyle.Italic);
            placeholder.color = new Color(textColor.r, textColor.g, textColor.b, 0.45f);
            placeholder.text = "Type your teaching reply…";
            placeholder.alignment = TextAnchor.UpperLeft;
            var phRt = placeholder.rectTransform;
            phRt.anchorMin = Vector2.zero;
            phRt.anchorMax = Vector2.one;
            phRt.offsetMin = new Vector2(10f, 8f);
            phRt.offsetMax = new Vector2(-10f, -8f);

            _input = inputGo.AddComponent<InputField>();
            _input.textComponent = inputText;
            _input.placeholder = placeholder;
            _input.lineType = InputField.LineType.MultiLineNewline;
            _input.interactable = true;

            _submit = CreateButton(_panelRoot.transform, "Submit", "Reply", new Vector2(0.02f, 0.04f), new Vector2(0.32f, 0.15f));
            _submit.onClick.AddListener(SubmitHuman);

            _askLlm = CreateButton(_panelRoot.transform, "AskLlm", "Ask LLM", new Vector2(0.34f, 0.04f), new Vector2(0.66f, 0.15f));
            _askLlm.onClick.AddListener(SubmitAskLlm);

            _skip = CreateButton(_panelRoot.transform, "Skip", "Skip", new Vector2(0.68f, 0.04f), new Vector2(0.98f, 0.15f));
            _skip.onClick.AddListener(SubmitSkip);
        }

        private Text CreateText(Transform parent, string name, int size, FontStyle style, TextAnchor align)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var text = go.AddComponent<Text>();
            StyleText(text, size, style);
            text.alignment = align;
            text.raycastTarget = false;
            return text;
        }

        private void StyleText(Text text, int size, FontStyle style)
        {
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (text.font == null)
            {
                text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }

            text.fontSize = size;
            text.fontStyle = style;
            text.color = textColor;
            text.supportRichText = false;
        }

        private static void PlaceText(Text text, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPos, Vector2 size)
        {
            var rt = text.rectTransform;
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = anchoredPos;
            rt.sizeDelta = size;
        }

        private Button CreateButton(Transform parent, string name, string label, Vector2 anchorMin, Vector2 anchorMax)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<Image>();
            image.color = new Color(0.18f, 0.28f, 0.38f, 1f);
            var rt = image.rectTransform;
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            var button = go.AddComponent<Button>();
            var text = CreateText(go.transform, "Label", 16, FontStyle.Bold, TextAnchor.MiddleCenter);
            var textRt = text.rectTransform;
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = Vector2.zero;
            textRt.offsetMax = Vector2.zero;
            text.text = label;
            text.raycastTarget = false;
            return button;
        }
    }

    /// <summary>Source codes mirrored from FatherDialogDebugChannel without #if.</summary>
    internal static class FatherDialogDebugChannelSource
    {
        public const int Human = 0;
        public const int AskLlm = 1;
        public const int Skip = 2;
    }
}
