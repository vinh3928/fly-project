using UnityEngine;
using UnityEngine.UI;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Shows left/right fly-eye RenderTextures as Game-view PIP panels (Canvas UI).
    /// Eye Cameras stay disabled; we Camera.Render() into their RTs each frame.
    /// Toggle with V. Works in the Game tab (not only Scene/Simulator camera previews).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FlyEyeViewOverlay : MonoBehaviour
    {
        [SerializeField] private bool showOverlay = true;
        [SerializeField] private KeyCode toggleKey = KeyCode.V;
        [SerializeField] private float panelPixels = 180f;
        [SerializeField] private float margin = 12f;

        private FlyEyeCamera _left;
        private FlyEyeCamera _right;
        private Canvas _canvas;
        private RawImage _leftImage;
        private RawImage _rightImage;
        private Text _hint;

        public bool ShowOverlay
        {
            get => showOverlay;
            set
            {
                showOverlay = value;
                ApplyVisibility();
            }
        }

        private void OnEnable()
        {
            EnsureUi();
            ResolveEyes();
            ApplyVisibility();
        }

        private void OnDisable()
        {
            if (_canvas != null)
            {
                _canvas.gameObject.SetActive(false);
            }
        }

        private void Update()
        {
            if (Input.GetKeyDown(toggleKey))
            {
                ShowOverlay = !showOverlay;
            }
        }

        private void LateUpdate()
        {
            if (!showOverlay)
            {
                return;
            }

            EnsureUi();
            ResolveEyes();
            RenderEye(_left);
            RenderEye(_right);
            Bind(_leftImage, _left);
            Bind(_rightImage, _right);
        }

        private void EnsureUi()
        {
            if (_canvas != null)
            {
                return;
            }

            var root = new GameObject("FlyEyeViewCanvas");
            root.transform.SetParent(transform, false);
            _canvas = root.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 5000;
            var scaler = root.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
            scaler.scaleFactor = 1f;
            root.AddComponent<GraphicRaycaster>();

            _leftImage = CreatePanel(root.transform, "LeftEyePanel", anchorMin: new Vector2(0f, 0f), anchorMax: new Vector2(0f, 0f), pivot: new Vector2(0f, 0f));
            _rightImage = CreatePanel(root.transform, "RightEyePanel", anchorMin: new Vector2(1f, 0f), anchorMax: new Vector2(1f, 0f), pivot: new Vector2(1f, 0f));

            var hintGo = new GameObject("Hint");
            hintGo.transform.SetParent(root.transform, false);
            _hint = hintGo.AddComponent<Text>();
            _hint.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (_hint.font == null)
            {
                // Older editors; may be null on some Unity 6 installs — hide hint if so.
                _hint.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }

            if (_hint.font == null)
            {
                Destroy(hintGo);
                _hint = null;
            }
            else
            {
                _hint.fontSize = 14;
                _hint.color = Color.white;
                _hint.supportRichText = false;
                _hint.text = "Fly eyes (V = toggle) — Game view";
                var hintRt = _hint.rectTransform;
                hintRt.anchorMin = new Vector2(0f, 0f);
                hintRt.anchorMax = new Vector2(0f, 0f);
                hintRt.pivot = new Vector2(0f, 0f);
                hintRt.anchoredPosition = new Vector2(margin, margin + panelPixels + 4f);
                hintRt.sizeDelta = new Vector2(360f, 20f);
            }

            LayoutPanels();
            ApplyVisibility();
        }

        private RawImage CreatePanel(Transform parent, string name, Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<RawImage>();
            image.color = Color.white;
            var rt = image.rectTransform;
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.pivot = pivot;
            // Dark frame behind via Outline-like second image is optional; keep simple.
            return image;
        }

        private void LayoutPanels()
        {
            float size = Mathf.Clamp(panelPixels, 64f, 512f);
            if (_leftImage != null)
            {
                _leftImage.rectTransform.anchoredPosition = new Vector2(margin, margin);
                _leftImage.rectTransform.sizeDelta = new Vector2(size, size);
            }

            if (_rightImage != null)
            {
                _rightImage.rectTransform.anchoredPosition = new Vector2(-margin, margin);
                _rightImage.rectTransform.sizeDelta = new Vector2(size, size);
            }

            if (_hint != null)
            {
                _hint.rectTransform.anchoredPosition = new Vector2(margin, margin + size + 4f);
            }
        }

        private void ApplyVisibility()
        {
            if (_canvas != null)
            {
                _canvas.gameObject.SetActive(showOverlay);
            }
        }

        private void ResolveEyes()
        {
            if (_left != null && _right != null)
            {
                return;
            }

            _left = null;
            _right = null;
            var eyes = FindObjectsByType<FlyEyeCamera>();
            foreach (var eye in eyes)
            {
                if (eye == null)
                {
                    continue;
                }

                if (eye.IsLeftEye)
                {
                    _left = eye;
                }
                else
                {
                    _right = eye;
                }
            }
        }

        private static void RenderEye(FlyEyeCamera eye)
        {
            if (eye == null)
            {
                return;
            }

            eye.EnsureRenderTarget();
            var cam = eye.GetComponent<Camera>();
            if (cam == null || eye.RenderTexture == null)
            {
                return;
            }

            cam.Render();
            eye.MaskRenderTextureInPlace();
        }

        private static void Bind(RawImage image, FlyEyeCamera eye)
        {
            if (image == null)
            {
                return;
            }

            image.texture = eye != null ? eye.RenderTexture : null;
            image.color = image.texture != null ? Color.white : new Color(0.1f, 0.1f, 0.1f, 0.85f);
        }
    }
}
