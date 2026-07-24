using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Low-resolution eye camera feeding a RenderTexture observation.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public sealed class FlyEyeCamera : MonoBehaviour
    {
        [SerializeField] private int width = 64;
        [SerializeField] private int height = 64;
        [SerializeField] private float fieldOfView = 140f;
        [SerializeField] private bool grayscale = true;
        [SerializeField] private bool isLeftEye = true;

        private Camera _camera;
        private RenderTexture _renderTexture;
        private Texture2D _readback;
        private readonly Color32[] _pixels = new Color32[64 * 64];
        private float[] _frameBuffer;

        public bool IsLeftEye => isLeftEye;
        public int Width => width;
        public int Height => height;
        public bool Grayscale => grayscale;
        public RenderTexture RenderTexture => _renderTexture;

        private void Awake()
        {
            _camera = GetComponent<Camera>();
            Configure(width, height, fieldOfView, grayscale);
        }

        /// <summary>Ensure RenderTexture + camera target exist (safe to call every frame).</summary>
        public void EnsureRenderTarget()
        {
            if (_camera == null)
            {
                _camera = GetComponent<Camera>();
            }

            if (_renderTexture == null || _readback == null || _frameBuffer == null)
            {
                Configure(width, height, fieldOfView, grayscale);
            }
            else if (_camera != null && _camera.targetTexture != _renderTexture)
            {
                _camera.targetTexture = _renderTexture;
            }
        }

        public void Configure(int w, int h, float fov, bool useGrayscale)
        {
            width = w;
            height = h;
            fieldOfView = fov;
            grayscale = useGrayscale;

            if (_renderTexture != null)
            {
                _renderTexture.Release();
            }

            _renderTexture = new RenderTexture(width, height, 16, RenderTextureFormat.ARGB32)
            {
                filterMode = FilterMode.Bilinear,
                name = isLeftEye ? "LeftEyeRT" : "RightEyeRT"
            };
            _readback = new Texture2D(width, height, TextureFormat.RGB24, false);
            _frameBuffer = new float[grayscale ? width * height : width * height * 3];

            if (_camera == null)
            {
                _camera = GetComponent<Camera>();
            }

            _camera.fieldOfView = fieldOfView;
            _camera.targetTexture = _renderTexture;
            _camera.nearClipPlane = 0.01f;
            _camera.farClipPlane = 10f;
            _camera.allowHDR = false;
            _camera.allowMSAA = false;
        }

        public float[] CaptureFrame()
        {
            if (_camera == null || _renderTexture == null)
            {
                return _frameBuffer;
            }

            var previous = RenderTexture.active;
            RenderTexture.active = _renderTexture;
            _camera.Render();
            _readback.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            _readback.Apply(false);
            RenderTexture.active = previous;

            var pixels = _readback.GetPixels32();
            int index = 0;
            for (int i = 0; i < pixels.Length; i++)
            {
                Color32 p = pixels[i];
                if (grayscale)
                {
                    _frameBuffer[index++] = (p.r + p.g + p.b) / (3f * 255f);
                }
                else
                {
                    _frameBuffer[index++] = p.r / 255f;
                    _frameBuffer[index++] = p.g / 255f;
                    _frameBuffer[index++] = p.b / 255f;
                }
            }

            return _frameBuffer;
        }

        public byte[] CapturePng()
        {
            CaptureFrame();
            return _readback != null ? _readback.EncodeToPNG() : System.Array.Empty<byte>();
        }

        public Vector3 Forward => transform.forward;
        public Vector3 Position => transform.position;
        public Quaternion Rotation => transform.rotation;

        private void OnDestroy()
        {
            if (_renderTexture != null)
            {
                _renderTexture.Release();
            }
        }
    }
}
