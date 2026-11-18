using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Low-resolution eye camera feeding a RenderTexture observation.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public sealed class FlyEyeCamera : MonoBehaviour
    {
        /// <summary>
        /// Near clip in meters. The near plane's corners must stay inside the solid head sphere
        /// (<see cref="Body.FlyBodyController.HeadRadius"/>) at <see cref="MountOffset"/> /
        /// <see cref="MountRotation"/>, or a wall touching the head is clipped away and the eye
        /// sees through it. 0.6 mm fits the 4 mm spacing with ±2° outward yaw.
        /// </summary>
        public const float NearClip = 0.0006f;

        /// <summary>Half the left/right eye spacing in meters (4 mm apart).</summary>
        public const float HalfSpacing = 0.002f;

        /// <summary>
        /// Yaw of each eye away from body forward in degrees (left −yaw, right +yaw).
        /// Gives distinct left/right views in the eye PIP overlay.
        /// </summary>
        public const float MountYawDegrees = 2f;

        /// <summary>Default vertical field of view in degrees.</summary>
        public const float DefaultFieldOfView = 140f;

        /// <summary>Far clip in meters: far enough to see the outside world's trees and town.</summary>
        public const float FarClip = FlySimulation.Environment.OutsideWorldLayout.ViewDistance;

        /// <summary>TagManager "Fly" layer: the fly's own antennae / proboscis, culled from the eyes.</summary>
        public const int SelfLayer = 8;

        [SerializeField] private int width = 64;
        [SerializeField] private int height = 64;
        [SerializeField] private float fieldOfView = DefaultFieldOfView;
        [SerializeField] private bool grayscale = false;
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

            // 24-bit depth: NearClip..FarClip spans five orders of magnitude; 16 bits z-fights.
            _renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32)
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
            _camera.nearClipPlane = NearClip;
            _camera.cullingMask = ~(1 << SelfLayer);
            _camera.farClipPlane = FarClip;
            _camera.allowHDR = false;
            _camera.allowMSAA = false;
        }

        /// <summary>
        /// Zero pixels outside the hexagonal aperture on the live RenderTexture
        /// (PIP / CameraSensor path that only calls Camera.Render).
        /// </summary>
        public void MaskRenderTextureInPlace()
        {
            EnsureRenderTarget();
            if (_camera == null || _renderTexture == null || _readback == null)
            {
                return;
            }

            var previous = RenderTexture.active;
            RenderTexture.active = _renderTexture;
            _readback.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            _readback.Apply(false);
            var pixels = _readback.GetPixels32();
            FlyEyeHexGeometry.ApplyApertureMask(pixels, width, height);
            // PIP-only guides so Game-view panels show facets; CaptureFrame stays clean.
            FlyEyeHexGeometry.DrawFacetGuides(pixels, width, height, new Color32(40, 220, 120, 255));
            _readback.SetPixels32(pixels);
            _readback.Apply(false);
            Graphics.Blit(_readback, _renderTexture);
            RenderTexture.active = previous;
        }

        /// <summary>
        /// Render and read back without the hex aperture mask, bottom row first — what the
        /// ML-Agents camera sensor sends to Python (after its top-first flip).
        /// </summary>
        public Color32[] CaptureRawPixels()
        {
            EnsureRenderTarget();
            if (_camera == null || _renderTexture == null)
            {
                return null;
            }

            var previous = RenderTexture.active;
            RenderTexture.active = _renderTexture;
            _camera.Render();
            _readback.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            _readback.Apply(false);
            RenderTexture.active = previous;
            return _readback.GetPixels32();
        }

        public float[] CaptureFrame()
        {
            if (_camera == null || _renderTexture == null)
            {
                return _frameBuffer;
            }

            var pixels = CaptureRawPixels();
            FlyEyeHexGeometry.ApplyApertureMask(pixels, width, height);
            _readback.SetPixels32(pixels);
            _readback.Apply(false);

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

        /// <summary>
        /// Local offset on an unscaled mount at the head center; +X is the fly's right.
        /// Parent under <c>AntennaMount</c> (not the scaled Head mesh) so spacing is true meters.
        /// </summary>
        public static Vector3 MountOffset(bool isLeft)
        {
            return new Vector3(isLeft ? -HalfSpacing : HalfSpacing, 0f, 0f);
        }

        /// <summary>Local yaw: left eye −<see cref="MountYawDegrees"/>, right eye +.</summary>
        public static Quaternion MountRotation(bool isLeft)
        {
            float yaw = isLeft ? -MountYawDegrees : MountYawDegrees;
            return Quaternion.Euler(0f, yaw, 0f);
        }

        /// <summary>Distance from the eye to a corner of its near clip plane.</summary>
        public static float NearPlaneCornerDistance(float verticalFovDegrees, float aspect)
        {
            float tanV = Mathf.Tan(0.5f * verticalFovDegrees * Mathf.Deg2Rad);
            float tanH = tanV * aspect;
            return NearClip * Mathf.Sqrt(1f + tanV * tanV + tanH * tanH);
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
