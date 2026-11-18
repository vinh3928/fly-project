using System.Collections.Generic;
using UnityEngine;
#if UNITY_MLAGENTS
using Unity.MLAgents;
#endif

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// Debug: draws the place-graph route as a lit LineRenderer and a marker on
    /// the fly's current path node. Driven by <see cref="RoutePathDebugChannel"/>.
    /// </summary>
    public sealed class RoutePathVisualizer : MonoBehaviour
    {
        /// <summary>World-space width that reads as ~1px at typical overview zoom.</summary>
        public const float DefaultLineWidth = 0.0015f;
        public const float DefaultMarkerRadius = 0.004f;
        public const float DefaultYOffset = 0.01f;

        [SerializeField] private Transform flyRoot;
        [SerializeField] private float lineWidth = DefaultLineWidth;
        [SerializeField] private float markerRadius = DefaultMarkerRadius;
        [SerializeField] private float yOffset = DefaultYOffset;
        [SerializeField] private Color routeColor = new Color(0.25f, 0.95f, 1f, 0.95f);
        [SerializeField] private Color scanColor = new Color(0.25f, 0.95f, 1f, 0.95f);
        [SerializeField] private Color currentColor = new Color(0.25f, 0.95f, 1f, 0.95f);

        private LineRenderer _line;
        private Transform _marker;
        private MeshRenderer _markerRenderer;
#if UNITY_MLAGENTS
        private RoutePathDebugChannel _channel;
#endif

        public LineRenderer Line => _line;
        public Transform Marker => _marker;

        public void Bind(Transform fly)
        {
            flyRoot = fly;
#if UNITY_MLAGENTS
            EnsureChannel();
#endif
            EnsureVisuals();
        }

        private void Awake()
        {
            EnsureVisuals();
        }

        private void OnEnable()
        {
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.PathUpdated += OnPathUpdated;
            }
#endif
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (_channel != null)
            {
                // Keep the shared SideChannel registered — PlaceGraphVisualizer
                // (and Play-mode reconnect) reuse RoutePathDebugChannel.Active.
                _channel.PathUpdated -= OnPathUpdated;
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

            Transform root = flyRoot != null ? flyRoot : transform;
            _channel = RoutePathDebugChannel.Register(root);
        }

        private void OnPathUpdated()
        {
            if (_channel == null)
            {
                return;
            }

            ApplyPath(_channel.Points, _channel.CurrentIndex, _channel.Mode);
        }

        private void FixedUpdate()
        {
            // Pose must be queued before Academy communicates on the Python step.
            _channel?.SendFlyPose();
        }
#endif

        public void EnsureVisuals()
        {
            if (_line == null)
            {
                var lineGo = new GameObject("RoutePathLine");
                lineGo.transform.SetParent(transform, false);
                ApplyDebugLayer(lineGo);
                _line = lineGo.AddComponent<LineRenderer>();
                _line.positionCount = 0;
                _line.widthMultiplier = 1f;
                _line.startWidth = lineWidth;
                _line.endWidth = lineWidth;
                _line.useWorldSpace = true;
                _line.numCapVertices = 0;
                _line.numCornerVertices = 0;
                _line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                _line.receiveShadows = false;
                _line.material = CreateUnlit(routeColor);
            }

            if (_marker == null)
            {
                var markerGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                markerGo.name = "RoutePathCurrent";
                markerGo.transform.SetParent(transform, false);
                ApplyDebugLayer(markerGo);
                markerGo.transform.localScale = Vector3.one * (markerRadius * 2f);
                Object.Destroy(markerGo.GetComponent<Collider>());
                _marker = markerGo.transform;
                _markerRenderer = markerGo.GetComponent<MeshRenderer>();
                _markerRenderer.sharedMaterial = CreateUnlit(currentColor);
                markerGo.SetActive(false);
            }
        }

        private void ApplyDebugLayer(GameObject go)
        {
            // Eyes cull Fly layer — keep the cyan path out of eye RTs.
            int layer = flyRoot != null
                ? flyRoot.gameObject.layer
                : FlySimulation.Fly.Sensors.FlyEyeCamera.SelfLayer;
            go.layer = layer;
        }

        /// <summary>Apply a path (also used by EditMode tests without SideChannel).</summary>
        public void ApplyPath(IReadOnlyList<Vector3> points, int currentIndex, int mode)
        {
            EnsureVisuals();
            if (points == null || points.Count == 0 || mode == 0)
            {
                _line.positionCount = 0;
                if (_marker != null)
                {
                    _marker.gameObject.SetActive(false);
                }

                return;
            }

            Color lineColor = mode == 2 ? scanColor : routeColor;
            if (_line.sharedMaterial != null)
            {
                _line.sharedMaterial.color = lineColor;
            }

            _line.startWidth = lineWidth;
            _line.endWidth = lineWidth;
            _line.startColor = lineColor;
            _line.endColor = lineColor;
            _line.positionCount = points.Count;
            for (int i = 0; i < points.Count; i++)
            {
                Vector3 p = points[i];
                p.y += yOffset;
                _line.SetPosition(i, p);
            }

            if (currentIndex >= 0 && currentIndex < points.Count)
            {
                Vector3 c = points[currentIndex];
                c.y += yOffset;
                _marker.position = c;
                _marker.gameObject.SetActive(true);
            }
            else
            {
                _marker.gameObject.SetActive(false);
            }
        }

        public static Material CreateUnlit(Color color)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
            {
                shader = Shader.Find("Unlit/Color");
            }

            if (shader == null)
            {
                shader = Shader.Find("Sprites/Default");
            }

            var mat = new Material(shader) { color = color };
            if (mat.HasProperty("_BaseColor"))
            {
                mat.SetColor("_BaseColor", color);
            }

            return mat;
        }
    }
}
