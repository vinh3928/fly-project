using System.Collections.Generic;
using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// Draws a single place-graph label (current/home). Edges are drawn by
    /// <see cref="RoutePathVisualizer"/> as one thin polyline — not N LineRenderers.
    /// Culled from eye cameras via the Fly layer.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class PlaceGraphVisualizer : MonoBehaviour
    {
        public const float DefaultYLift = 0.03f;

        public const int LabelUnknown = 0;
        public const int LabelPlace = 1;
        public const int LabelPuddle = 2;
        public const int LabelOutside = 3;
        public const int LabelOdorPlume = 4;
        public const int LabelWall = 5;

        public const int FlagCurrent = 1;
        public const int FlagHome = 2;
        public const int FlagRoute = 4;

        public readonly struct NodeSnap
        {
            public NodeSnap(int label, int flags, Vector3 position, float smell)
            {
                Label = label;
                Flags = flags;
                Position = position;
                Smell = smell;
            }

            public int Label { get; }
            public int Flags { get; }
            public Vector3 Position { get; }
            public float Smell { get; }
            public bool IsCurrent => (Flags & FlagCurrent) != 0;
            public bool IsHome => (Flags & FlagHome) != 0;
            public bool OnRoute => (Flags & FlagRoute) != 0;
        }

        [SerializeField] private Transform flyRoot;
        [SerializeField] private float yLift = DefaultYLift;
        [SerializeField] private bool showLabels = true;

        private GameObject _labelRoot;
        private TextMesh _label;
        private bool _dirty = true;

#if UNITY_MLAGENTS
        private RoutePathDebugChannel _channel;
#endif

        public int ActiveNodeCount { get; private set; }
        public int ActiveEdgeCount { get; private set; }

        public void Bind(Transform fly)
        {
            flyRoot = fly;
#if UNITY_MLAGENTS
            EnsureChannel();
#endif
            EnsureLabel();
        }

        private void Awake()
        {
            EnsureLabel();
        }

        private void OnEnable()
        {
            if (flyRoot == null)
            {
                var body = Object.FindFirstObjectByType<FlySimulation.Fly.Body.FlyBodyController>();
                if (body != null)
                {
                    flyRoot = body.transform;
                }
            }

#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.GraphUpdated += OnGraphUpdated;
            }
#endif
            _dirty = true;
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (_channel != null)
            {
                _channel.GraphUpdated -= OnGraphUpdated;
                _channel = null;
            }
#endif
        }

        private void OnDestroy()
        {
            if (_labelRoot != null)
            {
                Object.DestroyImmediate(_labelRoot);
                _labelRoot = null;
                _label = null;
            }
        }

        private void LateUpdate()
        {
#if UNITY_MLAGENTS
            // Pose is sent by RoutePathVisualizer.FixedUpdate — do not double-queue.
            if (_dirty)
            {
                RebuildFromChannel();
                _dirty = false;
            }
#endif
        }

#if UNITY_MLAGENTS
        private void OnGraphUpdated()
        {
            _dirty = true;
        }

        private void EnsureChannel()
        {
            if (_channel != null)
            {
                return;
            }

            Transform root = flyRoot != null ? flyRoot : transform;
            _channel = RoutePathDebugChannel.Register(root);
        }

        private void RebuildFromChannel()
        {
            if (_channel == null)
            {
                return;
            }

            var snaps = new List<NodeSnap>(_channel.GraphNodes.Count);
            foreach (var n in _channel.GraphNodes)
            {
                snaps.Add(new NodeSnap(n.Label, n.Flags, n.Position, n.Smell));
            }

            ApplySnapshot(snaps, _channel.GraphEdges);
        }
#endif

        /// <summary>Apply a snapshot without SideChannel (EditMode tests).</summary>
        public void ApplySnapshot(
            IReadOnlyList<NodeSnap> nodes,
            IReadOnlyList<Vector2Int> edges)
        {
            EnsureLabel();
            ActiveEdgeCount = edges?.Count ?? 0;
            ActiveNodeCount = 0;

            int nCount = nodes?.Count ?? 0;
            int labelIndex = -1;
            if (showLabels && nodes != null)
            {
                for (int i = 0; i < nCount; i++)
                {
                    if (nodes[i].IsCurrent)
                    {
                        labelIndex = i;
                        break;
                    }
                }

                if (labelIndex < 0)
                {
                    for (int i = 0; i < nCount; i++)
                    {
                        if (nodes[i].IsHome)
                        {
                            labelIndex = i;
                            break;
                        }
                    }
                }
            }

            if (labelIndex < 0 || nodes == null)
            {
                _labelRoot.SetActive(false);
                return;
            }

            var node = nodes[labelIndex];
            ActiveNodeCount = 1;
            _labelRoot.SetActive(true);
            _labelRoot.transform.position = node.Position + Vector3.up * yLift;
            _label.text = LabelText(node);
        }

        private static string LabelText(NodeSnap node)
        {
            string kind;
            switch (node.Label)
            {
                case LabelPuddle:
                    kind = "puddle";
                    break;
                case LabelOutside:
                    kind = "out";
                    break;
                case LabelOdorPlume:
                    kind = "plume";
                    break;
                case LabelWall:
                    kind = "wall";
                    break;
                default:
                    kind = "place";
                    break;
            }

            string flags = "";
            if (node.IsCurrent)
            {
                flags += "*";
            }

            if (node.IsHome)
            {
                flags += "H";
            }

            if (node.OnRoute)
            {
                flags += "R";
            }

            if (node.Smell > 0.05f)
            {
                return $"{kind}{flags} s={node.Smell:0.00}";
            }

            return $"{kind}{flags}";
        }

        private void EnsureLabel()
        {
            if (_labelRoot != null && _label != null)
            {
                ApplyDebugLayer(_labelRoot);
                return;
            }

            _labelRoot = new GameObject("PlaceGraphLabel");
            _labelRoot.transform.SetParent(transform, false);
            ApplyDebugLayer(_labelRoot);
            var labelGo = new GameObject("Label");
            labelGo.transform.SetParent(_labelRoot.transform, false);
            ApplyDebugLayer(labelGo);
            _label = labelGo.AddComponent<TextMesh>();
            _label.characterSize = 0.035f;
            _label.fontSize = 48;
            _label.anchor = TextAnchor.MiddleCenter;
            _label.alignment = TextAlignment.Center;
            _label.color = Color.white;
            _labelRoot.SetActive(false);
        }

        private void ApplyDebugLayer(GameObject go)
        {
            // Eyes cull the Fly layer — keep debug text out of eye RTs (big win).
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
