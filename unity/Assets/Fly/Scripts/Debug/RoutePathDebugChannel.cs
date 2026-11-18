#if UNITY_MLAGENTS
using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// ML-Agents SideChannel mirroring Python <c>RoutePathDebugChannel</c>.
    /// Receives route polylines (v1) and place-graph maps (v2); sends fly pose.
    /// Guid must match <c>ROUTE_PATH_CHANNEL_ID</c> in Python.
    /// </summary>
    public sealed class RoutePathDebugChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("c8f0a2b4-5d6e-4a1b-9c3d-7e8f90123456");

        public const int ModeOff = 0;
        public const int ModeRoute = 1;
        public const int ModeScan = 2;

        public const int LabelUnknown = 0;
        public const int LabelPlace = 1;
        public const int LabelPuddle = 2;
        public const int LabelOutside = 3;
        public const int LabelOdorPlume = 4;
        public const int LabelWall = 5;

        public const int FlagCurrent = 1;
        public const int FlagHome = 2;
        public const int FlagRoute = 4;

        public readonly struct GraphNode
        {
            public GraphNode(int label, int flags, Vector3 position, float smell)
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

        /// <summary>Shared instance so path + graph visualizers use one channel.</summary>
        public static RoutePathDebugChannel Active { get; private set; }

        public int Mode { get; private set; }
        public int CurrentIndex { get; private set; } = -1;
        public IReadOnlyList<Vector3> Points => _points;
        public IReadOnlyList<GraphNode> GraphNodes => _graphNodes;
        public IReadOnlyList<Vector2Int> GraphEdges => _graphEdges;

        public event Action PathUpdated;
        public event Action GraphUpdated;

        private readonly List<Vector3> _points = new List<Vector3>(64);
        private readonly List<GraphNode> _graphNodes = new List<GraphNode>(64);
        private readonly List<Vector2Int> _graphEdges = new List<Vector2Int>(96);
        private Transform _fly;

        public RoutePathDebugChannel()
        {
            ChannelId = ChannelGuid;
        }

        public void BindFly(Transform flyRoot)
        {
            _fly = flyRoot;
        }

        public void SendFlyPose()
        {
            if (_fly == null)
            {
                return;
            }

            // [x, y, z, yawDegrees] — yaw matches transform.eulerAngles.y (Unity forward=+Z).
            Vector3 p = _fly.position;
            float yaw = _fly.eulerAngles.y;
            using var msg = new OutgoingMessage();
            msg.WriteFloatList(new[] { p.x, p.y, p.z, yaw });
            QueueMessageToSend(msg);
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            IList<float> data = msg.ReadFloatList();
            if (data == null || data.Count < 3)
            {
                Mode = ModeOff;
                CurrentIndex = -1;
                _points.Clear();
                _graphNodes.Clear();
                _graphEdges.Clear();
                PathUpdated?.Invoke();
                GraphUpdated?.Invoke();
                return;
            }

            int version = Mathf.RoundToInt(data[0]);
            if (version >= 2)
            {
                ParseGraph(data);
                GraphUpdated?.Invoke();
                return;
            }

            ParsePath(data);
            PathUpdated?.Invoke();
        }

        private void ParsePath(IList<float> data)
        {
            if (data.Count < 4)
            {
                Mode = ModeOff;
                CurrentIndex = -1;
                _points.Clear();
                return;
            }

            Mode = Mathf.RoundToInt(data[1]);
            CurrentIndex = Mathf.RoundToInt(data[2]);
            int n = Mathf.Max(0, Mathf.RoundToInt(data[3]));
            _points.Clear();
            int offset = 4;
            for (int i = 0; i < n && offset + 2 < data.Count; i++)
            {
                _points.Add(new Vector3(data[offset], data[offset + 1], data[offset + 2]));
                offset += 3;
            }

            if (CurrentIndex >= _points.Count)
            {
                CurrentIndex = _points.Count - 1;
            }
        }

        private void ParseGraph(IList<float> data)
        {
            int nNodes = Mathf.Max(0, Mathf.RoundToInt(data[1]));
            int nEdges = Mathf.Max(0, Mathf.RoundToInt(data[2]));
            _graphNodes.Clear();
            _graphEdges.Clear();
            int offset = 3;
            for (int i = 0; i < nNodes && offset + 5 < data.Count; i++)
            {
                int label = Mathf.RoundToInt(data[offset]);
                int flags = Mathf.RoundToInt(data[offset + 1]);
                var pos = new Vector3(data[offset + 2], data[offset + 3], data[offset + 4]);
                float smell = data[offset + 5];
                _graphNodes.Add(new GraphNode(label, flags, pos, smell));
                offset += 6;
            }

            for (int i = 0; i < nEdges && offset + 1 < data.Count; i++)
            {
                int a = Mathf.RoundToInt(data[offset]);
                int b = Mathf.RoundToInt(data[offset + 1]);
                _graphEdges.Add(new Vector2Int(a, b));
                offset += 2;
            }
        }

        public static RoutePathDebugChannel Register(Transform flyRoot)
        {
            if (Active != null)
            {
                Active.BindFly(flyRoot);
                return Active;
            }

            var channel = new RoutePathDebugChannel();
            channel.BindFly(flyRoot);
            SideChannelManager.RegisterSideChannel(channel);
            Active = channel;
            return channel;
        }

        public void Unregister()
        {
            if (Academy.IsInitialized)
            {
                SideChannelManager.UnregisterSideChannel(this);
            }

            if (ReferenceEquals(Active, this))
            {
                Active = null;
            }
        }
    }
}
#endif
