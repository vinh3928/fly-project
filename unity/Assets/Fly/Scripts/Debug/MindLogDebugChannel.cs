#if UNITY_MLAGENTS
using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// ML-Agents SideChannel for human-readable mind / deep-thought logs.
    /// Guid must match <c>MIND_LOG_CHANNEL_ID</c> in Python.
    /// </summary>
    public sealed class MindLogDebugChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("d4e5f6a7-8b9c-4d0e-1f2a-3b4c5d6e7f80");

        public const int ProtocolVersion = 1;
        public const int KindMind = 0;
        public const int KindDeep = 1;

        public static MindLogDebugChannel Active { get; private set; }

        public int LastKind { get; private set; }
        public int LastStep { get; private set; }
        public string LastText { get; private set; } = "";
        public IReadOnlyList<string> Entries => _entries;

        public event Action LogUpdated;

        private readonly List<string> _entries = new List<string>(16);
        private int _maxEntries = 12;

        public MindLogDebugChannel(int maxEntries = 12)
        {
            ChannelId = ChannelGuid;
            _maxEntries = Mathf.Max(1, maxEntries);
        }

        public static MindLogDebugChannel Register(int maxEntries = 12)
        {
            if (Active != null)
            {
                return Active;
            }

            var channel = new MindLogDebugChannel(maxEntries);
            SideChannelManager.RegisterSideChannel(channel);
            Active = channel;
            return channel;
        }

        public void Unregister()
        {
            if (Active == this)
            {
                Active = null;
            }

            SideChannelManager.UnregisterSideChannel(this);
        }

        /// <summary>Apply a log entry without SideChannel (EditMode tests).</summary>
        public void ApplyEntry(int kind, int step, string text)
        {
            LastKind = kind;
            LastStep = step;
            LastText = text ?? "";
            if (string.IsNullOrEmpty(LastText))
            {
                _entries.Clear();
            }
            else
            {
                _entries.Insert(0, LastText);
                while (_entries.Count > _maxEntries)
                {
                    _entries.RemoveAt(_entries.Count - 1);
                }
            }

            LogUpdated?.Invoke();
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            int version = msg.ReadInt32();
            if (version < 1)
            {
                return;
            }

            int kind = msg.ReadInt32();
            int step = msg.ReadInt32();
            string text = msg.ReadString("") ?? "";
            ApplyEntry(kind, step, text);
        }
    }
}
#endif
