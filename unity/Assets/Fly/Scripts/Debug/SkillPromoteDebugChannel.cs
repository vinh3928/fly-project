#if UNITY_MLAGENTS
using System;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// Bidirectional SideChannel for Display 4 skill-promote queue.
    /// Guid must match <c>SKILL_PROMOTE_CHANNEL_ID</c> in Python.
    /// </summary>
    public sealed class SkillPromoteDebugChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("f6a7b8c9-0d1e-4f2a-3b4c-5d6e7f8091a2");

        public const int ProtocolVersion = 1;
        public const int KindShow = 0;
        public const int KindReply = 1;
        public const int KindHide = 2;
        public const int KindView = 3;

        public const int SourceAccept = 0;
        public const int SourceAskLlm = 1;
        public const int SourceRefuse = 2;
        public const int SourceTest = 3;

        public static SkillPromoteDebugChannel Active { get; private set; }

        public int LastRequestId { get; private set; }
        public int LastStep { get; private set; }
        public int QueueCount { get; private set; }
        public bool AllowLlm { get; private set; } = true;
        public string SkillId { get; private set; } = "";
        public string SkillName { get; private set; } = "";
        public string Tags { get; private set; } = "";
        public string Score { get; private set; } = "0";
        public string StepsCount { get; private set; } = "0";
        public string SuggestedName { get; private set; } = "";
        public bool IsVisible { get; private set; }

        public event Action ShowReceived;
        public event Action HideReceived;

        public SkillPromoteDebugChannel()
        {
            ChannelId = ChannelGuid;
        }

        public static SkillPromoteDebugChannel Register()
        {
            if (Active != null)
            {
                return Active;
            }

            var channel = new SkillPromoteDebugChannel();
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

        public void ApplyShow(
            int requestId,
            int step,
            int queueCount,
            bool allowLlm,
            string skillId,
            string name,
            string tags,
            string score,
            string stepsCount,
            string suggestedName)
        {
            LastRequestId = requestId;
            LastStep = step;
            QueueCount = Mathf.Max(0, queueCount);
            AllowLlm = allowLlm;
            SkillId = skillId ?? "";
            SkillName = name ?? "";
            Tags = tags ?? "";
            Score = score ?? "0";
            StepsCount = stepsCount ?? "0";
            SuggestedName = suggestedName ?? "";
            IsVisible = true;
            ShowReceived?.Invoke();
        }

        public void ApplyHide(int requestId)
        {
            if (requestId == LastRequestId || requestId == 0)
            {
                IsVisible = false;
                HideReceived?.Invoke();
            }
        }

        public void SendReply(int requestId, int source, string name)
        {
            using var msg = new OutgoingMessage();
            msg.WriteInt32(ProtocolVersion);
            msg.WriteInt32(KindReply);
            msg.WriteInt32(requestId);
            msg.WriteInt32(source);
            msg.WriteString(name ?? "");
            QueueMessageToSend(msg);
        }

        /// <summary>Tell Python whether the user is focused on Display 4.</summary>
        public void SendViewing(int requestId, bool viewing)
        {
            using var msg = new OutgoingMessage();
            msg.WriteInt32(ProtocolVersion);
            msg.WriteInt32(KindView);
            msg.WriteInt32(requestId);
            msg.WriteInt32(viewing ? 1 : 0);
            QueueMessageToSend(msg);
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            int version = msg.ReadInt32();
            if (version != ProtocolVersion)
            {
                return;
            }

            int kind = msg.ReadInt32();
            int requestId = msg.ReadInt32();
            if (kind == KindHide)
            {
                ApplyHide(requestId);
                return;
            }

            if (kind != KindShow)
            {
                return;
            }

            int step = msg.ReadInt32();
            int queueCount = msg.ReadInt32();
            int allowLlm = msg.ReadInt32();
            string skillId = msg.ReadString();
            string name = msg.ReadString();
            string tags = msg.ReadString();
            string score = msg.ReadString();
            string stepsCount = msg.ReadString();
            string suggestedName = msg.ReadString();
            ApplyShow(
                requestId,
                step,
                queueCount,
                allowLlm != 0,
                skillId,
                name,
                tags,
                score,
                stepsCount,
                suggestedName);
        }
    }
}
#endif
