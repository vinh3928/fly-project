#if UNITY_MLAGENTS
using System;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// Bidirectional SideChannel for Father-advocate ask/reply dialogs.
    /// Guid must match <c>FATHER_DIALOG_CHANNEL_ID</c> in Python.
    /// </summary>
    public sealed class FatherDialogDebugChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("e5f6a7b8-9c0d-4e1f-2a3b-4c5d6e7f8091");

        public const int ProtocolVersion = 1;
        public const int KindAsk = 0;
        public const int KindReply = 1;
        public const int KindHide = 2;

        public const int SourceHuman = 0;
        public const int SourceAskLlm = 1;
        public const int SourceSkip = 2;

        public static FatherDialogDebugChannel Active { get; private set; }

        public int LastRequestId { get; private set; }
        public int LastStep { get; private set; }
        public string LastCue { get; private set; } = "";
        public string LastQuestion { get; private set; } = "";
        public bool AllowLlm { get; private set; } = true;
        public bool IsVisible { get; private set; }

        public event Action AskReceived;
        public event Action HideReceived;

        public FatherDialogDebugChannel()
        {
            ChannelId = ChannelGuid;
        }

        public static FatherDialogDebugChannel Register()
        {
            if (Active != null)
            {
                return Active;
            }

            var channel = new FatherDialogDebugChannel();
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

        /// <summary>EditMode / no Academy: apply an ask locally.</summary>
        public void ApplyAsk(int requestId, int step, bool allowLlm, string cue, string question)
        {
            LastRequestId = requestId;
            LastStep = step;
            AllowLlm = allowLlm;
            LastCue = cue ?? "";
            LastQuestion = question ?? "";
            IsVisible = true;
            AskReceived?.Invoke();
        }

        public void ApplyHide(int requestId)
        {
            if (requestId == LastRequestId || requestId == 0)
            {
                IsVisible = false;
                HideReceived?.Invoke();
            }
        }

        public void SendReply(int requestId, int source, string answer)
        {
            using var msg = new OutgoingMessage();
            msg.WriteInt32(ProtocolVersion);
            msg.WriteInt32(KindReply);
            msg.WriteInt32(requestId);
            msg.WriteInt32(source);
            msg.WriteString(answer ?? "");
            QueueMessageToSend(msg);
            IsVisible = false;
            HideReceived?.Invoke();
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

            if (kind != KindAsk)
            {
                return;
            }

            int step = msg.ReadInt32();
            int allowLlm = msg.ReadInt32();
            string cue = msg.ReadString();
            string question = msg.ReadString();
            ApplyAsk(requestId, step, allowLlm != 0, cue, question);
        }
    }
}
#endif
