#if UNITY_MLAGENTS
using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Sensors;

namespace FlySimulation.Fly.DebugViz
{
    /// <summary>
    /// Ear bands → Python; articulatory mouth + hear/babble status ← Python.
    /// Guid must match <c>SPEECH_CHANNEL_ID</c> in Python.
    /// </summary>
    public sealed class SpeechDebugChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("b7c8d9e0-1f2a-4b3c-9d4e-5f6a7b8c9d0e");

        public const int KindAudio = 0;
        public const int KindMouth = 1;
        public const int KindStatus = 2;

        public static SpeechDebugChannel Active { get; private set; }

        public string LastHearAlphabet { get; private set; } = "";
        public string LastBabbleAlphabet { get; private set; } = "";

        public event Action StatusUpdated;

        private FlyAudioCapture _ears;
        private FlyMouthActuator _mouth;

        public SpeechDebugChannel()
        {
            ChannelId = ChannelGuid;
        }

        public static SpeechDebugChannel Register(FlyAudioCapture ears, FlyMouthActuator mouth)
        {
            if (Active != null)
            {
                Active.Bind(ears, mouth);
                return Active;
            }

            var channel = new SpeechDebugChannel();
            channel.Bind(ears, mouth);
            SideChannelManager.RegisterSideChannel(channel);
            Active = channel;
            return channel;
        }

        public void Bind(FlyAudioCapture ears, FlyMouthActuator mouth)
        {
            _ears = ears;
            _mouth = mouth;
        }

        public void Unregister()
        {
            if (Active == this)
            {
                Active = null;
            }

            SideChannelManager.UnregisterSideChannel(this);
        }

        /// <summary>Send latest ear feature vector to Python.</summary>
        public void SendAudioFeatures()
        {
            if (_ears == null)
            {
                return;
            }

            float[] bands = _ears.Bands;
            var payload = new List<float>(1 + FlyAudioCapture.BandCount + 2) { KindAudio };
            for (int i = 0; i < FlyAudioCapture.BandCount; i++)
            {
                payload.Add(i < bands.Length ? bands[i] : 0f);
            }

            payload.Add(_ears.Rms);
            payload.Add(_ears.Voiced);
            using var msg = new OutgoingMessage();
            msg.WriteFloatList(payload);
            QueueMessageToSend(msg);
        }

        /// <summary>EditMode: apply status without a SideChannel wire.</summary>
        public void ApplyStatusForTests(string hear, string babble)
        {
            LastHearAlphabet = hear ?? "";
            LastBabbleAlphabet = babble ?? "";
            StatusUpdated?.Invoke();
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            int kind = msg.ReadInt32();
            if (kind == KindMouth)
            {
                if (_mouth == null)
                {
                    return;
                }

                var ctrl = new float[FlyMouthActuator.ArticulatoryDim];
                for (int i = 0; i < ctrl.Length; i++)
                {
                    ctrl[i] = msg.ReadFloat32();
                }

                _mouth.SetArticulatory(ctrl);
                return;
            }

            if (kind == KindStatus)
            {
                LastHearAlphabet = msg.ReadString() ?? "";
                LastBabbleAlphabet = msg.ReadString() ?? "";
                StatusUpdated?.Invoke();
            }
        }
    }
}
#endif
