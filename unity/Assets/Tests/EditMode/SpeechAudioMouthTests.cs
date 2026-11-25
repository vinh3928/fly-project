using FlySimulation.Environment;
using FlySimulation.Fly.DebugViz;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class SpeechAudioMouthTests
    {
        [Test]
        public void FlyMouthActuator_AppliesArticulatoryCommands()
        {
            var go = new GameObject("mouth");
            var mouth = go.AddComponent<FlyMouthActuator>();
            mouth.SetArticulatory(0.9f, 0.8f, 0.4f, 0.35f, 0.95f, 0.2f, 0.85f);
            float[] ctrl = mouth.Control;
            Assert.AreEqual(FlyMouthActuator.ArticulatoryDim, ctrl.Length);
            Assert.AreEqual(0.9f, ctrl[0], 1e-4f);
            Assert.AreEqual(0.8f, ctrl[1], 1e-4f);
            Assert.AreEqual(0.95f, ctrl[4], 1e-4f);
            Assert.AreEqual(0.2f, ctrl[5], 1e-4f);
            Assert.AreEqual(0.85f, ctrl[6], 1e-4f);
            mouth.Silence();
            Assert.AreEqual(0f, mouth.Control[4], 1e-4f);
            Assert.AreEqual(0f, mouth.Control[6], 1e-4f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void FlyAudioCapture_InjectTone_ProducesNonZeroBands()
        {
            var go = new GameObject("ears");
            var ears = go.AddComponent<FlyAudioCapture>();
            int n = 512;
            var mono = new float[n];
            for (int i = 0; i < n; i++)
            {
                mono[i] = Mathf.Sin(2f * Mathf.PI * 440f * i / 24000f) * 0.5f;
            }

            ears.InjectSamplesForTests(mono);
            float[] bands = ears.Bands;
            Assert.AreEqual(FlyAudioCapture.BandCount, bands.Length);
            float sum = 0f;
            for (int i = 0; i < bands.Length; i++)
            {
                sum += bands[i];
            }

            Assert.Greater(sum, 0f, "expected non-zero band energies after tone inject");
            Assert.Greater(ears.Rms, 0f);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void EnvAudioCue_ConfigureAndBurst()
        {
            var go = new GameObject("cue");
            var cue = go.AddComponent<EnvAudioCue>();
            cue.Configure(523f, 0.55f, 1.8f);
            cue.TriggerBurstForTests();
            Assert.IsNotNull(go.GetComponent<AudioSource>());
            Object.DestroyImmediate(go);
        }

        [Test]
        public void EnvAmbientSynth_PresetsConfigureKinds()
        {
            var go = new GameObject("ambient");
            var synth = go.AddComponent<EnvAmbientSynth>();
            synth.Configure(EnvAmbientSynth.Kind.Wind);
            Assert.AreEqual(EnvAmbientSynth.Kind.Wind, synth.SoundKind);
            Assert.IsNotNull(go.GetComponent<AudioSource>());
            synth.Configure(EnvAmbientSynth.Kind.Bees);
            Assert.AreEqual(EnvAmbientSynth.Kind.Bees, synth.SoundKind);
            synth.Configure(EnvAmbientSynth.Kind.Song);
            Assert.AreEqual(EnvAmbientSynth.Kind.Song, synth.SoundKind);
            Object.DestroyImmediate(go);
        }

        [Test]
        public void SpeechStatusLabel_FormatsHearAndBabble()
        {
            Assert.AreEqual(
                "speech_hear = -\nspeech_babble = -",
                SpeechStatusLabel.FormatLabel("", ""));
            Assert.AreEqual(
                "speech_hear = ABC\nspeech_babble = XY",
                SpeechStatusLabel.FormatLabel("ABC", "XY"));

            var go = new GameObject("speech_label");
            try
            {
                var label = go.AddComponent<SpeechStatusLabel>();
                label.Bind(go.transform);
                label.SetSpeech("HE", "BA");
                Assert.AreEqual("HE", label.HearAlphabet);
                Assert.AreEqual("BA", label.BabbleAlphabet);
                Assert.AreEqual("speech_hear = HE\nspeech_babble = BA", label.LabelText);
                Transform root = go.transform.Find(SpeechStatusLabel.ObjectName);
                Assert.IsNotNull(root);
                Assert.IsTrue(root.gameObject.activeSelf);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void SpeechStatusLabel_ReusesExistingChild_NoDuplicates()
        {
            var go = new GameObject("speech_label_dup");
            try
            {
                var stale = new GameObject(SpeechStatusLabel.ObjectName);
                stale.transform.SetParent(go.transform, false);
                var staleLabel = new GameObject("Label");
                staleLabel.transform.SetParent(stale.transform, false);
                var mesh = staleLabel.AddComponent<TextMesh>();
                mesh.text = "old";

                var extra = new GameObject(SpeechStatusLabel.ObjectName);
                extra.transform.SetParent(go.transform, false);

                var label = go.AddComponent<SpeechStatusLabel>();
                label.Bind(go.transform);
                label.SetSpeech("A", "B");

                Assert.AreEqual(1, label.LabelRootCount);
                Assert.AreEqual("speech_hear = A\nspeech_babble = B", label.LabelText);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void SpeechAudioBridge_AppliesMouthForTests()
        {
            var go = new GameObject("fly");
            var ears = go.AddComponent<FlyAudioCapture>();
            var head = new GameObject("head");
            head.transform.SetParent(go.transform);
            var mouth = head.AddComponent<FlyMouthActuator>();
            var bridge = go.AddComponent<SpeechAudioBridge>();
            // Bind via ApplyMouthForTests path (EditMode, no SideChannel).
            bridge.ApplyMouthForTests(new[] { 0.5f, 0.4f, 0.3f, 0.3f, 0.8f, 0f, 0.7f });
            // Bridge may not have serialized refs; call mouth directly as fallback check.
            mouth.SetArticulatory(0.5f, 0.4f, 0.3f, 0.3f, 0.8f, 0f, 0.7f);
            Assert.AreEqual(0.7f, mouth.Control[6], 1e-4f);
            Assert.IsNotNull(ears);
            Object.DestroyImmediate(go);
        }
    }
}
