using FlySimulation.Environment;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class MindLogDisplayTests
    {
        [Test]
        public void Place_TargetsDisplay3()
        {
            var go = new GameObject(MindLogDisplay.ObjectName);
            try
            {
                var cam = go.AddComponent<Camera>();
                MindLogDisplay.Place(cam);
                Assert.AreEqual(MindLogDisplay.TargetDisplayIndex, cam.targetDisplay);
                Assert.IsTrue(cam.enabled);
                Assert.AreEqual(0, cam.cullingMask);
                Assert.AreEqual("Untagged", cam.tag);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void ApplyEntry_ShowsHumanReadableText()
        {
            var go = new GameObject(MindLogDisplay.ObjectName);
            try
            {
                go.AddComponent<Camera>();
                var display = go.AddComponent<MindLogDisplay>();
                display.ApplyEntry(
                    kind: 0,
                    step: 12,
                    text: "[MIND 12]\nplan=feed\nthought: go home");

                Assert.IsNotNull(display.Body);
                Assert.IsTrue(display.Body.text.Contains("[MIND 12]"));
                Assert.IsTrue(display.Body.text.Contains("thought: go home"));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void IsLogCamera_DetectsComponent()
        {
            var go = new GameObject("Other");
            try
            {
                var cam = go.AddComponent<Camera>();
                Assert.IsFalse(MindLogDisplay.IsLogCamera(cam));
                go.AddComponent<MindLogDisplay>();
                Assert.IsTrue(MindLogDisplay.IsLogCamera(cam));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
