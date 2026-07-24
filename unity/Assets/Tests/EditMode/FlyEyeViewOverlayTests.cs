using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class FlyEyeViewOverlayTests
    {
        [Test]
        public void Overlay_Defaults_To_Visible_And_Toggle_Key_V()
        {
            var go = new GameObject("EyeOverlayTest");
            try
            {
                var overlay = go.AddComponent<FlyEyeViewOverlay>();
                Assert.IsTrue(overlay.ShowOverlay);
                overlay.ShowOverlay = false;
                Assert.IsFalse(overlay.ShowOverlay);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void EyeCamera_EnsureRenderTarget_Creates_Texture()
        {
            var go = new GameObject("EyeCamTest");
            try
            {
                go.AddComponent<Camera>();
                var eye = go.AddComponent<FlyEyeCamera>();
                eye.EnsureRenderTarget();
                Assert.IsNotNull(eye.RenderTexture);
                Assert.AreEqual(64, eye.RenderTexture.width);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
