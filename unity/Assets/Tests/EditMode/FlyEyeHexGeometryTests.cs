using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class FlyEyeHexGeometryTests
    {
        [Test]
        public void Corners_AreOutsideHexagon()
        {
            Assert.IsFalse(FlyEyeHexGeometry.InsideHexagon(0f, 0f, 64, 64));
            Assert.IsFalse(FlyEyeHexGeometry.InsideHexagon(63f, 0f, 64, 64));
            Assert.IsFalse(FlyEyeHexGeometry.InsideHexagon(0f, 63f, 64, 64));
            Assert.IsTrue(FlyEyeHexGeometry.InsideHexagon(32f, 32f, 64, 64));
        }

        [Test]
        public void ApertureMask_ZerosCornerPixels()
        {
            const int w = 16;
            const int h = 16;
            var pixels = new Color32[w * h];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = new Color32(255, 128, 64, 255);
            }

            FlyEyeHexGeometry.ApplyApertureMask(pixels, w, h);
            Assert.AreEqual(0, pixels[0].r);
            Assert.AreEqual(0, pixels[0].g);
            Assert.AreEqual(0, pixels[0].b);
            int mid = (h / 2) * w + (w / 2);
            Assert.AreEqual(255, pixels[mid].r);
        }

        [Test]
        public void Vertices_PointyTopHasTopVertex()
        {
            Vector2[] verts = FlyEyeHexGeometry.VerticesPx(64, 64);
            Assert.AreEqual(6, verts.Length);
            // Vertex 0 is top-most (smallest y in image space).
            float minY = verts[0].y;
            for (int i = 1; i < verts.Length; i++)
            {
                Assert.GreaterOrEqual(verts[i].y, minY - 1e-3f);
            }
        }

        [Test]
        public void DrawFacetGuides_PaintsCenterPixel()
        {
            const int w = 32;
            const int h = 32;
            var pixels = new Color32[w * h];
            FlyEyeHexGeometry.ApplyApertureMask(pixels, w, h);
            var guide = new Color32(10, 200, 50, 255);
            FlyEyeHexGeometry.DrawFacetGuides(pixels, w, h, guide);
            int mid = (h / 2) * w + (w / 2);
            Assert.AreEqual(guide.r, pixels[mid].r);
            Assert.AreEqual(guide.g, pixels[mid].g);
        }
    }
}
