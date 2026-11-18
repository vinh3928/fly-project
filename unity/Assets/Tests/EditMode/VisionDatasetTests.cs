using System.IO;
using System.Linq;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class EyeRayLabelsTests
    {
        private static float[] Filled(float value)
        {
            return Enumerable.Repeat(value, EyeRayLabels.Columns).ToArray();
        }

        [Test]
        public void Columns_SpanTheEyeFieldOfView()
        {
            Assert.AreEqual(-FlyEyeCamera.DefaultFieldOfView * 0.5f, EyeRayLabels.ColumnAngle(0), 1e-4f);
            Assert.AreEqual(FlyEyeCamera.DefaultFieldOfView * 0.5f, EyeRayLabels.ColumnAngle(EyeRayLabels.Columns - 1), 1e-4f);
        }

        [Test]
        public void AllOpen_IsFullStrengthStraightAhead_NoHazard()
        {
            EyeRayLabel label = EyeRayLabels.FromDistances(Filled(EyeRayLabels.OpenDistance));
            Assert.AreEqual(1f, label.OpeningStrength, 1e-5f);
            Assert.AreEqual(0f, label.OpeningBearing, 1e-5f);
            Assert.AreEqual(0f, label.WallHazard, 1e-5f);
            Assert.AreEqual(1f, label.ForwardBias, 1e-5f);
            Assert.AreEqual(0f, label.TurnBias, 1e-5f);
        }

        [Test]
        public void InsideTheBox_NoOpening()
        {
            EyeRayLabel label = EyeRayLabels.FromDistances(Filled(0.3f));
            Assert.AreEqual(0f, label.OpeningStrength, 1e-5f);
            Assert.AreEqual(0f, label.OpeningBearing, 1e-5f);
            Assert.AreEqual(0f, label.WallHazard, 1e-5f);
        }

        [Test]
        public void WallAhead_OpeningRight_TurnsRight()
        {
            float[] d = Filled(0.3f);
            for (int i = 0; i < d.Length; i++)
            {
                if (Mathf.Abs(EyeRayLabels.ColumnAngle(i)) <= EyeRayLabels.CentralHalfAngle)
                {
                    d[i] = EyeRayLabels.HazardRange * 0.5f;
                }
            }

            d[d.Length - 1] = EyeRayLabels.OpenDistance;
            d[d.Length - 2] = EyeRayLabels.OpenDistance;

            EyeRayLabel label = EyeRayLabels.FromDistances(d);
            Assert.AreEqual(0.5f, label.WallHazard, 1e-5f);
            Assert.AreEqual(0f, label.ForwardBias, 1e-5f);
            Assert.AreEqual(2f / d.Length, label.OpeningStrength, 1e-5f);
            Assert.Greater(label.OpeningBearing, 0.8f);
            Assert.Greater(label.TurnBias, 0.8f);
        }

        [Test]
        public void Compute_MeasuresWallLayerOnly()
        {
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.layer = ChamberWallHider.WallLayer;
            wall.transform.position = new Vector3(0f, 0f, 0.03f + 0.005f);
            wall.transform.localScale = new Vector3(0.04f, 0.1f, 0.01f);
            // Closer, wider, on the Default layer: rays must pass straight through it.
            var other = GameObject.CreatePrimitive(PrimitiveType.Cube);
            other.transform.position = new Vector3(0f, 0f, 0.015f);
            other.transform.localScale = new Vector3(1f, 0.1f, 0.01f);
            Physics.SyncTransforms();
            try
            {
                EyeRayLabel label = EyeRayLabels.Compute(Vector3.zero, Vector3.forward);
                int center = EyeRayLabels.Columns / 2;
                Assert.That(label.Distances[center], Is.InRange(0.029f, 0.04f));
                Assert.AreEqual(1f - label.Distances[center] / EyeRayLabels.HazardRange, label.WallHazard, 0.05f);
                Assert.AreEqual(EyeRayLabels.OpenDistance, label.Distances[0], 1e-5f, "narrow wall: edge rays see out");
                Assert.Greater(label.OpeningStrength, 0f);
            }
            finally
            {
                Object.DestroyImmediate(wall);
                Object.DestroyImmediate(other);
            }
        }
    }

    public sealed class VisionPoseSamplerTests
    {
        [Test]
        public void SameSeed_SamePoses()
        {
            var config = ExperimentConfiguration.Default();
            var a = new VisionPoseSampler(config, 7);
            var b = new VisionPoseSampler(config, 7);
            for (int i = 0; i < 50; i++)
            {
                VisionPose pa = a.Next();
                VisionPose pb = b.Next();
                Assert.AreEqual(pa.Region, pb.Region);
                Assert.AreEqual(pa.PositionXZ, pb.PositionXZ);
                Assert.AreEqual(pa.YawDegrees, pb.YawDegrees);
            }
        }

        [Test]
        public void Regions_StayInTheirAreas()
        {
            var config = ExperimentConfiguration.Default();
            Vector3 size = config.ChamberSize;
            float wall = ChamberController.WallThickness(size);
            float innerX = size.x * 0.5f - wall * 0.5f;
            float innerZ = size.z * 0.5f - wall * 0.5f;
            float exitZ = TunnelLayout.ExitZ(config);
            float halfOpening = TunnelLayout.OpeningWidth(config) * 0.5f;
            var sampler = new VisionPoseSampler(config, 1);
            var seen = new System.Collections.Generic.HashSet<VisionPoseRegion>();
            for (int i = 0; i < 500; i++)
            {
                VisionPose p = sampler.Next();
                seen.Add(p.Region);
                Vector2 xz = p.PositionXZ;
                switch (p.Region)
                {
                    case VisionPoseRegion.Tunnel:
                        Assert.LessOrEqual(Mathf.Abs(xz.x), halfOpening);
                        Assert.That(xz.y, Is.InRange(innerZ, exitZ));
                        break;
                    case VisionPoseRegion.Outside:
                        Assert.Greater(xz.y, exitZ);
                        break;
                    default:
                        Assert.LessOrEqual(Mathf.Abs(xz.x), innerX - VisionPoseSampler.WallMargin + 1e-5f);
                        Assert.LessOrEqual(Mathf.Abs(xz.y), innerZ - VisionPoseSampler.WallMargin + 1e-5f);
                        break;
                }
            }

            Assert.AreEqual(5, seen.Count, "every region sampled");
        }

        [Test]
        public void NoTunnel_StaysInside()
        {
            var config = ExperimentConfiguration.Default();
            config.TunnelEnabled = false;
            var sampler = new VisionPoseSampler(config, 3);
            for (int i = 0; i < 200; i++)
            {
                VisionPoseRegion region = sampler.Next().Region;
                Assert.That(region, Is.EqualTo(VisionPoseRegion.Interior).Or.EqualTo(VisionPoseRegion.NearWall));
            }
        }

        [Test]
        public void YawToward_PointsAlongZAndX()
        {
            Assert.AreEqual(0f, VisionPoseSampler.YawToward(Vector2.zero, Vector2.up), 1e-4f);
            Assert.AreEqual(90f, VisionPoseSampler.YawToward(Vector2.zero, Vector2.right), 1e-4f);
        }
    }

    public sealed class VisionDatasetWriterTests
    {
        private static string TempDir()
        {
            return Path.Combine(Path.GetTempPath(), "fly_vision_" + System.Guid.NewGuid().ToString("N"));
        }

        [Test]
        public void ToTopFirstRgb_FlipsRows()
        {
            var pixels = new Color32[4];
            pixels[0] = new Color32(10, 20, 30, 255); // bottom-left
            pixels[2] = new Color32(40, 50, 60, 255); // top-left
            var bytes = new byte[4 * 3];
            VisionDatasetWriter.ToTopFirstRgb(pixels, 2, 2, bytes);
            CollectionAssert.AreEqual(new byte[] { 40, 50, 60 }, bytes.Take(3).ToArray());
            CollectionAssert.AreEqual(new byte[] { 10, 20, 30 }, bytes.Skip(6).Take(3).ToArray());
        }

        [Test]
        public void Write_ProducesFramesLabelsAndMeta()
        {
            string dir = TempDir();
            try
            {
                var eye = new Color32[4 * 4];
                EyeRayLabel label = EyeRayLabels.FromDistances(Enumerable.Repeat(0.3f, EyeRayLabels.Columns).ToArray());
                using (var writer = new VisionDatasetWriter(dir, 4, 4))
                {
                    writer.Write(eye, eye, VisionPoseRegion.Interior, label);
                    writer.Write(eye, eye, VisionPoseRegion.Outside, label);
                }

                Assert.AreEqual(2 * 2 * 4 * 4 * 3, new FileInfo(Path.Combine(dir, VisionDatasetWriter.FramesFile)).Length);
                string[] lines = File.ReadAllLines(Path.Combine(dir, VisionDatasetWriter.LabelsFile));
                Assert.AreEqual(3, lines.Length);
                Assert.AreEqual(VisionDatasetWriter.LabelHeader, lines[0]);
                Assert.AreEqual(6 + EyeRayLabels.Columns, lines[1].Split(',').Length);
                StringAssert.StartsWith("Outside,", lines[2]);
                StringAssert.Contains("\"count\": 2", File.ReadAllText(Path.Combine(dir, VisionDatasetWriter.MetaFile)));
            }
            finally
            {
                if (Directory.Exists(dir))
                {
                    Directory.Delete(dir, true);
                }
            }
        }

        [Test]
        public void Write_RejectsWrongFrameSize()
        {
            string dir = TempDir();
            try
            {
                EyeRayLabel label = EyeRayLabels.FromDistances(Enumerable.Repeat(0.3f, EyeRayLabels.Columns).ToArray());
                using var writer = new VisionDatasetWriter(dir, 4, 4);
                Assert.Throws<System.ArgumentException>(
                    () => writer.Write(new Color32[3], new Color32[16], VisionPoseRegion.Interior, label));
            }
            finally
            {
                if (Directory.Exists(dir))
                {
                    Directory.Delete(dir, true);
                }
            }
        }
    }

    public sealed class VisionDatasetRecorderTests
    {
        private static FlyEyeCamera MakeEye(Transform parent, bool isLeft)
        {
            var go = new GameObject(isLeft ? "LeftEye" : "RightEye");
            go.transform.SetParent(parent, false);
            go.transform.localPosition = FlyBodyController.HeadCenter + FlyEyeCamera.MountOffset(isLeft);
            var cam = go.AddComponent<Camera>();
            cam.enabled = false;
            var eye = go.AddComponent<FlyEyeCamera>();
            Phase2TestHelpers.SetPrivate(eye, "isLeftEye", isLeft);
            eye.Configure(8, 8, FlyEyeCamera.DefaultFieldOfView, false);
            return eye;
        }

        [Test]
        public void Record_WritesSamples_AndRestoresPose()
        {
            using var fixture = new LocomotionFixture();
            var chamberGo = new GameObject("Chamber");
            var chamber = chamberGo.AddComponent<ChamberController>();
            chamber.Configuration = ExperimentConfiguration.Default();
            FlyEyeCamera left = MakeEye(fixture.Fly.transform, true);
            FlyEyeCamera right = MakeEye(fixture.Fly.transform, false);
            Vector3 position = fixture.Fly.transform.position;
            Quaternion rotation = fixture.Fly.transform.rotation;
            string dir = Path.Combine(Path.GetTempPath(), "fly_vision_" + System.Guid.NewGuid().ToString("N"));
            try
            {
                int written = VisionDatasetRecorder.Record(fixture.Body, chamber, left, right, dir, 5, 11);

                Assert.AreEqual(5, written);
                Assert.AreEqual(5 * 2 * 8 * 8 * 3, new FileInfo(Path.Combine(dir, VisionDatasetWriter.FramesFile)).Length);
                Assert.AreEqual(position, fixture.Fly.transform.position);
                Assert.AreEqual(rotation, fixture.Fly.transform.rotation);
            }
            finally
            {
                Object.DestroyImmediate(chamberGo);
                if (Directory.Exists(dir))
                {
                    Directory.Delete(dir, true);
                }
            }
        }
    }
}
