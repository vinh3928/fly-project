using FlySimulation.Environment;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class BodyClearanceTests
    {
        private const float HeadFrontZ = 0.01f + FlyBodyController.HeadRadius;

        private static GameObject MakeWall(Vector3 center, Vector3 size, int layer = ChamberWallHider.WallLayer)
        {
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = "Wall_test";
            wall.layer = layer;
            wall.transform.position = center;
            wall.transform.localScale = size;
            Physics.SyncTransforms();
            return wall;
        }

        /// <summary>Fly at y=0.02 facing +Z; wall face (near side) at z = <paramref name="faceZ"/>.</summary>
        private static GameObject MakeFrontWall(float faceZ, int layer = ChamberWallHider.WallLayer)
        {
            return MakeWall(new Vector3(0f, 0.02f, faceZ + 0.005f), new Vector3(0.2f, 0.1f, 0.01f), layer);
        }

        [Test]
        public void EnsureSolidShell_AddsHeadAndAbdomenOnce()
        {
            using var fixture = new LocomotionFixture();
            fixture.Body.EnsureSolidShell();
            fixture.Body.EnsureSolidShell();

            var spheres = fixture.Fly.GetComponents<SphereCollider>();
            Assert.AreEqual(3, spheres.Length, "ground + head + abdomen");
            Assert.IsTrue(System.Array.Exists(spheres, s =>
                s.center == FlyBodyController.HeadCenter &&
                Mathf.Approximately(s.radius, FlyBodyController.HeadRadius) &&
                !s.isTrigger));
            Assert.IsTrue(System.Array.Exists(spheres, s =>
                s.center == FlyBodyController.AbdomenCenter &&
                Mathf.Approximately(s.radius, FlyBodyController.GroundRadius)));
        }

        [Test]
        public void HeadSphere_ContainsEyeNearPlane()
        {
            float tanV = Mathf.Tan(0.5f * FlyEyeCamera.DefaultFieldOfView * Mathf.Deg2Rad);
            float halfW = FlyEyeCamera.NearClip * tanV;
            float halfH = halfW;
            foreach (bool isLeft in new[] { true, false })
            {
                Vector3 eye = FlyEyeCamera.MountOffset(isLeft);
                Quaternion rot = FlyEyeCamera.MountRotation(isLeft);
                for (int sx = -1; sx <= 1; sx += 2)
                {
                    for (int sy = -1; sy <= 1; sy += 2)
                    {
                        Vector3 local = new Vector3(sx * halfW, sy * halfH, FlyEyeCamera.NearClip);
                        Vector3 corner = eye + rot * local;
                        Assert.Less(corner.magnitude + 1e-4f, FlyBodyController.HeadRadius);
                    }
                }
            }
        }

        [Test]
        public void ClampMove_StopsHeadAtWallFace()
        {
            using var fixture = new LocomotionFixture();
            var wall = MakeFrontWall(0.05f);
            try
            {
                Transform root = fixture.Fly.transform;
                Vector3 clamped = BodyClearance.ClampMove(
                    root, root.position, root.rotation, Vector3.forward * 0.1f);

                Assert.AreEqual(0.05f - HeadFrontZ, clamped.z, 1e-4f);
                Assert.LessOrEqual(clamped.z + HeadFrontZ, 0.05f + 1e-5f);
            }
            finally
            {
                Object.DestroyImmediate(wall);
            }
        }

        [Test]
        public void ClampMove_LeavesFreeMovesAlone()
        {
            using var fixture = new LocomotionFixture();
            var wall = MakeFrontWall(0.05f);
            try
            {
                Transform root = fixture.Fly.transform;
                Vector3 back = Vector3.back * 0.01f;
                Vector3 nudge = Vector3.forward * 0.001f;
                Assert.Less(Vector3.Distance(back, BodyClearance.ClampMove(root, root.position, root.rotation, back)), 1e-7f);
                Assert.Less(Vector3.Distance(nudge, BodyClearance.ClampMove(root, root.position, root.rotation, nudge)), 1e-7f);
            }
            finally
            {
                Object.DestroyImmediate(wall);
            }
        }

        [Test]
        public void ClampMove_WhenTouching_OnlyAllowsBackingAway()
        {
            using var fixture = new LocomotionFixture();
            var wall = MakeFrontWall(HeadFrontZ - BodyClearance.Skin * 0.5f);
            try
            {
                Transform root = fixture.Fly.transform;
                Vector3 forward = BodyClearance.ClampMove(
                    root, root.position, root.rotation, Vector3.forward * 0.01f);
                Vector3 back = BodyClearance.ClampMove(
                    root, root.position, root.rotation, Vector3.back * 0.01f);

                Assert.AreEqual(0f, forward.magnitude, 1e-6f);
                Assert.AreEqual(-0.01f, back.z, 1e-6f);
            }
            finally
            {
                Object.DestroyImmediate(wall);
            }
        }

        [Test]
        public void ClampMove_IgnoresNonWallLayers()
        {
            using var fixture = new LocomotionFixture();
            var block = MakeFrontWall(0.05f, layer: 0);
            try
            {
                Transform root = fixture.Fly.transform;
                Vector3 delta = Vector3.forward * 0.1f;
                Assert.Less(Vector3.Distance(delta, BodyClearance.ClampMove(root, root.position, root.rotation, delta)), 1e-7f);
            }
            finally
            {
                Object.DestroyImmediate(block);
            }
        }

        [Test]
        public void CanRotate_RefusesSwingingHeadIntoSideWall()
        {
            using var fixture = new LocomotionFixture();
            // Wall face at x = 0.005 beside the head only (z 0.005..0.03), so the abdomen swing is free.
            var wall = MakeWall(new Vector3(0.01f, 0.02f, 0.0175f), new Vector3(0.01f, 0.1f, 0.025f));
            try
            {
                Transform root = fixture.Fly.transform;
                Quaternion now = root.rotation;
                Quaternion towardWall = now * Quaternion.Euler(0f, 30f, 0f);
                Quaternion awayFromWall = now * Quaternion.Euler(0f, -30f, 0f);

                Assert.IsFalse(BodyClearance.CanRotate(root, root.position, now, towardWall));
                Assert.IsTrue(BodyClearance.CanRotate(root, root.position, now, awayFromWall));
            }
            finally
            {
                Object.DestroyImmediate(wall);
            }
        }

        [Test]
        public void WalkingIntoWall_HeadNeverPassesWallFace()
        {
            using var fixture = new LocomotionFixture();
            const float face = 0.03f;
            var wall = MakeFrontWall(face);
            try
            {
                fixture.Motors.SetHighLevelCommands(1f, 0.3f, 0.5f, 0.8f);
                const float dt = 0.005f;
                for (int i = 0; i < 600; i++)
                {
                    fixture.Body.FixedSensorUpdate();
                    fixture.Motors.Tick(dt);
                    Physics.Simulate(dt);

                    Vector3 head = fixture.Fly.transform.TransformPoint(FlyBodyController.HeadCenter);
                    Assert.LessOrEqual(
                        head.z + FlyBodyController.HeadRadius,
                        face + 1e-4f,
                        $"head entered the wall at step {i}");
                }
            }
            finally
            {
                Object.DestroyImmediate(wall);
            }
        }
    }
}
