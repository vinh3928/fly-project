using FlySimulation.Environment;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class FlyFollowCameraTests
    {
        [Test]
        public void Place_TargetsDisplay2_AndStaysEnabled()
        {
            var go = new GameObject(FlyFollowCamera.ObjectName);
            try
            {
                var cam = go.AddComponent<Camera>();
                FlyFollowCamera.Place(cam);
                Assert.AreEqual(FlyFollowCamera.TargetDisplayIndex, cam.targetDisplay);
                Assert.IsTrue(cam.enabled);
                Assert.AreEqual("Untagged", cam.tag);
                Assert.AreEqual(0, cam.cullingMask & (1 << ChamberWallHider.WallLayer));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void SnapToTarget_CentersBehindFly()
        {
            var fly = new GameObject("Fly");
            var camGo = new GameObject(FlyFollowCamera.ObjectName);
            try
            {
                fly.transform.position = new Vector3(0.1f, 0.02f, -0.2f);
                fly.transform.rotation = Quaternion.Euler(0f, 45f, 0f);

                var cam = camGo.AddComponent<Camera>();
                var follow = camGo.AddComponent<FlyFollowCamera>();
                follow.Configure(new Vector3(0f, 0.04f, -0.08f), 55f);
                follow.Bind(fly.transform);

                Vector3 expectedPos = fly.transform.TransformPoint(new Vector3(0f, 0.04f, -0.08f));
                Assert.AreEqual(expectedPos.x, camGo.transform.position.x, 1e-5f);
                Assert.AreEqual(expectedPos.y, camGo.transform.position.y, 1e-5f);
                Assert.AreEqual(expectedPos.z, camGo.transform.position.z, 1e-5f);

                Vector3 toFly = fly.transform.TransformPoint(new Vector3(0f, 0.008f, 0.01f)) - camGo.transform.position;
                Assert.Greater(Vector3.Dot(camGo.transform.forward, toFly.normalized), 0.99f);
                Assert.AreEqual(FlyFollowCamera.TargetDisplayIndex, cam.targetDisplay);
            }
            finally
            {
                Object.DestroyImmediate(camGo);
                Object.DestroyImmediate(fly);
            }
        }

        [Test]
        public void IsFollowCamera_DetectsComponentAndName()
        {
            var named = new GameObject(FlyFollowCamera.ObjectName);
            var other = new GameObject("OtherCam");
            try
            {
                var namedCam = named.AddComponent<Camera>();
                Assert.IsTrue(FlyFollowCamera.IsFollowCamera(namedCam));

                var otherCam = other.AddComponent<Camera>();
                Assert.IsFalse(FlyFollowCamera.IsFollowCamera(otherCam));
                other.AddComponent<FlyFollowCamera>();
                Assert.IsTrue(FlyFollowCamera.IsFollowCamera(otherCam));
            }
            finally
            {
                Object.DestroyImmediate(named);
                Object.DestroyImmediate(other);
            }
        }
    }
}
