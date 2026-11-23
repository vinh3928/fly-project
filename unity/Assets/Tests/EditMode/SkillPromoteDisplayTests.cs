using FlySimulation.Environment;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class SkillPromoteDisplayTests
    {
        [Test]
        public void ShowForTests_SetsQueueHeaderAndNameField()
        {
            var go = new GameObject("SkillPromoteDisplayTest");
            try
            {
                go.AddComponent<Camera>();
                var display = go.AddComponent<SkillPromoteDisplay>();
                display.ShowForTests(
                    requestId: 4,
                    queueCount: 2,
                    skillId: "skill-9",
                    name: "candidate_dance_abc",
                    tags: "dance",
                    score: "0.72",
                    stepsCount: "5",
                    suggestedName: "dance_spin",
                    allowLlm: true);
                Assert.IsTrue(display.IsVisible);
                StringAssert.Contains("2 ready", display.TitleText);
                Assert.AreEqual("dance_spin", display.NameFieldText);
                Assert.AreEqual(0f, Time.timeScale, 1e-4f);
                display.Hide();
                Assert.IsFalse(display.IsVisible);
                Assert.Greater(Time.timeScale, 0f);
            }
            finally
            {
                Time.timeScale = 1f;
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void SubmitAskLlmForTests_ThawsTimeScaleWhileKeepingPanel()
        {
            // Ask LLM must thaw timeScale so ML-Agents FixedUpdate can flush
            // the SideChannel reply; Accept/Refuse Hide() already did that.
            var go = new GameObject("SkillPromoteAskLlmTest");
            try
            {
                go.AddComponent<Camera>();
                var display = go.AddComponent<SkillPromoteDisplay>();
                display.ShowForTests(
                    requestId: 7,
                    queueCount: 1,
                    skillId: "skill-llm",
                    name: "candidate_pivot",
                    tags: "pivot,forward",
                    score: "0.8",
                    stepsCount: "3",
                    suggestedName: "pivot_forward",
                    allowLlm: true);
                Assert.AreEqual(0f, Time.timeScale, 1e-4f);
                display.SubmitAskLlmForTests();
                Assert.IsTrue(display.IsVisible);
                Assert.IsTrue(display.IsAwaitingLlmJudge);
                Assert.Greater(Time.timeScale, 0f);
                StringAssert.Contains("judging", display.TitleText.ToLowerInvariant());
                display.Hide();
                Assert.IsFalse(display.IsAwaitingLlmJudge);
            }
            finally
            {
                Time.timeScale = 1f;
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void DesiredFraming_LooksAtFlyCenter()
        {
            var fly = new GameObject("FlyCenterTarget");
            var camGo = new GameObject("SkillPromoteCenterCam");
            try
            {
                fly.transform.position = new Vector3(1f, 0.02f, 2f);
                fly.transform.rotation = Quaternion.Euler(0f, 90f, 0f);
                var marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
                marker.transform.SetParent(fly.transform, false);
                marker.transform.localPosition = Vector3.zero;
                marker.transform.localScale = Vector3.one * 0.02f;

                var cam = camGo.AddComponent<Camera>();
                var display = camGo.AddComponent<SkillPromoteDisplay>();
                display.Bind(fly.transform);
                display.SnapToTargetForTests();

                Vector3 toFly = fly.transform.position - camGo.transform.position;
                float align = Vector3.Dot(camGo.transform.forward.normalized, toFly.normalized);
                Assert.Greater(align, 0.95f, "Camera should look toward the fly center");
                Assert.AreEqual(SkillPromoteDisplay.TargetDisplayIndex, cam.targetDisplay);
            }
            finally
            {
                Time.timeScale = 1f;
                Object.DestroyImmediate(camGo);
                Object.DestroyImmediate(fly);
            }
        }

        [Test]
        public void Place_SetsTargetDisplay4()
        {
            var go = new GameObject("SkillPromoteCamTest");
            try
            {
                var cam = go.AddComponent<Camera>();
                SkillPromoteDisplay.Place(cam);
                Assert.AreEqual(SkillPromoteDisplay.TargetDisplayIndex, cam.targetDisplay);
                Assert.IsTrue(cam.enabled);
                Assert.IsTrue(SkillPromoteDisplay.IsPromoteCamera(cam));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void EnsureInScene_CreatesCameraOnDisplay4()
        {
            var existing = Object.FindFirstObjectByType<SkillPromoteDisplay>();
            if (existing != null)
            {
                Object.DestroyImmediate(existing.gameObject);
            }

            var display = SkillPromoteDisplay.EnsureInScene();
            try
            {
                Assert.IsNotNull(display);
                Assert.IsNotNull(display.Camera);
                Assert.IsTrue(display.Camera.enabled);
                Assert.AreEqual(SkillPromoteDisplay.TargetDisplayIndex, display.Camera.targetDisplay);
                Assert.AreEqual(SkillPromoteDisplay.ObjectName, display.gameObject.name);
                Assert.GreaterOrEqual(display.Camera.nearClipPlane, 0.01f);
            }
            finally
            {
                if (display != null)
                {
                    Object.DestroyImmediate(display.gameObject);
                }
            }
        }

        [Test]
        public void Place_UsesSafeNearClipForMetal()
        {
            var go = new GameObject("SkillPromoteNearClip");
            try
            {
                var cam = go.AddComponent<Camera>();
                cam.nearClipPlane = 0.001f;
                SkillPromoteDisplay.Place(cam);
                Assert.GreaterOrEqual(cam.nearClipPlane, 0.01f);
                Assert.IsTrue(cam.enabled);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
