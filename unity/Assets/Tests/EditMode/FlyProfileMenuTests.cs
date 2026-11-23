using FlySimulation.Environment;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class FlyProfileMenuTests
    {
        [Test]
        public void MenuComponent_DefaultsNotReadyUntilShown()
        {
            var go = new GameObject(FlyProfileMenu.ObjectName);
            try
            {
                // showOnPlay defaults true; Awake loads profiles and gates the session.
                var menu = go.AddComponent<FlyProfileMenu>();
                Assert.IsNotNull(menu);
                Assert.IsFalse(string.IsNullOrEmpty(menu.ActiveFlyId));
                Assert.IsFalse(menu.IsReady);
                Assert.IsFalse(FlyProfileMenu.IsSessionReady);
            }
            finally
            {
                Object.DestroyImmediate(go);
                Time.timeScale = 1f;
            }
        }

        [Test]
        public void IsSessionReady_TrueWhenNoMenuPresent()
        {
            var existing = Object.FindFirstObjectByType<FlyProfileMenu>();
            if (existing != null)
            {
                Object.DestroyImmediate(existing.gameObject);
            }

            Assert.IsTrue(FlyProfileMenu.IsSessionReady);
        }
    }
}
