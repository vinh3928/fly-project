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
                // showOnPlay defaults true; Awake loads profiles. We only assert component wires.
                var menu = go.AddComponent<FlyProfileMenu>();
                Assert.IsNotNull(menu);
                Assert.IsFalse(string.IsNullOrEmpty(menu.ActiveFlyId));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
