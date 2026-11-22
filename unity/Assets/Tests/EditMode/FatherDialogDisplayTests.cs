using FlySimulation.Environment;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class FatherDialogDisplayTests
    {
        [Test]
        public void ShowForTests_SetsQuestionAndVisibility()
        {
            var go = new GameObject("FatherDialogDisplayTest");
            try
            {
                var dialog = go.AddComponent<FatherDialogDisplay>();
                dialog.ShowForTests(9, "smell_novelty", "Is this food?", allowLlm: true);
                Assert.IsTrue(dialog.IsVisible);
                Assert.AreEqual("Is this food?", dialog.QuestionText);
                dialog.Hide();
                Assert.IsFalse(dialog.IsVisible);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
