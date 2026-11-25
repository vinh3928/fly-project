using System.Collections.Generic;
using FlySimulation.Fly.DebugViz;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class PlaceGraphVisualizerTests
    {
        [Test]
        public void ApplySnapshot_ShowsOnlyCurrentLabel()
        {
            var go = new GameObject("PlaceGraphVizTest");
            try
            {
                var viz = go.AddComponent<PlaceGraphVisualizer>();
                viz.ShowLabels = true;
                var nodes = new List<PlaceGraphVisualizer.NodeSnap>
                {
                    new PlaceGraphVisualizer.NodeSnap(
                        PlaceGraphVisualizer.LabelPlace,
                        PlaceGraphVisualizer.FlagCurrent,
                        new Vector3(0f, 0.02f, 0f),
                        0.1f),
                    new PlaceGraphVisualizer.NodeSnap(
                        PlaceGraphVisualizer.LabelPuddle,
                        PlaceGraphVisualizer.FlagHome | PlaceGraphVisualizer.FlagRoute,
                        new Vector3(0.1f, 0.02f, 0.2f),
                        0.8f),
                };
                var edges = new List<Vector2Int> { new Vector2Int(0, 1) };

                viz.ApplySnapshot(nodes, edges);

                Assert.AreEqual(1, viz.ActiveNodeCount);
                Assert.AreEqual(1, viz.ActiveEdgeCount);

                Transform labelRoot = go.transform.Find("PlaceGraphLabel");
                Assert.IsNotNull(labelRoot);
                Assert.IsTrue(labelRoot.gameObject.activeSelf);
                Assert.AreEqual(
                    nodes[0].Position.x,
                    labelRoot.position.x,
                    1e-5f);
                var label = labelRoot.GetComponentInChildren<TextMesh>();
                Assert.IsNotNull(label);
                Assert.AreEqual("place* s=0.10", label.text);
                // No per-edge LineRenderer pool (path line is RoutePathVisualizer).
                Assert.IsNull(go.transform.Find("PlaceGraphEdges"));
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void ApplySnapshot_Empty_HidesLabel()
        {
            var go = new GameObject("PlaceGraphVizEmpty");
            try
            {
                var viz = go.AddComponent<PlaceGraphVisualizer>();
                viz.ShowLabels = true;
                viz.ApplySnapshot(
                    new List<PlaceGraphVisualizer.NodeSnap>
                    {
                        new PlaceGraphVisualizer.NodeSnap(
                            PlaceGraphVisualizer.LabelPlace,
                            PlaceGraphVisualizer.FlagCurrent,
                            Vector3.zero,
                            0f),
                    },
                    new List<Vector2Int>());
                Assert.AreEqual(1, viz.ActiveNodeCount);

                viz.ApplySnapshot(
                    new List<PlaceGraphVisualizer.NodeSnap>(),
                    new List<Vector2Int>());
                Assert.AreEqual(0, viz.ActiveNodeCount);
                Transform labelRoot = go.transform.Find("PlaceGraphLabel");
                Assert.IsNotNull(labelRoot);
                Assert.IsFalse(labelRoot.gameObject.activeSelf);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
