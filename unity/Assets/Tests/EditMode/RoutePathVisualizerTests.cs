using System.Collections.Generic;
using FlySimulation.Fly.DebugViz;
using NUnit.Framework;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class RoutePathVisualizerTests
    {
        [Test]
        public void ApplyPath_SetsLinePositionsAndMarker()
        {
            var go = new GameObject("RoutePathVizTest");
            try
            {
                var viz = go.AddComponent<RoutePathVisualizer>();
                var points = new List<Vector3>
                {
                    new Vector3(0f, 0.02f, 0f),
                    new Vector3(0.05f, 0.02f, 0.1f),
                    new Vector3(0.1f, 0.02f, 0.2f),
                };

                viz.ApplyPath(points, currentIndex: 1, mode: 1);

                Assert.IsNotNull(viz.Line);
                Assert.AreEqual(3, viz.Line.positionCount);
                Assert.AreEqual(
                    RoutePathVisualizer.DefaultLineWidth,
                    viz.Line.startWidth,
                    1e-6f);
                Assert.AreEqual(viz.Line.startWidth, viz.Line.endWidth, 1e-6f);
                Assert.AreEqual(
                    points[0].x,
                    viz.Line.GetPosition(0).x,
                    1e-5f);
                Assert.IsTrue(viz.Marker.gameObject.activeSelf);
                Assert.AreEqual(points[1].x, viz.Marker.position.x, 1e-5f);
                Assert.AreEqual(
                    points[1].y + RoutePathVisualizer.DefaultYOffset,
                    viz.Marker.position.y,
                    1e-5f);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        [Test]
        public void ApplyPath_ModeOff_ClearsLine()
        {
            var go = new GameObject("RoutePathVizOff");
            try
            {
                var viz = go.AddComponent<RoutePathVisualizer>();
                viz.ApplyPath(
                    new List<Vector3> { Vector3.zero, Vector3.forward },
                    currentIndex: 0,
                    mode: 1);
                viz.ApplyPath(
                    new List<Vector3> { Vector3.zero, Vector3.forward },
                    currentIndex: 0,
                    mode: 0);

                Assert.AreEqual(0, viz.Line.positionCount);
                Assert.IsFalse(viz.Marker.gameObject.activeSelf);
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }
    }
}
