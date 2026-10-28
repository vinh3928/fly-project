using System.Collections.Generic;
using System.Linq;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Sensors;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace FlySimulation.Tests.EditMode
{
    public sealed class ScriptFileNameTests
    {
        [Test]
        public void EveryRuntimeMonoBehaviour_HasMatchingScriptFile()
        {
            // MonoScript.GetClass() only resolves when the class name matches its file name;
            // otherwise components built into a scene cannot be reloaded (missing script).
            var resolved = new HashSet<System.Type>(
                AssetDatabase.FindAssets("t:MonoScript", new[] { "Assets" })
                    .Select(AssetDatabase.GUIDToAssetPath)
                    .Select(AssetDatabase.LoadAssetAtPath<MonoScript>)
                    .Where(s => s != null)
                    .Select(s => s.GetClass())
                    .Where(t => t != null));

            var missing = typeof(ProboscisSensor).Assembly.GetTypes()
                .Where(t => typeof(MonoBehaviour).IsAssignableFrom(t) && !t.IsAbstract && !t.IsGenericType)
                .Where(t => !resolved.Contains(t))
                .Select(t => t.FullName)
                .ToList();

            Assert.IsEmpty(missing, "MonoBehaviours without a matching .cs file: " + string.Join(", ", missing));
        }

        [Test]
        public void MotorProboscisCommand_ReachesSensor()
        {
            var go = new GameObject("fly");
            var sensor = go.AddComponent<ProboscisSensor>();
            var actuator = go.AddComponent<ProboscisActuator>();
            actuator.Bind(sensor);
            var motors = go.AddComponent<FlyMotorController>();
            motors.Bind(null, actuator);

            motors.SetProboscisExtension(1f);
            Assert.AreEqual(1f, sensor.ExtensionNormalized, 1e-6f);
            motors.SetProboscisExtension(0.25f);
            Assert.AreEqual(0.25f, sensor.ExtensionNormalized, 1e-6f);

            Object.DestroyImmediate(go);
        }
    }
}
