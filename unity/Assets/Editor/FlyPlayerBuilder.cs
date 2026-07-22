using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Sensors;

namespace FlySimulation.EditorTools
{
    /// <summary>
    /// Builds a standalone player that Python / ML-Agents can launch.
    /// </summary>
    public static class FlyPlayerBuilder
    {
        public const string BootScenePath = "Assets/Scenes/FlyBoot.unity";
        public const string ChamberScenePath = "Assets/Scenes/Chamber.unity";

        [MenuItem("Tools/Fly/Build Training Player", priority = 20)]
        [MenuItem("Fly/Build Training Player", priority = 20)]
        public static void BuildFromMenu()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.Log("Stopping Play mode before building the training player...");
                EditorApplication.playModeStateChanged -= BuildAfterLeavingPlayMode;
                EditorApplication.playModeStateChanged += BuildAfterLeavingPlayMode;
                EditorApplication.isPlaying = false;
                return;
            }

            BuildPlayer(exitEditor: false);
        }

        private static void BuildAfterLeavingPlayMode(PlayModeStateChange state)
        {
            if (state != PlayModeStateChange.EnteredEditMode)
            {
                return;
            }

            EditorApplication.playModeStateChanged -= BuildAfterLeavingPlayMode;
            BuildPlayer(exitEditor: false);
        }

        /// <summary>
        /// CLI entry: Unity -batchmode -quit -projectPath unity
        ///   -executeMethod FlySimulation.EditorTools.FlyPlayerBuilder.BuildCli
        /// </summary>
        public static void BuildCli()
        {
            try
            {
                if (EditorApplication.isPlayingOrWillChangePlaymode)
                {
                    throw new InvalidOperationException(
                        "Cannot build while Play mode is active. Exit Play mode first.");
                }

                BuildPlayer(exitEditor: true);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Fly player build failed: {ex}");
                EditorApplication.Exit(1);
            }
        }

        public static void BuildPlayer(bool exitEditor)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                throw new InvalidOperationException(
                    "Cannot build while Play mode is active. Stop Play mode, then run " +
                    "Tools → Fly → Build Training Player again.");
            }

            PlayerSettings.companyName = "FlyProject";
            PlayerSettings.productName = "FlyChamber";

            CreateBootScene();

            var scenes = new[]
            {
                new EditorBuildSettingsScene(BootScenePath, true)
            };
            EditorBuildSettings.scenes = scenes;

            string repoRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", ".."));
            string buildsDir = Path.Combine(repoRoot, "builds");
            Directory.CreateDirectory(buildsDir);

            string location = ResolveOutputPath(buildsDir);
            if (Directory.Exists(location) || File.Exists(location))
            {
                FileUtil.DeleteFileOrDirectory(location);
                FileUtil.DeleteFileOrDirectory(location + ".meta");
            }

            var options = new BuildPlayerOptions
            {
                scenes = new[] { BootScenePath },
                locationPathName = location,
                target =
#if UNITY_EDITOR_OSX
                    BuildTarget.StandaloneOSX,
#elif UNITY_EDITOR_WIN
                    BuildTarget.StandaloneWindows64,
#else
                    BuildTarget.StandaloneLinux64,
#endif
                options = BuildOptions.Development
            };

            Debug.Log($"Building FlyChamber player → {location} ({options.target})");
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new InvalidOperationException(
                    $"Build failed: {report.summary.result} ({report.summary.totalErrors} errors)");
            }

            Debug.Log(
                $"FlyChamber player ready: {location}. " +
                "Launch from Python with: python -m scripts.launch_unity");

            if (exitEditor)
            {
                EditorApplication.Exit(0);
            }
        }

        /// <summary>
        /// Lean ML-Agents scene: floor + agent. Avoids Chamber hierarchy serialization issues
        /// while still satisfying the Phase 0 Python launch exit criterion.
        /// </summary>
        public static void CreateBootScene()
        {
            DirectoryEnsure("Assets/Scenes");
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var lightGo = new GameObject("MainLight");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1f;
            lightGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            floor.transform.localScale = Vector3.one * 0.2f;

            var agentGo = new GameObject("FlyAgent");
            agentGo.transform.position = new Vector3(0f, 0.5f, 0f);
            agentGo.AddComponent<MinimalLaunchAgent>();
            // Agent.RequireComponent may add BehaviorParameters asynchronously to Reset;
            // configure twice so DecisionRequester + name persist into the saved scene.
            ChamberSceneBuilder.ConfigureMlAgents(agentGo);
            ChamberSceneBuilder.ConfigureMlAgents(agentGo);

            var camGo = new GameObject("MainCamera");
            var cam = camGo.AddComponent<Camera>();
            cam.tag = "MainCamera";
            camGo.transform.position = new Vector3(0f, 2f, -4f);
            camGo.transform.LookAt(agentGo.transform);
            camGo.AddComponent<AudioListener>();

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, BootScenePath);
            AssetDatabase.SaveAssets();

            // Force-persist ML-Agents fields (Agent serialization can reset defaults on save).
            PatchBootSceneYaml(BootScenePath);
            AssetDatabase.Refresh();

            string yaml = File.ReadAllText(BootScenePath);
            if (!yaml.Contains("m_BehaviorName: FlyAgent"))
            {
                throw new InvalidOperationException(
                    "FlyBoot scene did not persist BehaviorName=FlyAgent.");
            }

            if (!yaml.Contains("DecisionRequester"))
            {
                throw new InvalidOperationException(
                    "FlyBoot scene is missing DecisionRequester. Check batchmode type lookup logs.");
            }

            Debug.Log($"Boot scene saved: {BootScenePath}");
        }

        private static void PatchBootSceneYaml(string path)
        {
            string yaml = File.ReadAllText(path);
            yaml = yaml.Replace("m_BehaviorName: My Behavior", "m_BehaviorName: FlyAgent");
            yaml = System.Text.RegularExpressions.Regex.Replace(
                yaml,
                @"VectorObservationSize:\s*\d+",
                $"VectorObservationSize: {SensorManager.ExpectedVectorSize}");
            yaml = System.Text.RegularExpressions.Regex.Replace(
                yaml,
                @"m_NumContinuousActions:\s*\d+",
                "m_NumContinuousActions: 5");
            File.WriteAllText(path, yaml);
        }

        private static string ResolveOutputPath(string buildsDir)
        {
#if UNITY_EDITOR_OSX
            return Path.Combine(buildsDir, "FlyChamber.app");
#elif UNITY_EDITOR_WIN
            return Path.Combine(buildsDir, "FlyChamber.exe");
#else
            return Path.Combine(buildsDir, "FlyChamber.x86_64");
#endif
        }

        private static void DirectoryEnsure(string path)
        {
            if (AssetDatabase.IsValidFolder(path))
            {
                return;
            }

            var parts = path.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(current, parts[i]);
                }

                current = next;
            }
        }
    }
}
