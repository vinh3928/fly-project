using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Recording;
using FlySimulation.Fly.Sensors;

namespace FlySimulation.EditorTools
{
    /// <summary>
    /// Builds a visible, runnable chamber scene hierarchy without manual wiring.
    /// </summary>
    [InitializeOnLoad]
    public static class ChamberSceneBuilder
    {
        static ChamberSceneBuilder()
        {
            // Confirms the editor script loaded; look for menus under Tools/Fly and Fly.
            Debug.Log("Fly chamber tools loaded. Use menu: Tools > Fly > Setup Chamber Scene");
        }

        [MenuItem("Tools/Fly/Setup Chamber Scene", priority = 0)]
        [MenuItem("Fly/Setup Chamber Scene", priority = 0)]
        public static void SetupChamberScene()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.LogWarning(
                    "Stop Play mode before running Tools → Fly → Setup Chamber Scene.");
                EditorApplication.isPlaying = false;
                return;
            }

            // Keep Built-in RP (no SRP asset) so Unlit/Color materials never go pink.
            UnityEngine.Rendering.GraphicsSettings.defaultRenderPipeline = null;
            QualitySettings.renderPipeline = null;
            EnsureTags("Fly", "Chamber", "ChemicalSource", "Obstacle", "VisualTarget");

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var config = ExperimentConfiguration.Default();

            // Built-in Unlit/Color — works with GraphicsSettings pipeline = null.
            Material floorMat = CreateColorMaterial("FloorMat", new Color(0.45f, 0.48f, 0.52f), overwrite: true);
            Material wallMat = CreateColorMaterial("WallMat", new Color(0.55f, 0.7f, 0.95f, 0.12f), transparent: true, overwrite: true);
            Material sugarMat = CreateColorMaterial("SugarMat", new Color(0.95f, 0.75f, 0.2f), overwrite: true);
            Material bitterMat = CreateColorMaterial("BitterMat", new Color(0.55f, 0.2f, 0.75f), overwrite: true);
            Material flyMat = CreateColorMaterial("FlyMat", new Color(0.12f, 0.12f, 0.12f), overwrite: true);
            Material accentMat = CreateColorMaterial("FlyAccentMat", new Color(1f, 0.4f, 0.05f), overwrite: true);
            Material outlineMat = CreateColorMaterial("OutlineMat", new Color(0.1f, 1f, 1f), overwrite: true);
            Material targetMat = CreateColorMaterial("VisualTargetMat", new Color(0.1f, 0.95f, 0.35f), overwrite: true);
            Material obstacleMat = CreateColorMaterial("ObstacleMat", new Color(0.18f, 0.16f, 0.14f), overwrite: true);
            Material floorMarkMat = CreateColorMaterial("FloorMarkMat", new Color(0.72f, 0.74f, 0.78f), overwrite: true);

            var chamberRoot = new GameObject("Chamber");
            TrySetTag(chamberRoot, "Chamber");

            var floor = CreateCube("Floor", chamberRoot.transform, floorMat);
            var ceiling = CreateCube("Ceiling", chamberRoot.transform, wallMat);
            ceiling.GetComponent<MeshRenderer>().enabled = false; // keep collider, hide mesh

            var walls = new Transform[4];
            for (int i = 0; i < 4; i++)
            {
                walls[i] = CreateCube($"Wall_{i}", chamberRoot.transform, wallMat).transform;
            }

            // Hide all wall meshes so Game view looks into the arena.
            // Colliders stay active for physics.
            for (int i = 0; i < walls.Length; i++)
            {
                SetRendererEnabled(walls[i], false);
            }

            var sugar = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            sugar.name = "SugarSource";
            TrySetTag(sugar, "ChemicalSource");
            sugar.transform.SetParent(chamberRoot.transform, false);
            sugar.transform.localScale = new Vector3(
                ExperimentConfiguration.SourcePuddleDiameter,
                0.002f,
                ExperimentConfiguration.SourcePuddleDiameter);
            ApplyMaterial(sugar, sugarMat);
            var sugarSource = sugar.AddComponent<ChemicalSource>();
            sugarSource.Configure(
                ChemicalSource.SourceType.Sugar,
                1f,
                ExperimentConfiguration.DefaultChemicalSpread);
            EnsureSurface(sugar, new ChemicalChannels { Sugar = 1f });

            var bitter = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            bitter.name = "BitterSource";
            TrySetTag(bitter, "ChemicalSource");
            bitter.transform.SetParent(chamberRoot.transform, false);
            bitter.transform.localScale = new Vector3(
                ExperimentConfiguration.SourcePuddleDiameter,
                0.002f,
                ExperimentConfiguration.SourcePuddleDiameter);
            ApplyMaterial(bitter, bitterMat);
            var bitterSource = bitter.AddComponent<ChemicalSource>();
            bitterSource.Configure(
                ChemicalSource.SourceType.Bitter,
                1f,
                ExperimentConfiguration.DefaultChemicalSpread);
            EnsureSurface(bitter, new ChemicalChannels { Bitter = 1f });

            // Phase 5: high-contrast floor patches for visual landmarking (walls stay mesh-hidden).
            for (int i = 0; i < 4; i++)
            {
                var mark = CreateCube($"FloorMark_{i}", chamberRoot.transform, floorMarkMat);
                float ox = (i % 2 == 0 ? -1f : 1f) * config.ChamberSize.x * 0.22f;
                float oz = (i < 2 ? -1f : 1f) * config.ChamberSize.z * 0.22f;
                mark.transform.localPosition = new Vector3(ox, 0.0006f, oz);
                mark.transform.localScale = new Vector3(
                    config.ChamberSize.x * 0.12f,
                    0.001f,
                    config.ChamberSize.z * 0.12f);
            }

            // Phase 5: bright visual target (distinct from sugar/bitter chemicals).
            var visualTarget = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            visualTarget.name = "VisualTarget";
            TrySetTag(visualTarget, Phase5ExitCriteria.VisualTargetTag);
            visualTarget.transform.SetParent(chamberRoot.transform, false);
            float targetRadius = 0.006f;
            Vector3 targetPos = DeterministicLayout.GenerateVisualTargetPosition(
                config.RandomSeed,
                config.ChamberSize,
                placementMargin: 0.02f,
                targetRadius: targetRadius);
            visualTarget.transform.localPosition = targetPos;
            visualTarget.transform.localScale = Vector3.one * (targetRadius * 2f);
            ApplyMaterial(visualTarget, targetMat);

            // Chamber bounds are the floor/walls; skip LineRenderer outline (can corrupt player builds
            // when paired with Unlit/Color materials).
            // CreateChamberOutline(chamberRoot.transform, config.ChamberSize, outlineMat);

            var lightGo = new GameObject("MainLight");
            lightGo.transform.SetParent(chamberRoot.transform, false);
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.2f;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            var chemicalField = chamberRoot.AddComponent<ChemicalFieldSystem>();
            var chamber = chamberRoot.AddComponent<ChamberController>();
            var randomizer = chamberRoot.AddComponent<EnvironmentRandomizer>();
            chamberRoot.AddComponent<ChamberWallHider>();

            SetPrivate(chamber, "floor", floor.transform);
            SetPrivate(chamber, "ceiling", ceiling.transform);
            SetPrivate(chamber, "walls", walls);
            SetPrivate(chamber, "mainLight", light);
            SetPrivate(chamber, "sugarSource", sugarSource);
            SetPrivate(chamber, "bitterSource", bitterSource);
            SetPrivate(chamber, "chemicalField", chemicalField);
            SetPrivate(chamber, "configuration", config);
            chamber.ApplyConfiguration(config);

            // Phase 5: visible dark pillars (not giant white cubes).
            var soRandomizer = new SerializedObject(randomizer);
            soRandomizer.FindProperty("obstacleCount").intValue = Phase5ExitCriteria.DefaultObstacleCount;
            soRandomizer.FindProperty("obstacleRadius").floatValue = Phase5ExitCriteria.DefaultObstacleRadius;
            soRandomizer.FindProperty("placementMargin").floatValue = 0.015f;
            soRandomizer.FindProperty("obstacleMaterial").objectReferenceValue = obstacleMat;
            soRandomizer.ApplyModifiedPropertiesWithoutUndo();

            var fly = BuildFly(config, chemicalField, flyMat, accentMat);
            var episode = fly.AddComponent<FlyEpisodeManager>();
            var recorder = fly.AddComponent<ExperimentRecorder>();
            var stability = fly.AddComponent<PhysicsStabilityMonitor>();
            var navMetrics = fly.AddComponent<VisualNavMetrics>();
            navMetrics.Bind(fly.transform, visualTarget.transform, config.ChamberSize);
            fly.AddComponent<FoodSearchMetrics>();
            SetPrivate(episode, "configuration", config);
            SetPrivate(episode, "chamber", chamber);
            SetPrivate(episode, "randomizer", randomizer);
            SetPrivate(episode, "body", fly.GetComponent<FlyBodyController>());
            SetPrivate(episode, "homeostasis", fly.GetComponent<FlyHomeostasis>());
            SetPrivate(episode, "motors", fly.GetComponent<FlyMotorController>());
            SetPrivate(episode, "chemicalField", chemicalField);
            SetPrivate(episode, "recorder", recorder);
            SetPrivate(episode, "stabilityMonitor", stability);
            stability.Bind(fly.GetComponent<Rigidbody>());

            var agent = fly.GetComponent<FlyAgent>();
            SetPrivate(agent, "configuration", config);
            SetPrivate(agent, "episodeManager", episode);
            SetPrivate(agent, "recorder", recorder);

            // Scene overview camera: cutaway corner view into the open chamber.
            var overview = new GameObject("OverviewCamera");
            var cam = overview.AddComponent<Camera>();
            var overviewRig = overview.AddComponent<ChamberOverviewCamera>();
            overviewRig.Configure(config.ChamberSize);
            overview.AddComponent<AudioListener>();
            ChamberOverviewCamera.Place(cam, overview.transform, config.ChamberSize);

            DirectoryEnsure("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, "Assets/Scenes/Chamber.unity");
            AssetDatabase.Refresh();

            Selection.activeGameObject = fly;
            FrameSceneViewOnChamber(config.ChamberSize);
            EditorSceneManager.MarkSceneDirty(scene);

            Debug.Log(
                "Chamber scene ready: Assets/Scenes/Chamber.unity. " +
                $"Arena {config.ChamberSize}, BodyScale={config.BodyScale}, " +
                $"spawn Y={config.FlySpawnPosition.y:F4} (feet on floor). " +
                "Game view uses OverviewCamera (cutaway).");
        }

        [MenuItem("Tools/Fly/Fix Chamber Visibility", priority = 2)]
        [MenuItem("Fly/Fix Chamber Visibility", priority = 2)]
        public static void FixChamberVisibility()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Debug.LogWarning("Stop Play mode before Fix Chamber Visibility.");
                EditorApplication.isPlaying = false;
                return;
            }

            RecreateAllMaterials();

            var wallMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/Fly/Materials/WallMat.mat");
            var ceiling = GameObject.Find("Ceiling");
            var overview = GameObject.Find("OverviewCamera");
            Vector3 chamberSize = ExperimentConfiguration.Default().ChamberSize;
            var chamber = UnityEngine.Object.FindFirstObjectByType<ChamberController>();
            if (chamber != null && chamber.Configuration != null)
            {
                chamberSize = chamber.Configuration.ChamberSize;
            }

            // Ensure transparent wall material is applied and near walls are cut away.
            for (int i = 0; i < 4; i++)
            {
                var wall = GameObject.Find($"Wall_{i}");
                if (wall == null)
                {
                    continue;
                }

                ApplyMaterial(wall, wallMat);
                // Hide every wall mesh — opaque slabs block the Game-view camera.
                SetRendererEnabled(wall.transform, false);
            }

            if (ceiling != null)
            {
                SetRendererEnabled(ceiling.transform, false);
            }

            if (overview == null)
            {
                overview = new GameObject("OverviewCamera");
                overview.AddComponent<Camera>();
                overview.AddComponent<AudioListener>();
            }

            var cam = overview.GetComponent<Camera>();
            var rig = overview.GetComponent<ChamberOverviewCamera>() ?? overview.AddComponent<ChamberOverviewCamera>();
            rig.Configure(chamberSize);
            ChamberOverviewCamera.Place(cam, overview.transform, chamberSize);

            // Disable fly-eye cameras in Game view if they were left enabled.
            foreach (var eye in UnityEngine.Object.FindObjectsByType<Camera>(FindObjectsSortMode.None))
            {
                if (eye != null && eye.gameObject.name != "OverviewCamera")
                {
                    eye.enabled = false;
                    if (eye.CompareTag("MainCamera"))
                    {
                        eye.tag = "Untagged";
                    }
                }
            }

            overview.tag = "MainCamera";
            // Phase 5: keep a few visible dark obstacles for navigation.
            var randomizer = UnityEngine.Object.FindFirstObjectByType<EnvironmentRandomizer>();
            if (randomizer != null)
            {
                var obstacleMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/Fly/Materials/ObstacleMat.mat");
                var so = new SerializedObject(randomizer);
                so.FindProperty("obstacleCount").intValue = Phase5ExitCriteria.DefaultObstacleCount;
                so.FindProperty("obstacleRadius").floatValue = Phase5ExitCriteria.DefaultObstacleRadius;
                so.FindProperty("placementMargin").floatValue = 0.015f;
                if (obstacleMat != null)
                {
                    so.FindProperty("obstacleMaterial").objectReferenceValue = obstacleMat;
                }

                so.ApplyModifiedPropertiesWithoutUndo();
            }

            // Ensure a VisualTarget exists for Phase 5.
            if (GameObject.Find("VisualTarget") == null)
            {
                var targetMat = AssetDatabase.LoadAssetAtPath<Material>("Assets/Fly/Materials/VisualTargetMat.mat")
                    ?? CreateColorMaterial("VisualTargetMat", new Color(0.1f, 0.95f, 0.35f), overwrite: true);
                var chamberGo = GameObject.Find("Chamber");
                var visualTarget = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                visualTarget.name = "VisualTarget";
                TrySetTag(visualTarget, Phase5ExitCriteria.VisualTargetTag);
                if (chamberGo != null)
                {
                    visualTarget.transform.SetParent(chamberGo.transform, false);
                }

                Vector3 size = chamberSize;
                visualTarget.transform.localPosition = DeterministicLayout.GenerateVisualTargetPosition(
                    ExperimentConfiguration.Default().RandomSeed,
                    size,
                    0.02f,
                    0.006f);
                visualTarget.transform.localScale = Vector3.one * 0.012f;
                ApplyMaterial(visualTarget, targetMat);
            }

            ChamberWallHider.HideWalls();
            if (UnityEngine.Object.FindFirstObjectByType<ChamberWallHider>() == null)
            {
                var chamberGo = GameObject.Find("Chamber");
                if (chamberGo != null)
                {
                    chamberGo.AddComponent<ChamberWallHider>();
                }
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log(
                "Chamber visibility fixed: wall meshes hidden, Phase 5 visual target/obstacles ready, " +
                $"OverviewCamera reframed for {chamberSize}. Press Play → Game tab.");
        }

        [MenuItem("Tools/Fly/Fix Pink Materials", priority = 1)]
        [MenuItem("Fly/Fix Pink Materials", priority = 1)]
        public static void FixPinkMaterials()
        {
            // Built-in Unlit/Color works without URP. Do not assign a Scriptable Render Pipeline.
            UnityEngine.Rendering.GraphicsSettings.defaultRenderPipeline = null;
            QualitySettings.renderPipeline = null;
            RecreateAllMaterials();
            Debug.Log("Materials rebuilt with Unlit/Color. Pink objects should turn colored.");
        }

        private static void RecreateAllMaterials()
        {
            CreateColorMaterial("FloorMat", new Color(0.45f, 0.48f, 0.52f), overwrite: true);
            CreateColorMaterial("WallMat", new Color(0.55f, 0.7f, 0.95f, 0.12f), transparent: true, overwrite: true);
            CreateColorMaterial("SugarMat", new Color(0.95f, 0.75f, 0.2f), overwrite: true);
            CreateColorMaterial("BitterMat", new Color(0.55f, 0.2f, 0.75f), overwrite: true);
            CreateColorMaterial("FlyMat", new Color(0.12f, 0.12f, 0.12f), overwrite: true);
            CreateColorMaterial("FlyAccentMat", new Color(1f, 0.4f, 0.05f), overwrite: true);
            CreateColorMaterial("OutlineMat", new Color(0.1f, 1f, 1f), overwrite: true);
            CreateColorMaterial("VisualTargetMat", new Color(0.1f, 0.95f, 0.35f), overwrite: true);
            CreateColorMaterial("ObstacleMat", new Color(0.18f, 0.16f, 0.14f), overwrite: true);
            CreateColorMaterial("FloorMarkMat", new Color(0.72f, 0.74f, 0.78f), overwrite: true);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        private static GameObject BuildFly(
            ExperimentConfiguration config,
            ChemicalFieldSystem field,
            Material flyMat,
            Material accentMat)
        {
            var fly = new GameObject("FlyAgent");
            TrySetTag(fly, "Fly");
            fly.transform.position = config.FlySpawnPosition;
            fly.transform.localScale = Vector3.one * config.BodyScale;

            var bodyRb = fly.AddComponent<Rigidbody>();
            bodyRb.mass = 0.02f;
            bodyRb.linearDamping = 0.2f;
            bodyRb.angularDamping = 4f;
            bodyRb.useGravity = false;
            // Phase 2 hybrid: yaw only — pitch/roll locked so the body cannot tip through the floor.
            bodyRb.constraints =
                RigidbodyConstraints.FreezeRotationX |
                RigidbodyConstraints.FreezeRotationZ;
            // Sphere under the thorax is the ground contact; legs remain visual + sensing.
            const float groundRadius = 0.005f;
            var ground = fly.AddComponent<SphereCollider>();
            ground.radius = groundRadius;
            ground.center = Vector3.zero;

            var thorax = CreateSegment(fly.transform, "Thorax", Vector3.zero, 0.012f, flyMat);
            var head = CreateSegment(fly.transform, "Head", new Vector3(0f, 0f, 0.01f), 0.009f, accentMat);
            var abdomen = CreateSegment(fly.transform, "Abdomen", new Vector3(0f, 0f, -0.012f), 0.011f, flyMat);

            // Mount legs on an unscaled child. Parenting under Thorax (scaled ~0.012)
            // previously crushed leg meshes to near-invisible size.
            var legMount = new GameObject("Legs");
            legMount.transform.SetParent(fly.transform, false);

            var legs = new LegController[6];
            var feet = new FootContactSensor[6];
            Vector3[] legOffsets =
            {
                new Vector3(-0.006f, -0.001f, 0.006f),
                new Vector3(-0.007f, -0.001f, 0f),
                new Vector3(-0.006f, -0.001f, -0.006f),
                new Vector3(0.006f, -0.001f, 0.006f),
                new Vector3(0.007f, -0.001f, 0f),
                new Vector3(0.006f, -0.001f, -0.006f)
            };

            for (int i = 0; i < 6; i++)
            {
                legs[i] = BuildLeg(
                    legMount.transform,
                    (LegController.LegId)i,
                    legOffsets[i],
                    out feet[i],
                    flyMat,
                    accentMat);
            }

            // Plant the physics body near the floor, then raise so visual feet sit on the surface.
            float floorTop = ChamberController.FloorTopY(config.ChamberSize);
            float standY = floorTop + groundRadius * config.BodyScale;
            fly.transform.position = new Vector3(0f, standY, 0f);
            SnapFeetToFloor(fly, feet, floorTop);
            config.FlySpawnPosition = fly.transform.position;

            var leftEyeGo = new GameObject("LeftEye");
            leftEyeGo.transform.SetParent(head.transform, false);
            leftEyeGo.transform.localPosition = new Vector3(-0.002f, 0.002f, 0.004f);
            var leftCam = leftEyeGo.AddComponent<Camera>();
            leftCam.enabled = false; // RenderTexture capture only; keep OverviewCamera as Game view
            leftCam.depth = -10;
            var leftEye = leftEyeGo.AddComponent<FlyEyeCamera>();
            SetPrivate(leftEye, "isLeftEye", true);
            SetPrivate(leftEye, "grayscale", false);

            var rightEyeGo = new GameObject("RightEye");
            rightEyeGo.transform.SetParent(head.transform, false);
            rightEyeGo.transform.localPosition = new Vector3(0.002f, 0.002f, 0.004f);
            var rightCam = rightEyeGo.AddComponent<Camera>();
            rightCam.enabled = false;
            rightCam.depth = -10;
            var rightEye = rightEyeGo.AddComponent<FlyEyeCamera>();
            SetPrivate(rightEye, "isLeftEye", false);
            SetPrivate(rightEye, "grayscale", false);

            var leftAntennaGo = new GameObject("LeftAntenna");
            leftAntennaGo.transform.SetParent(head.transform, false);
            leftAntennaGo.transform.localPosition = new Vector3(-0.0015f, 0.003f, 0.005f);
            var leftAntenna = leftAntennaGo.AddComponent<AntennaOdorSensor>();
            leftAntenna.BindField(field);

            var rightAntennaGo = new GameObject("RightAntenna");
            rightAntennaGo.transform.SetParent(head.transform, false);
            rightAntennaGo.transform.localPosition = new Vector3(0.0015f, 0.003f, 0.005f);
            var rightAntenna = rightAntennaGo.AddComponent<AntennaOdorSensor>();
            rightAntenna.BindField(field);

            var proboscisGo = new GameObject("Proboscis");
            proboscisGo.transform.SetParent(head.transform, false);
            var tip = new GameObject("Tip");
            tip.transform.SetParent(proboscisGo.transform, false);
            var tipCollider = tip.AddComponent<SphereCollider>();
            tipCollider.isTrigger = true;
            tipCollider.radius = 0.002f;
            tip.AddComponent<Rigidbody>().isKinematic = true;
            var proboscis = proboscisGo.AddComponent<ProboscisSensor>();
            SetPrivate(proboscis, "tip", tip.transform);

            var body = fly.AddComponent<FlyBodyController>();
            SetPrivate(body, "rootBody", bodyRb);
            SetPrivate(body, "head", head.transform);
            SetPrivate(body, "thorax", thorax.transform);
            SetPrivate(body, "abdomen", abdomen.transform);
            SetPrivate(body, "legs", legs);

            var balance = fly.AddComponent<BalanceController>();
            balance.Bind(bodyRb);
            balance.TargetHeight = config.FlySpawnPosition.y;
            body.BindBalance(balance);
            body.Configure(config);

            var homeostasis = fly.AddComponent<FlyHomeostasis>();
            homeostasis.Configure(config);

            var vision = fly.AddComponent<VisionFrameProvider>();
            vision.Bind(leftEye, rightEye, config.VisionHz);

            var sensorManager = fly.AddComponent<SensorManager>();
            sensorManager.Bind(body, homeostasis, feet, leftAntenna, rightAntenna, proboscis, vision);

            var observationCollector = fly.AddComponent<FlyObservationCollector>();
            observationCollector.Bind(sensorManager);

            var proboscisActuator = fly.AddComponent<ProboscisActuator>();
            proboscisActuator.Bind(proboscis);

            var motors = fly.AddComponent<FlyMotorController>();
            motors.Bind(body, proboscisActuator);

            var demo = fly.AddComponent<WalkingDemoDriver>();
            demo.Bind(motors);
            demo.enabled = false; // live mind default; enable + Use Walking Demo for Phase 2 demo

            var agent = fly.AddComponent<FlyAgent>();
            SetPrivate(agent, "body", body);
            SetPrivate(agent, "homeostasis", homeostasis);
            SetPrivate(agent, "motors", motors);
            SetPrivate(agent, "observations", observationCollector);
            SetPrivate(agent, "useWalkingDemo", false);
            fly.AddComponent<FlyEyeMlAgentsBridge>();
            ConfigureMlAgents(fly);

            return fly;
        }

        /// <summary>
        /// Configures BehaviorParameters + DecisionRequester via SerializedObject so values
        /// persist into saved scenes and player builds.
        /// </summary>
        public static void ConfigureMlAgents(GameObject fly)
        {
            Type behaviorType = FindType(
                "Unity.MLAgents.Policies.BehaviorParameters",
                "Unity.ML-Agents");
            Type decisionType = FindType(
                "Unity.MLAgents.DecisionRequester",
                "Unity.ML-Agents");

            if (behaviorType == null)
            {
                Debug.LogError(
                    "BehaviorParameters type not found. Is com.unity.ml-agents installed?");
                return;
            }

            var behavior = fly.GetComponent(behaviorType) ?? fly.AddComponent(behaviorType);
            var behaviorSo = new SerializedObject((UnityEngine.Object)behavior);
            behaviorSo.Update();
            behaviorSo.FindProperty("m_BehaviorName").stringValue = "FlyAgent";
            SerializedProperty brain = behaviorSo.FindProperty("m_BrainParameters");
            brain.FindPropertyRelative("VectorObservationSize").intValue =
                SensorManager.ExpectedVectorSize;
            brain.FindPropertyRelative("NumStackedVectorObservations").intValue = 1;
            SerializedProperty actionSpec = brain.FindPropertyRelative("m_ActionSpec");
            actionSpec.FindPropertyRelative("m_NumContinuousActions").intValue = 5;
            SerializedProperty branchSizes = actionSpec.FindPropertyRelative("BranchSizes");
            if (branchSizes != null)
            {
                branchSizes.arraySize = 0;
            }

            SerializedProperty behaviorTypeProp = behaviorSo.FindProperty("m_BehaviorType");
            if (behaviorTypeProp != null)
            {
                behaviorTypeProp.enumValueIndex = 0; // Default
            }

            behaviorSo.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty((UnityEngine.Object)behavior);

            if (decisionType == null)
            {
                Debug.LogError(
                    "DecisionRequester type not found — Python reset/step will hang without it.");
            }
            else
            {
                var decision = fly.GetComponent(decisionType) ?? fly.AddComponent(decisionType);
                var decisionSo = new SerializedObject((UnityEngine.Object)decision);
                decisionSo.Update();
                decisionSo.FindProperty("DecisionPeriod").intValue = 1;
                decisionSo.FindProperty("TakeActionsBetweenDecisions").boolValue = true;
                decisionSo.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty((UnityEngine.Object)decision);
            }

            EditorUtility.SetDirty(fly);
            Debug.Log(
                $"ML-Agents configured: behavior=FlyAgent obs={SensorManager.ExpectedVectorSize} " +
                $"actions=5 decisionRequester={(decisionType != null)}");
        }

        private static Type FindType(string fullName, string assemblyName)
        {
            Type direct = Type.GetType($"{fullName}, {assemblyName}");
            if (direct != null)
            {
                return direct;
            }

            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type type = assembly.GetType(fullName);
                if (type != null)
                {
                    return type;
                }
            }

            return null;
        }

        private static LegController BuildLeg(
            Transform parent,
            LegController.LegId id,
            Vector3 offset,
            out FootContactSensor footSensor,
            Material limbMat,
            Material accentMat)
        {
            const float coxaLen = 0.005f;
            const float femurLen = 0.014f;
            const float tibiaLen = 0.014f;
            const float limbRadius = 0.0022f;

            var legRoot = new GameObject(id.ToString());
            legRoot.transform.SetParent(parent, false);
            legRoot.transform.localPosition = offset;

            // Aim hips outward/down so standing pose plants feet on the floor.
            bool left = id <= LegController.LegId.LeftHind;
            float side = left ? -1f : 1f;
            legRoot.transform.localRotation = Quaternion.Euler(28f, 0f, side * 28f);

            var coxa = CreateJoint(legRoot.transform, "Coxa", Vector3.zero, coxaLen, limbRadius, limbMat);
            var femur = CreateJoint(coxa.transform, "Femur", new Vector3(0f, -coxaLen, 0f), femurLen, limbRadius, limbMat);
            var tibia = CreateJoint(femur.transform, "Tibia", new Vector3(0f, -femurLen, 0f), tibiaLen, limbRadius * 0.85f, limbMat);

            var foot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            foot.name = "Foot";
            foot.transform.SetParent(tibia.transform, false);
            foot.transform.localPosition = new Vector3(0f, -tibiaLen, 0f);
            foot.transform.localScale = Vector3.one * 0.005f;
            UnityEngine.Object.DestroyImmediate(foot.GetComponent<Collider>());
            var footCollider = foot.AddComponent<SphereCollider>();
            footCollider.radius = 0.5f;
            footCollider.isTrigger = true;
            ApplyMaterial(foot, accentMat);
            footSensor = foot.AddComponent<FootContactSensor>();

            var controller = legRoot.AddComponent<LegController>();
            SetPrivate(controller, "legId", id);
            SetPrivate(controller, "coxa", coxa.GetComponent<LegJoint>());
            SetPrivate(controller, "femur", femur.GetComponent<LegJoint>());
            SetPrivate(controller, "tibia", tibia.GetComponent<LegJoint>());
            SetPrivate(controller, "foot", foot.transform);
            controller.ApplyStandingPose();
            return controller;
        }

        private static GameObject CreateJoint(
            Transform parent,
            string name,
            Vector3 localPosition,
            float length,
            float radius,
            Material material)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            go.AddComponent<LegJoint>();
            AddLimbMesh(go.transform, length, radius, material);
            return go;
        }

        private static void AddLimbMesh(Transform joint, float length, float radius, Material material)
        {
            var mesh = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            mesh.name = "Mesh";
            mesh.transform.SetParent(joint, false);
            // Default capsule height=2, radius=0.5 → scale Y = length/2, XZ = radius*2.
            mesh.transform.localPosition = new Vector3(0f, -length * 0.5f, 0f);
            mesh.transform.localScale = new Vector3(radius * 2f, length * 0.5f, radius * 2f);
            UnityEngine.Object.DestroyImmediate(mesh.GetComponent<Collider>());
            ApplyMaterial(mesh, material);
        }

        private static GameObject CreateSegment(
            Transform parent,
            string name,
            Vector3 localPosition,
            float scale,
            Material material)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            go.transform.localScale = Vector3.one * scale;
            UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
            ApplyMaterial(go, material);
            return go;
        }

        private static GameObject CreateCube(string name, Transform parent, Material material)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            ApplyMaterial(go, material);
            EnsureSurface(go, ChemicalChannels.Zero);
            return go;
        }

        private static void CreateChamberOutline(Transform parent, Vector3 size, Material material)
        {
            var outline = new GameObject("ChamberOutline");
            outline.transform.SetParent(parent, false);
            var line = outline.AddComponent<LineRenderer>();
            line.useWorldSpace = false;
            line.loop = false;
            line.widthMultiplier = 0.02f;
            line.positionCount = 16;
            line.sharedMaterial = material;
            line.startColor = Color.cyan;
            line.endColor = Color.cyan;

            float hx = size.x * 0.5f;
            float hy = size.y;
            float hz = size.z * 0.5f;
            Vector3[] corners =
            {
                new Vector3(-hx, 0f, -hz), new Vector3(hx, 0f, -hz),
                new Vector3(hx, 0f, hz), new Vector3(-hx, 0f, hz),
                new Vector3(-hx, 0f, -hz), new Vector3(-hx, hy, -hz),
                new Vector3(hx, hy, -hz), new Vector3(hx, 0f, -hz),
                new Vector3(hx, hy, -hz), new Vector3(hx, hy, hz),
                new Vector3(hx, 0f, hz), new Vector3(hx, hy, hz),
                new Vector3(-hx, hy, hz), new Vector3(-hx, 0f, hz),
                new Vector3(-hx, hy, hz), new Vector3(-hx, hy, -hz)
            };
            line.SetPositions(corners);
        }

        private static Material CreateColorMaterial(
            string name,
            Color color,
            bool transparent = false,
            bool overwrite = false)
        {
            DirectoryEnsure("Assets/Fly/Materials");
            string path = $"Assets/Fly/Materials/{name}.mat";
            Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material != null && !overwrite)
            {
                ApplyColorToMaterial(material, color, transparent);
                return material;
            }

            // Prefer built-in Unlit so pink never appears when GraphicsSettings has no SRP.
            // Transparent walls need a real alpha shader — Unlit/Color ignores alpha.
            Shader shader;
            if (transparent)
            {
                shader =
                    Shader.Find("Fly/UnlitColorTransparent") ??
                    Shader.Find("Sprites/Default") ??
                    Shader.Find("Legacy Shaders/Transparent/Diffuse") ??
                    Shader.Find("Unlit/Color");
            }
            else
            {
                shader =
                    Shader.Find("Unlit/Color") ??
                    Shader.Find("Sprites/Default") ??
                    Shader.Find("Legacy Shaders/Diffuse") ??
                    Shader.Find("Standard");
            }

            if (shader == null)
            {
                throw new InvalidOperationException("No usable built-in shader found for chamber materials.");
            }

            if (material == null)
            {
                material = new Material(shader) { name = name };
                AssetDatabase.CreateAsset(material, path);
            }
            else
            {
                material.shader = shader;
            }

            ApplyColorToMaterial(material, color, transparent);
            EditorUtility.SetDirty(material);
            return material;
        }

        private static void ApplyColorToMaterial(Material material, Color color, bool transparent)
        {
            if (material.HasProperty("_Color"))
            {
                material.SetColor("_Color", color);
            }

            material.color = color;

            if (transparent)
            {
                material.SetOverrideTag("RenderType", "Transparent");
                material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                if (material.HasProperty("_MainTex"))
                {
                    // Sprites/Default / Unlit/Transparent sample texture alpha; keep white.
                    material.SetTexture("_MainTex", Texture2D.whiteTexture);
                }
            }
        }

        private static void SetRendererEnabled(Transform target, bool enabled)
        {
            if (target == null)
            {
                return;
            }

            var renderer = target.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.enabled = enabled;
            }
        }

        private static void ApplyMaterial(GameObject go, Material material)
        {
            var renderer = go.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = material;
            }
        }

        private static void SnapFeetToFloor(GameObject fly, FootContactSensor[] feet, float floorTopY)
        {
            float lowest = float.MaxValue;
            float footRadiusWorld = 0f;
            for (int i = 0; i < feet.Length; i++)
            {
                if (feet[i] == null)
                {
                    continue;
                }

                Transform foot = feet[i].transform;
                lowest = Mathf.Min(lowest, foot.position.y);
                // Primitive sphere mesh radius 0.5, scaled by lossyScale.
                footRadiusWorld = Mathf.Max(footRadiusWorld, 0.5f * foot.lossyScale.y);
            }

            if (lowest > 1e5f)
            {
                return;
            }

            float targetLowest = floorTopY + footRadiusWorld;
            fly.transform.position += Vector3.up * (targetLowest - lowest);
        }

        private static void FrameSceneViewOnChamber(Vector3 chamberSize)
        {
            if (SceneView.lastActiveSceneView == null)
            {
                return;
            }

            var bounds = new Bounds(new Vector3(0f, chamberSize.y * 0.5f, 0f), chamberSize * 1.2f);
            SceneView.lastActiveSceneView.Frame(bounds, false);
            SceneView.lastActiveSceneView.Repaint();
        }

        private static void EnsureTags(params string[] tags)
        {
            var tagManager = new SerializedObject(
                AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            SerializedProperty tagsProp = tagManager.FindProperty("tags");
            var existing = new HashSet<string>();
            for (int i = 0; i < tagsProp.arraySize; i++)
            {
                existing.Add(tagsProp.GetArrayElementAtIndex(i).stringValue);
            }

            foreach (string tag in tags)
            {
                if (existing.Contains(tag))
                {
                    continue;
                }

                tagsProp.InsertArrayElementAtIndex(tagsProp.arraySize);
                tagsProp.GetArrayElementAtIndex(tagsProp.arraySize - 1).stringValue = tag;
            }

            tagManager.ApplyModifiedProperties();
        }

        private static void TrySetTag(GameObject go, string tag)
        {
            try
            {
                go.tag = tag;
            }
            catch (UnityException ex)
            {
                Debug.LogWarning($"Could not set tag '{tag}' on {go.name}: {ex.Message}");
            }
        }

        private static void EnsureSurface(GameObject go, ChemicalChannels channels)
        {
            var surface = go.GetComponent<SurfaceChemistry>() ?? go.AddComponent<SurfaceChemistry>();
            surface.SetChannels(channels);
        }

        private static void SetPrivate(object target, string fieldName, object value)
        {
            var flags = System.Reflection.BindingFlags.Instance |
                        System.Reflection.BindingFlags.NonPublic |
                        System.Reflection.BindingFlags.Public;
            var field = target.GetType().GetField(fieldName, flags);
            if (field == null)
            {
                Debug.LogWarning($"Field {fieldName} not found on {target.GetType().Name}");
                return;
            }

            field.SetValue(target, value);
        }

        private static void DirectoryEnsure(string path)
        {
            if (!AssetDatabase.IsValidFolder(path))
            {
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
}
