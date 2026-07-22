using System;
using UnityEngine;

namespace FlySimulation.Experiments
{
    /// <summary>
    /// Serializable experiment-controlled environment and episode parameters.
    /// </summary>
    [Serializable]
    public sealed class ExperimentConfiguration
    {
        [Header("Identity")]
        public string ExperimentId = "experiment_000001";
        public string FlyId = "fly_001";
        public int RandomSeed = 42;

        [Header("Timing")]
        public float EpisodeDurationSeconds = 120f;
        public float PhysicsHz = 200f;
        public float DecisionHz = 20f;
        public float VisionHz = 30f;
        public float RecordingHz = 50f;

        [Header("Chamber")]
        // ~10× smaller than the original 1×0.5×1 m arena.
        public Vector3 ChamberSize = new Vector3(0.1f, 0.05f, 0.1f);
        public float LightIntensity = 1f;
        public float TemperatureCelsius = 25f;
        public float Humidity = 0.5f;
        public Vector3 WindDirection = Vector3.forward;
        public float WindStrength = 0f;

        [Header("Chemistry")]
        public Vector3 SugarSourcePosition = new Vector3(0.03f, 0.002f, 0.03f);
        public Vector3 BitterSourcePosition = new Vector3(-0.03f, 0.002f, -0.03f);
        public float SugarStrength = 1f;
        public float BitterStrength = 1f;
        public float ChemicalSpread = 0.025f;

        [Header("Obstacles")]
        public Vector3[] ObstaclePositions = Array.Empty<Vector3>();

        [Header("Fly spawn")]
        // BodyScale 1 (was 10). Y is refined at scene build so feet rest on the floor.
        public Vector3 FlySpawnPosition = new Vector3(0f, 0.02f, 0f);
        public Vector3 FlySpawnEuler = Vector3.zero;
        public float BodyScale = 1f;

        [Header("Locomotion")]
        [Tooltip("Phase 2: unlock rotation and enable balance + walking forces.")]
        public bool LocomotionEnabled = true;

        [Header("Homeostasis")]
        public float InitialEnergy = 1f;
        public float InitialHydration = 1f;
        public float MetabolismRate = 0.002f;
        public float MovementEnergyCost = 0.01f;
        public float WingEnergyCost = 0.05f;

        [Header("Rewards")]
        public float CuriosityWeight = 0.1f;
        public float EnergyWeight = 1f;
        public float HydrationWeight = 1f;
        public float DamageWeight = 2f;
        public float EffortWeight = 0.05f;
        public float InstabilityWeight = 0.5f;

        public void ApplySeed()
        {
            UnityEngine.Random.InitState(RandomSeed);
        }

        public static ExperimentConfiguration Default()
        {
            return new ExperimentConfiguration();
        }

        public string ToJson(bool pretty = true)
        {
            return JsonUtility.ToJson(this, pretty);
        }

        public static ExperimentConfiguration FromJson(string json)
        {
            return JsonUtility.FromJson<ExperimentConfiguration>(json);
        }
    }
}
