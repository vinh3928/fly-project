using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Applies deterministic randomization to obstacle and optional source jitter.
    /// </summary>
    public sealed class EnvironmentRandomizer : MonoBehaviour
    {
        [SerializeField] private Transform obstacleRoot;
        [SerializeField] private GameObject obstaclePrefab;
        [SerializeField] private Material obstacleMaterial;
        [Tooltip("0 = no obstacles (recommended for clear Play-mode view).")]
        [SerializeField] private int obstacleCount = 0;
        [SerializeField] private float obstacleRadius = 0.005f;
        [SerializeField] private float placementMargin = 0.015f;

        private readonly System.Collections.Generic.List<GameObject> _spawned =
            new System.Collections.Generic.List<GameObject>();

        public void Randomize(ExperimentConfiguration config)
        {
            ClearObstacles();
            config.ApplySeed();

            float extent = Mathf.Max(config.ChamberSize.x, config.ChamberSize.z);
            float radius = Mathf.Min(obstacleRadius, extent * 0.08f);
            float margin = Mathf.Max(placementMargin, radius * 2f);

            Vector3[] positions;
            if (config.ObstaclePositions != null && config.ObstaclePositions.Length > 0)
            {
                positions = config.ObstaclePositions;
            }
            else if (obstacleCount <= 0)
            {
                positions = System.Array.Empty<Vector3>();
            }
            else
            {
                positions = DeterministicLayout.GenerateObstaclePositions(
                    config.RandomSeed,
                    config.ChamberSize,
                    obstacleCount,
                    margin,
                    radius);
            }

            for (int i = 0; i < positions.Length; i++)
            {
                SpawnObstacle(positions[i], radius);
            }

            config.ObstaclePositions = positions;
        }

        private void SpawnObstacle(Vector3 localPosition, float radius)
        {
            GameObject instance;
            if (obstaclePrefab != null)
            {
                instance = Instantiate(obstaclePrefab, obstacleRoot != null ? obstacleRoot : transform);
            }
            else
            {
                instance = GameObject.CreatePrimitive(PrimitiveType.Cube);
                instance.transform.SetParent(obstacleRoot != null ? obstacleRoot : transform, false);
                instance.transform.localScale = Vector3.one * (radius * 2f);
                TrySetTag(instance, "Obstacle");
                ApplyObstacleMaterial(instance);
            }

            instance.transform.localPosition = localPosition;
            _spawned.Add(instance);
        }

        private void ApplyObstacleMaterial(GameObject instance)
        {
            var renderer = instance.GetComponent<MeshRenderer>();
            if (renderer == null)
            {
                return;
            }

            if (obstacleMaterial != null)
            {
                renderer.sharedMaterial = obstacleMaterial;
                return;
            }

            // Avoid Unity's default white/grey primitive material.
            var mat = new Material(Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default"));
            mat.color = new Color(0.25f, 0.28f, 0.32f, 1f);
            renderer.sharedMaterial = mat;
        }

        private static void TrySetTag(GameObject go, string tag)
        {
            try
            {
                go.tag = tag;
            }
            catch (UnityException)
            {
                // Tag may not exist in a fresh project; ignore.
            }
        }

        public void ClearObstacles()
        {
            for (int i = 0; i < _spawned.Count; i++)
            {
                if (_spawned[i] == null)
                {
                    continue;
                }

                if (Application.isPlaying)
                {
                    Destroy(_spawned[i]);
                }
                else
                {
                    DestroyImmediate(_spawned[i]);
                }
            }

            _spawned.Clear();
        }
    }
}
