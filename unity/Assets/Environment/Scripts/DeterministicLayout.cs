using System;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Pure deterministic placement helpers for Phase 1 seed replay.
    /// Uses <see cref="System.Random"/> so results do not depend on UnityEngine.Random consumers.
    /// </summary>
    public static class DeterministicLayout
    {
        public static Vector3[] GenerateObstaclePositions(
            int seed,
            Vector3 chamberSize,
            int count,
            float placementMargin,
            float obstacleRadius)
        {
            if (count <= 0)
            {
                return Array.Empty<Vector3>();
            }

            var rng = new System.Random(seed);
            float halfX = Mathf.Max(0.01f, chamberSize.x * 0.5f - placementMargin);
            float halfZ = Mathf.Max(0.01f, chamberSize.z * 0.5f - placementMargin);
            var positions = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                positions[i] = new Vector3(
                    NextRange(rng, -halfX, halfX),
                    obstacleRadius,
                    NextRange(rng, -halfZ, halfZ));
            }

            return positions;
        }

        public static Vector3 GenerateVisualTargetPosition(
            int seed,
            Vector3 chamberSize,
            float placementMargin,
            float targetRadius)
        {
            var rng = new System.Random(seed ^ unchecked((int)0x5EEDFACE));
            float halfX = Mathf.Max(0.01f, chamberSize.x * 0.5f - placementMargin);
            float halfZ = Mathf.Max(0.01f, chamberSize.z * 0.5f - placementMargin);
            return new Vector3(
                NextRange(rng, -halfX, halfX),
                targetRadius,
                NextRange(rng, -halfZ, halfZ));
        }

        public static bool AreEqual(Vector3[] a, Vector3[] b, float epsilon = 1e-5f)
        {
            if (ReferenceEquals(a, b))
            {
                return true;
            }

            if (a == null || b == null || a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; i++)
            {
                if ((a[i] - b[i]).sqrMagnitude > epsilon * epsilon)
                {
                    return false;
                }
            }

            return true;
        }

        private static float NextRange(System.Random rng, float minInclusive, float maxInclusive)
        {
            double t = rng.NextDouble();
            return (float)(minInclusive + (maxInclusive - minInclusive) * t);
        }
    }
}
