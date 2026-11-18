using FlySimulation.Fly.Body;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>Ground-truth vision labels for one head pose (training data only, never fed to the fly).</summary>
    public readonly struct EyeRayLabel
    {
        public EyeRayLabel(
            float[] distances,
            float wallHazard,
            float forwardBias,
            float turnBias,
            float openingBearing,
            float openingStrength)
        {
            Distances = distances;
            WallHazard = wallHazard;
            ForwardBias = forwardBias;
            TurnBias = turnBias;
            OpeningBearing = openingBearing;
            OpeningStrength = openingStrength;
        }

        /// <summary>Wall distance per column, left to right; <see cref="EyeRayLabels.OpenDistance"/> = open.</summary>
        public float[] Distances { get; }

        /// <summary>0 = no wall within <see cref="EyeRayLabels.HazardRange"/> ahead, 1 = touching.</summary>
        public float WallHazard { get; }

        /// <summary>1 − 2·hazard: walk on when clear, back off when touching.</summary>
        public float ForwardBias { get; }

        /// <summary>Toward the most open column (−1 left … +1 right), scaled by hazard.</summary>
        public float TurnBias { get; }

        /// <summary>Mean bearing (−1 left … +1 right) of columns that see out of the box.</summary>
        public float OpeningBearing { get; }

        /// <summary>Fraction of columns that see out of the box.</summary>
        public float OpeningStrength { get; }
    }

    /// <summary>
    /// Horizontal fan of rays across the eyes' field of view against the wall layer. A column
    /// with no wall within <see cref="OpenDistance"/> (longer than the chamber diagonal) is
    /// looking out through the tunnel.
    /// </summary>
    public static class EyeRayLabels
    {
        public const int Columns = 16;
        public const float OpenDistance = 1f;
        // Near-field wall distance that maps to hazard=1. Larger ⇒ milder hazard at the
        // same clearance (grass-height clutter / approaching walls ramp up more slowly).
        public const float HazardRange = 0.10f;
        public const float CentralHalfAngle = 35f;

        public static float HalfFieldOfView => FlyEyeCamera.DefaultFieldOfView * 0.5f;

        /// <summary>Yaw offset in degrees of column <paramref name="index"/> (left = negative).</summary>
        public static float ColumnAngle(int index)
        {
            float t = Columns == 1 ? 0.5f : index / (float)(Columns - 1);
            return Mathf.Lerp(-HalfFieldOfView, HalfFieldOfView, t);
        }

        public static EyeRayLabel Compute(Vector3 eyeCenter, Vector3 forward)
        {
            forward.y = 0f;
            forward = forward.sqrMagnitude > 1e-9f ? forward.normalized : Vector3.forward;
            var distances = new float[Columns];
            for (int i = 0; i < Columns; i++)
            {
                Vector3 dir = Quaternion.Euler(0f, ColumnAngle(i), 0f) * forward;
                distances[i] = Physics.Raycast(
                    eyeCenter, dir, out RaycastHit hit, OpenDistance, BodyClearance.WallMask, QueryTriggerInteraction.Ignore)
                    ? hit.distance
                    : OpenDistance;
            }

            return FromDistances(distances);
        }

        public static EyeRayLabel FromDistances(float[] distances)
        {
            float nearestAhead = OpenDistance;
            float farthest = 0f;
            float openSum = 0f;
            int openCount = 0;
            for (int i = 0; i < distances.Length; i++)
            {
                float angle = ColumnAngle(i);
                if (Mathf.Abs(angle) <= CentralHalfAngle)
                {
                    nearestAhead = Mathf.Min(nearestAhead, distances[i]);
                }

                farthest = Mathf.Max(farthest, distances[i]);
                if (distances[i] >= OpenDistance)
                {
                    openSum += angle / HalfFieldOfView;
                    openCount++;
                }
            }

            // Several equally free columns (e.g. all open) share one averaged bearing.
            float freeSum = 0f;
            int freeCount = 0;
            for (int i = 0; i < distances.Length; i++)
            {
                if (distances[i] >= farthest - 1e-4f)
                {
                    freeSum += ColumnAngle(i) / HalfFieldOfView;
                    freeCount++;
                }
            }

            float hazard = Mathf.Clamp01(1f - nearestAhead / HazardRange);
            float freeBearing = freeCount > 0 ? freeSum / freeCount : 0f;
            return new EyeRayLabel(
                distances,
                hazard,
                1f - 2f * hazard,
                Mathf.Clamp(freeBearing * Mathf.Clamp01(2f * hazard), -1f, 1f),
                openCount > 0 ? openSum / openCount : 0f,
                openCount / (float)distances.Length);
        }
    }
}
