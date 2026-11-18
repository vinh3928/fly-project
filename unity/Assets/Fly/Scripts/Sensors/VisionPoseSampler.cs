using FlySimulation.Environment;
using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    public enum VisionPoseRegion
    {
        Interior,
        NearWall,
        FacingOpening,
        Tunnel,
        Outside,
    }

    /// <summary>Fly root pose in chamber-local XZ plus yaw (degrees, 0 = +Z).</summary>
    public readonly struct VisionPose
    {
        public VisionPose(VisionPoseRegion region, Vector2 positionXZ, float yawDegrees)
        {
            Region = region;
            PositionXZ = positionXZ;
            YawDegrees = yawDegrees;
        }

        public VisionPoseRegion Region { get; }
        public Vector2 PositionXZ { get; }
        public float YawDegrees { get; }
    }

    /// <summary>
    /// Deterministic random fly poses for the vision dataset: open interior, close to walls
    /// and facing them, facing the opening, inside the tunnel, and outside in the meadow.
    /// </summary>
    public sealed class VisionPoseSampler
    {
        /// <summary>Keeps the root this far from wall faces (head reach + clearance).</summary>
        public const float WallMargin = 0.016f;

        public const float NearWallMaxDistance = 0.06f;
        public const float FacingSpreadDegrees = 60f;
        public const float OutsideHalfWidth = 0.3f;
        public const float OutsideDepth = 0.4f;

        private static readonly (VisionPoseRegion region, float weight)[] Weights =
        {
            (VisionPoseRegion.Interior, 0.35f),
            (VisionPoseRegion.NearWall, 0.2f),
            (VisionPoseRegion.FacingOpening, 0.2f),
            (VisionPoseRegion.Tunnel, 0.1f),
            (VisionPoseRegion.Outside, 0.15f),
        };

        private readonly System.Random _rng;
        private readonly float _innerX;
        private readonly float _innerZ;
        private readonly float _halfOpening;
        private readonly float _exitZ;
        private readonly bool _tunnel;

        public VisionPoseSampler(ExperimentConfiguration config, int seed)
        {
            _rng = new System.Random(seed);
            Vector3 size = config.ChamberSize;
            float wall = ChamberController.WallThickness(size);
            _innerX = size.x * 0.5f - wall * 0.5f;
            _innerZ = size.z * 0.5f - wall * 0.5f;
            _halfOpening = TunnelLayout.OpeningWidth(config) * 0.5f;
            _exitZ = TunnelLayout.ExitZ(config);
            _tunnel = config.TunnelEnabled;
        }

        public VisionPose Next()
        {
            VisionPoseRegion region = PickRegion();
            if (!_tunnel && (region == VisionPoseRegion.Tunnel || region == VisionPoseRegion.Outside ||
                             region == VisionPoseRegion.FacingOpening))
            {
                region = VisionPoseRegion.Interior;
            }

            switch (region)
            {
                case VisionPoseRegion.NearWall:
                    return NearWall();
                case VisionPoseRegion.FacingOpening:
                {
                    Vector2 p = InteriorPoint();
                    float toOpening = YawToward(p, new Vector2(0f, _innerZ));
                    return new VisionPose(region, p, toOpening + Range(-FacingSpreadDegrees, FacingSpreadDegrees));
                }
                case VisionPoseRegion.Tunnel:
                    return new VisionPose(
                        region,
                        new Vector2(
                            Range(-_halfOpening + WallMargin, _halfOpening - WallMargin),
                            Range(_innerZ, _exitZ)),
                        Range(0f, 360f));
                case VisionPoseRegion.Outside:
                    return new VisionPose(
                        region,
                        new Vector2(
                            Range(-OutsideHalfWidth, OutsideHalfWidth),
                            Range(_exitZ + WallMargin, _exitZ + OutsideDepth)),
                        Range(0f, 360f));
                default:
                    return new VisionPose(VisionPoseRegion.Interior, InteriorPoint(), Range(0f, 360f));
            }
        }

        public static float YawToward(Vector2 from, Vector2 to)
        {
            Vector2 d = to - from;
            return Mathf.Atan2(d.x, d.y) * Mathf.Rad2Deg;
        }

        private VisionPose NearWall()
        {
            int side = _rng.Next(4);
            float gap = Range(WallMargin, NearWallMaxDistance);
            Vector2 p = InteriorPoint();
            float faceYaw;
            switch (side)
            {
                case 0:
                    p.y = _innerZ - gap;
                    faceYaw = 0f;
                    break;
                case 1:
                    p.x = _innerX - gap;
                    faceYaw = 90f;
                    break;
                case 2:
                    p.y = -_innerZ + gap;
                    faceYaw = 180f;
                    break;
                default:
                    p.x = -_innerX + gap;
                    faceYaw = -90f;
                    break;
            }

            return new VisionPose(
                VisionPoseRegion.NearWall, p, faceYaw + Range(-FacingSpreadDegrees, FacingSpreadDegrees));
        }

        private Vector2 InteriorPoint()
        {
            return new Vector2(
                Range(-_innerX + WallMargin, _innerX - WallMargin),
                Range(-_innerZ + WallMargin, _innerZ - WallMargin));
        }

        private VisionPoseRegion PickRegion()
        {
            float total = 0f;
            foreach (var (_, weight) in Weights)
            {
                total += weight;
            }

            float r = (float)_rng.NextDouble() * total;
            foreach (var (region, weight) in Weights)
            {
                r -= weight;
                if (r <= 0f)
                {
                    return region;
                }
            }

            return VisionPoseRegion.Interior;
        }

        private float Range(float min, float max)
        {
            return min + (float)_rng.NextDouble() * (max - min);
        }
    }
}
