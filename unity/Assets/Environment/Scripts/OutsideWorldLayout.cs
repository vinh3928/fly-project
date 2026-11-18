using System.Collections.Generic;
using UnityEngine;

namespace FlySimulation.Environment
{
    public enum OutsideFeature
    {
        Flower,
        Tree,
        House
    }

    /// <summary>One outside-world object, standing on the ground at <see cref="Position"/> (chamber-local).</summary>
    public readonly struct OutsidePlacement
    {
        public readonly OutsideFeature Kind;
        public readonly Vector3 Position;
        public readonly float Height;
        public readonly float Width;
        public readonly float YawDegrees;
        public readonly int Variant;

        public OutsidePlacement(OutsideFeature kind, Vector3 position, float height, float width, float yawDegrees, int variant)
        {
            Kind = kind;
            Position = position;
            Height = height;
            Width = width;
            YawDegrees = yawDegrees;
            Variant = variant;
        }
    }

    /// <summary>A single grass blade: a tapered triangle from <see cref="Base"/> leaning toward its yaw.</summary>
    public readonly struct GrassBlade
    {
        public readonly Vector3 Base;
        public readonly float Height;
        public readonly float Width;
        public readonly float YawDegrees;
        public readonly float Lean;
        public readonly int Shade;

        public GrassBlade(Vector3 basePosition, float height, float width, float yawDegrees, float lean, int shade)
        {
            Base = basePosition;
            Height = height;
            Width = width;
            YawDegrees = yawDegrees;
            Lean = lean;
            Shade = shade;
        }
    }

    /// <summary>
    /// Deterministic layout of the outside world at the fly's real scale: a meadow of grass and
    /// flowers in front of the tunnel exit that towers over a 3 mm fly, with trees and a town
    /// as giant backdrops. All sizes are meters in chamber-local space.
    /// </summary>
    public static class OutsideWorldLayout
    {
        /// <summary>Camera far clip that keeps the town backdrop in view.</summary>
        public const float ViewDistance = 150f;

        /// <summary>Outside ground sits just under the chamber floor so the overlap never z-fights.</summary>
        public const float GroundDrop = 0.0005f;
        public const float GroundHalfExtent = 75f;
        public const float GroundThickness = 0.02f;

        public const float MeadowHalfWidth = 1.2f;
        public const float MeadowDepth = 2.5f;
        /// <summary>Bare ground right outside the tunnel mouth.</summary>
        public const float MeadowClearance = 0.03f;

        public const float GrassDensityPerSquareMeter = 500f;
        public const float GrassMinHeight = 0.03f;
        public const float GrassMaxHeight = 0.12f;
        public const float GrassMinWidth = 0.002f;
        public const float GrassMaxWidth = 0.005f;
        public const float GrassMaxLean = 0.35f;
        public const int GrassShades = 3;

        public const int FlowerCount = 36;
        public const float FlowerMinHeight = 0.12f;
        public const float FlowerMaxHeight = 0.35f;
        public const float FlowerMinHeadWidth = 0.025f;
        public const float FlowerMaxHeadWidth = 0.05f;
        public const int FlowerVariants = 5;

        public const int TreeCount = 16;
        public const float TreeMinDistance = 6f;
        /// <summary>Keeps trees (canopy included) short of the town's first row.</summary>
        public const float TreeMaxDistance = 30f;
        public const float TreeMinHeight = 5f;
        public const float TreeMaxHeight = 12f;
        public const int TreeVariants = 3;

        /// <summary>The town is a grid of houses straight out past the tunnel (+Z).</summary>
        public const int TownColumns = 3;
        public const int TownRows = 3;
        public const int HouseCount = TownColumns * TownRows;
        public const float TownNearestRowDistance = 40f;
        public const float TownLotSpacing = 14f;
        public const float HouseMinHeight = 5f;
        public const float HouseMaxHeight = 8f;
        public const float HouseMinWidth = 6f;
        public const float HouseMaxWidth = 10f;
        /// <summary>House depth relative to its width.</summary>
        public const float HouseDepthRatio = 0.8f;
        public const int HouseVariants = 4;

        public static float GroundTopY(Vector3 chamberSize)
        {
            return ChamberController.FloorTopY(chamberSize) - GroundDrop;
        }

        /// <summary>Meadow footprint in XZ (Rect.x/width = X, Rect.y/height = Z).</summary>
        public static Rect MeadowRect(float exitZ)
        {
            return new Rect(-MeadowHalfWidth, exitZ + MeadowClearance, MeadowHalfWidth * 2f, MeadowDepth);
        }

        public static List<OutsidePlacement> Generate(int seed, Vector3 chamberSize, float exitZ)
        {
            var rng = new System.Random(seed);
            float groundY = GroundTopY(chamberSize);
            var placements = new List<OutsidePlacement>(FlowerCount + TreeCount + HouseCount);
            Rect meadow = MeadowRect(exitZ);

            for (int i = 0; i < FlowerCount; i++)
            {
                placements.Add(new OutsidePlacement(
                    OutsideFeature.Flower,
                    RandomInRect(rng, meadow, groundY),
                    Range(rng, FlowerMinHeight, FlowerMaxHeight),
                    Range(rng, FlowerMinHeadWidth, FlowerMaxHeadWidth),
                    Range(rng, 0f, 360f),
                    rng.Next(FlowerVariants)));
            }

            for (int i = 0; i < TreeCount; i++)
            {
                placements.Add(new OutsidePlacement(
                    OutsideFeature.Tree,
                    OnRing(rng, groundY, TreeMinDistance, TreeMaxDistance, -180f, 180f),
                    Range(rng, TreeMinHeight, TreeMaxHeight),
                    0f,
                    Range(rng, 0f, 360f),
                    rng.Next(TreeVariants)));
            }

            // Jitter stays inside each lot so the widest house never touches its neighbor.
            float jitter = (TownLotSpacing - HouseMaxWidth) * 0.5f;
            for (int row = 0; row < TownRows; row++)
            {
                for (int column = 0; column < TownColumns; column++)
                {
                    float x = (column - (TownColumns - 1) * 0.5f) * TownLotSpacing + Range(rng, -jitter, jitter);
                    float z = TownNearestRowDistance + row * TownLotSpacing + Range(rng, -jitter, jitter);
                    placements.Add(new OutsidePlacement(
                        OutsideFeature.House,
                        new Vector3(x, groundY, z),
                        Range(rng, HouseMinHeight, HouseMaxHeight),
                        Range(rng, HouseMinWidth, HouseMaxWidth),
                        180f,
                        rng.Next(HouseVariants)));
                }
            }

            return placements;
        }

        public static List<GrassBlade> GrassBlades(int seed, Vector3 chamberSize, float exitZ)
        {
            var rng = new System.Random(seed ^ 0x5eed);
            float groundY = GroundTopY(chamberSize);
            Rect meadow = MeadowRect(exitZ);
            int count = Mathf.RoundToInt(meadow.width * meadow.height * GrassDensityPerSquareMeter);
            var blades = new List<GrassBlade>(count);
            for (int i = 0; i < count; i++)
            {
                blades.Add(new GrassBlade(
                    RandomInRect(rng, meadow, groundY),
                    Range(rng, GrassMinHeight, GrassMaxHeight),
                    Range(rng, GrassMinWidth, GrassMaxWidth),
                    Range(rng, 0f, 360f),
                    Range(rng, 0f, GrassMaxLean),
                    rng.Next(GrassShades)));
            }

            return blades;
        }

        private static float Range(System.Random rng, float min, float max)
        {
            return min + (float)rng.NextDouble() * (max - min);
        }

        private static Vector3 RandomInRect(System.Random rng, Rect rect, float y)
        {
            return new Vector3(Range(rng, rect.xMin, rect.xMax), y, Range(rng, rect.yMin, rect.yMax));
        }

        /// <summary>Random point at a distance band, at an angle measured from +Z toward +X.</summary>
        private static Vector3 OnRing(System.Random rng, float y, float minDistance, float maxDistance, float minAngle, float maxAngle)
        {
            float distance = Range(rng, minDistance, maxDistance);
            float angle = Range(rng, minAngle, maxAngle) * Mathf.Deg2Rad;
            return new Vector3(Mathf.Sin(angle) * distance, y, Mathf.Cos(angle) * distance);
        }
    }
}
