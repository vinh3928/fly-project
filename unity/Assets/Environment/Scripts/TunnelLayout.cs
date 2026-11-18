using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>Axis-aligned box in chamber-local space: center plus full size (cube primitive scale).</summary>
    public readonly struct BoxSpec
    {
        public readonly string Name;
        public readonly Vector3 Center;
        public readonly Vector3 Size;

        public BoxSpec(string name, Vector3 center, Vector3 size)
        {
            Name = name;
            Center = center;
            Size = size;
        }

        public Vector3 Min => Center - Size * 0.5f;
        public Vector3 Max => Center + Size * 0.5f;

        /// <summary>False when any side is zero, e.g. no lintel above a full-height opening.</summary>
        public bool HasVolume => Size.x > 1e-6f && Size.y > 1e-6f && Size.z > 1e-6f;
    }

    /// <summary>
    /// Geometry of the exit tunnel through the +Z (north) wall: the wall split around the
    /// opening, and the short passage that leads out to the outside world.
    /// Every method reads the tunnel settings from the configuration and the chamber size
    /// from <c>chamberSize</c> (defaults to <see cref="ExperimentConfiguration.ChamberSize"/>).
    /// </summary>
    public static class TunnelLayout
    {
        public const string NorthWallLeft = "Wall_0_Left";
        public const string NorthWallRight = "Wall_0_Right";
        public const string NorthWallLintel = "Wall_0_Lintel";
        public const string TunnelFloor = "TunnelFloor";
        public const string TunnelWallPrefix = "TunnelWall_";
        public const string TunnelWallLeft = TunnelWallPrefix + "Left";
        public const string TunnelWallRight = TunnelWallPrefix + "Right";
        public const string TunnelRoof = TunnelWallPrefix + "Roof";

        public static float OpeningWidth(ExperimentConfiguration config, Vector3? chamberSize = null)
        {
            Vector3 size = chamberSize ?? config.ChamberSize;
            return Mathf.Clamp01(config.TunnelWidthFraction) * size.x;
        }

        public static float OpeningHeight(ExperimentConfiguration config, Vector3? chamberSize = null)
        {
            Vector3 size = chamberSize ?? config.ChamberSize;
            float maxHeight = size.y - ChamberController.FloorTopY(size);
            return config.TunnelFullHeight ? maxHeight : Mathf.Clamp(config.TunnelHeight, 0f, maxHeight);
        }

        /// <summary>Z of the north wall's outside face, where the tunnel passage starts.</summary>
        public static float OuterWallZ(Vector3 chamberSize)
        {
            return chamberSize.z * 0.5f + ChamberController.WallThickness(chamberSize) * 0.5f;
        }

        /// <summary>Z of the tunnel mouth that opens onto the outside world.</summary>
        public static float ExitZ(ExperimentConfiguration config, Vector3? chamberSize = null)
        {
            Vector3 size = chamberSize ?? config.ChamberSize;
            return OuterWallZ(size) + Mathf.Max(0f, config.TunnelLength);
        }

        /// <summary>The north wall with the opening cut out: left, right and lintel above the opening.</summary>
        public static BoxSpec[] NorthWallSegments(ExperimentConfiguration config, Vector3? chamberSize = null)
        {
            Vector3 size = chamberSize ?? config.ChamberSize;
            float wall = ChamberController.WallThickness(size);
            float hx = size.x * 0.5f;
            float hz = size.z * 0.5f;
            float halfOpening = OpeningWidth(config, size) * 0.5f;
            float openingTop = ChamberController.FloorTopY(size) + OpeningHeight(config, size);
            float sideWidth = hx - halfOpening;
            float sideCenterX = (hx + halfOpening) * 0.5f;

            return new[]
            {
                new BoxSpec(NorthWallLeft, new Vector3(-sideCenterX, size.y * 0.5f, hz), new Vector3(sideWidth, size.y, wall)),
                new BoxSpec(NorthWallRight, new Vector3(sideCenterX, size.y * 0.5f, hz), new Vector3(sideWidth, size.y, wall)),
                new BoxSpec(
                    NorthWallLintel,
                    new Vector3(0f, (openingTop + size.y) * 0.5f, hz),
                    new Vector3(halfOpening * 2f, size.y - openingTop, wall)),
            };
        }

        /// <summary>Passage from the opening to the exit: floor slab, two side walls and a roof.</summary>
        public static BoxSpec[] TunnelPieces(ExperimentConfiguration config, Vector3? chamberSize = null)
        {
            Vector3 size = chamberSize ?? config.ChamberSize;
            float wall = ChamberController.WallThickness(size);
            float slab = ChamberController.FloorThickness(size);
            float floorTop = ChamberController.FloorTopY(size);
            float halfOpening = OpeningWidth(config, size) * 0.5f;
            float height = OpeningHeight(config, size);
            float outer = OuterWallZ(size);
            float exit = ExitZ(config, size);
            float length = exit - outer;
            float midZ = (outer + exit) * 0.5f;
            float outerWidth = halfOpening * 2f + wall * 2f;
            float sideX = halfOpening + wall * 0.5f;

            // The floor starts at the wall's center line, where the chamber floor slab ends.
            float floorStart = size.z * 0.5f;
            return new[]
            {
                new BoxSpec(
                    TunnelFloor,
                    new Vector3(0f, 0f, (floorStart + exit) * 0.5f),
                    new Vector3(outerWidth, slab, exit - floorStart)),
                new BoxSpec(TunnelWallLeft, new Vector3(-sideX, floorTop + height * 0.5f, midZ), new Vector3(wall, height, length)),
                new BoxSpec(TunnelWallRight, new Vector3(sideX, floorTop + height * 0.5f, midZ), new Vector3(wall, height, length)),
                new BoxSpec(TunnelRoof, new Vector3(0f, floorTop + height + wall * 0.5f, midZ), new Vector3(outerWidth, wall, length)),
            };
        }

        /// <summary>
        /// Position and scale cube transforms to match the specs (same order, same count).
        /// Pieces without volume are deactivated so no zero-size collider is left behind.
        /// </summary>
        public static void Apply(Transform[] targets, BoxSpec[] specs)
        {
            if (targets == null)
            {
                return;
            }

            int count = Mathf.Min(targets.Length, specs.Length);
            for (int i = 0; i < count; i++)
            {
                if (targets[i] == null)
                {
                    continue;
                }

                targets[i].localPosition = specs[i].Center;
                targets[i].gameObject.SetActive(specs[i].HasVolume);
                if (specs[i].HasVolume)
                {
                    targets[i].localScale = specs[i].Size;
                }
            }
        }
    }
}
