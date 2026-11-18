using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Chamber walls, ceiling and tunnel walls are opaque to the fly's eyes (so the tunnel is the
    /// only view out) but hidden from the overview camera so Play mode always looks into the arena.
    /// They live on <see cref="WallLayer"/>, which only the overview camera culls.
    /// </summary>
    public sealed class ChamberWallHider : MonoBehaviour
    {
        /// <summary>TagManager "Environment" layer.</summary>
        public const int WallLayer = 9;

        public const string WallPrefix = "Wall_";
        public const string CeilingName = "Ceiling";

        /// <summary>Culling mask for the overview camera: everything except the chamber walls.</summary>
        public static int OverviewCullingMask => ~(1 << WallLayer);

        private void Awake()
        {
            HideWalls();
        }

        private void Start()
        {
            HideWalls();
        }

        public static bool IsWallSurface(string objectName)
        {
            return objectName == CeilingName ||
                   objectName.StartsWith(WallPrefix) ||
                   objectName.StartsWith(TunnelLayout.TunnelWallPrefix);
        }

        public static void HideWalls()
        {
            var chamber = FindFirstObjectByType<ChamberController>();
            Transform root = chamber != null ? chamber.transform : null;
            if (root == null)
            {
                var chamberGo = GameObject.Find("Chamber");
                root = chamberGo != null ? chamberGo.transform : null;
            }

            if (root != null)
            {
                ApplyWallLayer(root);
            }
        }

        /// <summary>Move every wall surface under <paramref name="root"/> onto the wall layer, rendered.</summary>
        public static void ApplyWallLayer(Transform root)
        {
            foreach (Transform child in root.GetComponentsInChildren<Transform>(includeInactive: true))
            {
                if (!IsWallSurface(child.name))
                {
                    continue;
                }

                child.gameObject.layer = WallLayer;
                var renderer = child.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    renderer.enabled = true;
                }
            }
        }
    }
}
