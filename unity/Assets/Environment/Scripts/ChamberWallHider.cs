using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Runtime safety net: hide chamber wall meshes so Play mode always looks into the arena.
    /// </summary>
    public sealed class ChamberWallHider : MonoBehaviour
    {
        private void Awake()
        {
            HideWalls();
        }

        private void Start()
        {
            HideWalls();
        }

        public static void HideWalls()
        {
            for (int i = 0; i < 4; i++)
            {
                var wall = GameObject.Find($"Wall_{i}");
                if (wall == null)
                {
                    continue;
                }

                var renderer = wall.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    renderer.enabled = false;
                }
            }

            var ceiling = GameObject.Find("Ceiling");
            if (ceiling != null)
            {
                var renderer = ceiling.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    renderer.enabled = false;
                }
            }
        }
    }
}
