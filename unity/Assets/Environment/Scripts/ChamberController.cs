using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Builds and configures the controlled chamber from experiment settings.
    /// </summary>
    public sealed class ChamberController : MonoBehaviour
    {
        [SerializeField] private ExperimentConfiguration configuration;
        [SerializeField] private Transform floor;
        [SerializeField] private Transform ceiling;
        [SerializeField] private Transform[] walls = new Transform[4];
        [Tooltip("North wall split around the tunnel opening (TunnelLayout.NorthWallSegments order).")]
        [SerializeField] private Transform[] northWallSegments = new Transform[0];
        [Tooltip("Tunnel floor, side walls and roof (TunnelLayout.TunnelPieces order).")]
        [SerializeField] private Transform[] tunnelPieces = new Transform[0];
        [SerializeField] private Light mainLight;
        [SerializeField] private ChemicalSource sugarSource;
        [SerializeField] private ChemicalSource bitterSource;
        [SerializeField] private ChemicalFieldSystem chemicalField;

        public ExperimentConfiguration Configuration
        {
            get => configuration;
            set => configuration = value;
        }

        private void Awake()
        {
            ChamberWallHider.HideWalls();
        }

        private void Start()
        {
            ChamberWallHider.HideWalls();
        }

        public void BindStructure(
            Transform floorSlab,
            Transform ceilingSlab,
            Transform[] chamberWalls,
            Transform[] northSegments,
            Transform[] tunnel)
        {
            floor = floorSlab;
            ceiling = ceilingSlab;
            walls = chamberWalls;
            northWallSegments = northSegments ?? new Transform[0];
            tunnelPieces = tunnel ?? new Transform[0];
        }

        public void ApplyConfiguration(ExperimentConfiguration config)
        {
            configuration = config;
            ResizeChamber(config.ChamberSize);
            PlaceSources(config);
            if (mainLight != null)
            {
                mainLight.intensity = config.LightIntensity;
            }

            chemicalField?.DiscoverSources();
        }

        public void ResizeChamber(Vector3 size)
        {
            float slab = FloorThickness(size);
            float wall = WallThickness(size);

            if (floor != null)
            {
                floor.localScale = new Vector3(size.x, slab, size.z);
                floor.localPosition = Vector3.zero;
            }

            if (ceiling != null)
            {
                ceiling.localScale = new Vector3(size.x, slab, size.z);
                ceiling.localPosition = new Vector3(0f, size.y, 0f);
            }

            if (walls == null || walls.Length < 4)
            {
                return;
            }

            float hx = size.x * 0.5f;
            float hz = size.z * 0.5f;
            float hy = size.y * 0.5f;

            SetWall(walls[0], new Vector3(0f, hy, hz), new Vector3(size.x, size.y, wall));
            SetWall(walls[1], new Vector3(0f, hy, -hz), new Vector3(size.x, size.y, wall));
            SetWall(walls[2], new Vector3(hx, hy, 0f), new Vector3(wall, size.y, size.z));
            SetWall(walls[3], new Vector3(-hx, hy, 0f), new Vector3(wall, size.y, size.z));
            ApplyTunnel(size);
        }

        public bool TunnelActive =>
            configuration != null &&
            configuration.TunnelEnabled &&
            HasAll(northWallSegments) &&
            HasAll(tunnelPieces);

        /// <summary>Swap the solid north wall for the split wall + passage when the tunnel is on.</summary>
        private void ApplyTunnel(Vector3 size)
        {
            bool tunnel = TunnelActive;
            if (walls[0] != null)
            {
                walls[0].gameObject.SetActive(!tunnel);
            }

            SetActive(northWallSegments, tunnel);
            SetActive(tunnelPieces, tunnel);
            if (!tunnel)
            {
                return;
            }

            TunnelLayout.Apply(northWallSegments, TunnelLayout.NorthWallSegments(configuration, size));
            TunnelLayout.Apply(tunnelPieces, TunnelLayout.TunnelPieces(configuration, size));
        }

        private static bool HasAll(Transform[] parts)
        {
            if (parts == null || parts.Length == 0)
            {
                return false;
            }

            for (int i = 0; i < parts.Length; i++)
            {
                if (parts[i] == null)
                {
                    return false;
                }
            }

            return true;
        }

        private static void SetActive(Transform[] parts, bool active)
        {
            if (parts == null)
            {
                return;
            }

            for (int i = 0; i < parts.Length; i++)
            {
                if (parts[i] != null)
                {
                    parts[i].gameObject.SetActive(active);
                }
            }
        }

        /// <summary>World Y of the floor top surface for the given chamber size.</summary>
        public static float FloorTopY(Vector3 chamberSize)
        {
            return FloorThickness(chamberSize) * 0.5f;
        }

        public static float FloorThickness(Vector3 size)
        {
            // Keep a collision-stable slab; ultra-thin floors let the fly tunnel through.
            return Mathf.Max(0.01f, size.y * 0.08f);
        }

        public static float WallThickness(Vector3 size)
        {
            return Mathf.Max(0.001f, Mathf.Min(size.x, size.z) * 0.02f);
        }

        private static void SetWall(Transform wall, Vector3 position, Vector3 scale)
        {
            if (wall == null)
            {
                return;
            }

            wall.localPosition = position;
            wall.localScale = scale;
        }

        private void PlaceSources(ExperimentConfiguration config)
        {
            if (sugarSource != null)
            {
                sugarSource.transform.localPosition = config.SugarSourcePosition;
                sugarSource.Configure(
                    ChemicalSource.SourceType.Sugar,
                    config.SugarStrength,
                    ExperimentConfiguration.SugarPlumeRadius,
                    ExperimentConfiguration.SugarPlumeEdgeFraction);
                var surface = sugarSource.GetComponent<SurfaceChemistry>();
                if (surface != null)
                {
                    surface.Sugar = config.SugarStrength;
                }
            }

            if (bitterSource != null)
            {
                bitterSource.transform.localPosition = config.BitterSourcePosition;
                bitterSource.Configure(
                    ChemicalSource.SourceType.Bitter,
                    config.BitterStrength,
                    config.ResolvedChemicalSpread);
                var surface = bitterSource.GetComponent<SurfaceChemistry>();
                if (surface != null)
                {
                    surface.Bitter = config.BitterStrength;
                }
            }
        }
    }
}
