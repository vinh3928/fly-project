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
                sugarSource.Configure(ChemicalSource.SourceType.Sugar, config.SugarStrength, config.ChemicalSpread);
                var surface = sugarSource.GetComponent<SurfaceChemistry>();
                if (surface != null)
                {
                    surface.Sugar = config.SugarStrength;
                }
            }

            if (bitterSource != null)
            {
                bitterSource.transform.localPosition = config.BitterSourcePosition;
                bitterSource.Configure(ChemicalSource.SourceType.Bitter, config.BitterStrength, config.ChemicalSpread);
                var surface = bitterSource.GetComponent<SurfaceChemistry>();
                if (surface != null)
                {
                    surface.Bitter = config.BitterStrength;
                }
            }
        }
    }
}
