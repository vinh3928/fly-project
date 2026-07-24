using FlySimulation.Environment;
using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Keeps the Game-view camera framed on the chamber every Play session.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public sealed class ChamberOverviewCamera : MonoBehaviour
    {
        [SerializeField] private Vector3 chamberSize = new Vector3(0.5f, 0.25f, 0.5f);
        [SerializeField] private float fieldOfView = 50f;

        private Camera _camera;

        public void Configure(Vector3 size)
        {
            chamberSize = size;
            ApplyFraming();
        }

        private void Awake()
        {
            _camera = GetComponent<Camera>();
            ClaimMainCamera();
            ResolveChamberSize();
            ApplyFraming();
            ChamberWallHider.HideWalls();
        }

        private void OnEnable()
        {
            ClaimMainCamera();
            ApplyFraming();
            ChamberWallHider.HideWalls();
        }

        private void ResolveChamberSize()
        {
            var chamber = FindFirstObjectByType<ChamberController>();
            if (chamber != null && chamber.Configuration != null)
            {
                chamberSize = chamber.Configuration.ChamberSize;
                return;
            }

            var floor = GameObject.Find("Floor");
            if (floor != null)
            {
                Vector3 scale = floor.transform.localScale;
                float height = 0.05f;
                var ceiling = GameObject.Find("Ceiling");
                if (ceiling != null)
                {
                    height = Mathf.Max(0.02f, ceiling.transform.localPosition.y);
                }

                chamberSize = new Vector3(Mathf.Max(0.05f, scale.x), height, Mathf.Max(0.05f, scale.z));
            }
        }

        private void ClaimMainCamera()
        {
            if (_camera == null)
            {
                _camera = GetComponent<Camera>();
            }

            foreach (var other in FindObjectsByType<Camera>(FindObjectsSortMode.None))
            {
                if (other == null || other == _camera)
                {
                    continue;
                }

                if (other.gameObject.name != "OverviewCamera")
                {
                    other.enabled = false;
                }

                if (other.CompareTag("MainCamera"))
                {
                    other.tag = "Untagged";
                }
            }

            gameObject.tag = "MainCamera";
            _camera.enabled = true;
            _camera.depth = 100f;
        }

        public void ApplyFraming()
        {
            if (_camera == null)
            {
                _camera = GetComponent<Camera>();
            }

            Place(_camera, transform, chamberSize, fieldOfView);
        }

        /// <summary>Shared framing used by editor tools and Play mode.</summary>
        public static void Place(Camera cam, Transform pose, Vector3 chamberSize, float fov = 50f)
        {
            if (cam == null || pose == null)
            {
                return;
            }

            float extent = Mathf.Max(chamberSize.x, chamberSize.z, 0.05f);
            float height = Mathf.Max(chamberSize.y, 0.03f);
            // Sit outside the cutaway corner (+X, -Z) and look into the arena.
            Vector3 focus = new Vector3(0f, height * 0.2f, 0f);
            Vector3 position = new Vector3(extent * 1.6f, height * 2.4f, -extent * 1.6f);

            pose.position = position;
            pose.LookAt(focus);

            cam.enabled = true;
            cam.tag = "MainCamera";
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.12f, 0.13f, 0.16f);
            cam.fieldOfView = fov;
            // Avoid ultra-small near clips — they can make the whole Game view blank on Metal.
            cam.nearClipPlane = Mathf.Clamp(extent * 0.05f, 0.01f, 0.05f);
            cam.farClipPlane = Mathf.Max(10f, extent * 40f);
            cam.depth = 100f;
            cam.cullingMask = ~0;
            cam.targetTexture = null;
        }
    }
}
