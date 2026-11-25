using FlySimulation.Fly.Body;
using UnityEngine;

namespace FlySimulation.Environment
{
    /// <summary>
    /// Third-person chase cam locked on the fly, rendered on Unity Display 2
    /// (<see cref="Camera.targetDisplay"/> = 1). Display 1 keeps the chamber overview.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public sealed class FlyFollowCamera : MonoBehaviour
    {
        public const string ObjectName = "FlyFollowCamera";
        public const int TargetDisplayIndex = 1; // Unity Display 2

        [SerializeField] private Transform target;
        [SerializeField] private Vector3 localOffset = new Vector3(0f, 0.045f, -0.09f);
        [SerializeField] private Vector3 lookAtOffset = new Vector3(0f, 0.008f, 0.01f);
        [SerializeField] private float fieldOfView = 55f;
        [SerializeField] private float positionLerp = 18f;
        [SerializeField] private bool activateDisplay2 = true;

        private Camera _camera;
        private bool _displayActivated;

        public Transform Target => target;
        public Camera Camera => _camera != null ? _camera : (_camera = GetComponent<Camera>());

        public void Bind(Transform fly)
        {
            target = fly;
            SnapToTarget();
        }

        public void Configure(Vector3 offset, float fov)
        {
            localOffset = offset;
            fieldOfView = fov;
            ApplyLens();
        }

        private void Awake()
        {
            _camera = GetComponent<Camera>();
            ApplyLens();
            ResolveTarget();
            TryActivateDisplay2();
            SnapToTarget();
        }

        private void OnEnable()
        {
            ApplyLens();
            ResolveTarget();
            TryActivateDisplay2();
            SnapToTarget();
        }

        private void LateUpdate()
        {
            if (target == null)
            {
                ResolveTarget();
                if (target == null)
                {
                    return;
                }
            }

            Follow(Time.deltaTime);
        }

        private void ResolveTarget()
        {
            if (target != null)
            {
                return;
            }

            var body = FindAnyObjectByType<FlyBodyController>();
            if (body != null)
            {
                target = body.transform;
            }
        }

        private void TryActivateDisplay2()
        {
            if (!activateDisplay2 || _displayActivated)
            {
                return;
            }

            if (Display.displays.Length > TargetDisplayIndex)
            {
                Display.displays[TargetDisplayIndex].Activate();
                _displayActivated = true;
            }
        }

        public void ApplyLens()
        {
            if (_camera == null)
            {
                _camera = GetComponent<Camera>();
            }

            Place(_camera, fieldOfView);
        }

        public void SnapToTarget()
        {
            if (target == null || _camera == null)
            {
                return;
            }

            transform.position = DesiredPosition();
            transform.rotation = DesiredRotation();
        }

        private void Follow(float dt)
        {
            float t = 1f - Mathf.Exp(-Mathf.Max(0.01f, positionLerp) * Mathf.Max(0f, dt));
            transform.position = Vector3.Lerp(transform.position, DesiredPosition(), t);
            transform.rotation = Quaternion.Slerp(transform.rotation, DesiredRotation(), t);
        }

        private Vector3 DesiredPosition()
        {
            return target.TransformPoint(localOffset);
        }

        private Quaternion DesiredRotation()
        {
            Vector3 lookAt = target.TransformPoint(lookAtOffset);
            Vector3 forward = lookAt - transform.position;
            if (forward.sqrMagnitude < 1e-8f)
            {
                forward = target.forward;
            }

            return Quaternion.LookRotation(forward.normalized, Vector3.up);
        }

        /// <summary>Shared lens setup for editor tools and Play mode.</summary>
        public static void Place(Camera cam, float fov = 55f)
        {
            if (cam == null)
            {
                return;
            }

            cam.enabled = true;
            cam.tag = "Untagged";
            cam.targetDisplay = TargetDisplayIndex;
            cam.clearFlags = RenderSettings.skybox != null ? CameraClearFlags.Skybox : CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.12f, 0.13f, 0.16f);
            cam.fieldOfView = fov;
            cam.nearClipPlane = 0.005f;
            cam.farClipPlane = Mathf.Max(OutsideWorldLayout.ViewDistance, 20f);
            cam.depth = 50f;
            // Same cutaway as overview so the fly stays visible inside the box.
            cam.cullingMask = ChamberWallHider.OverviewCullingMask;
            cam.targetTexture = null;
        }

        public static bool IsFollowCamera(Camera cam)
        {
            if (cam == null)
            {
                return false;
            }

            return cam.GetComponent<FlyFollowCamera>() != null || cam.gameObject.name == ObjectName;
        }
    }
}
