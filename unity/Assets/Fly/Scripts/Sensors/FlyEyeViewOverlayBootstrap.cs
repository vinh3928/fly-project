using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Ensures the eye PIP overlay exists even if OverviewCamera setup is skipped.
    /// </summary>
    public static class FlyEyeViewOverlayBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsureOverlay()
        {
            if (Object.FindFirstObjectByType<FlyEyeViewOverlay>() != null)
            {
                return;
            }

            var host = GameObject.Find("OverviewCamera");
            if (host == null && Camera.main != null)
            {
                host = Camera.main.gameObject;
            }

            if (host == null)
            {
                host = new GameObject("FlyEyeViewOverlayHost");
            }

            host.AddComponent<FlyEyeViewOverlay>();
        }
    }
}
