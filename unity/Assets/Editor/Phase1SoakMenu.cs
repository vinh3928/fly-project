using FlySimulation.Experiments;
using UnityEditor;
using UnityEngine;

namespace FlySimulation.EditorTools
{
    /// <summary>
    /// Optional long-running Phase 1 physics soak (10 simulated minutes).
    /// Prefer EditMode automated soak for CI; use this when validating in Play mode.
    /// </summary>
    public static class Phase1SoakMenu
    {
        [MenuItem("Tools/Fly/Phase 1 Physics Soak (Play Mode)", priority = 30)]
        [MenuItem("Fly/Phase 1 Physics Soak (Play Mode)", priority = 30)]
        public static void StartPlayModeSoakHint()
        {
            Debug.Log(
                "Phase 1 soak: enter Play on Chamber.unity and leave running for " +
                Phase1ExitCriteria.StabilitySeconds +
                " simulated seconds (episode_summary.json → physics_stable). " +
                "Automated EditMode soak covers " +
                Phase1ExitCriteria.AutomatedStabilitySeconds +
                "s via Physics.Simulate.");
        }
    }
}
