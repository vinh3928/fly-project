using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using UnityEngine;

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Combines curiosity (external) with homeostatic and stability terms.
    /// </summary>
    public sealed class RewardCalculator
    {
        private readonly ExperimentConfiguration _config;
        private float _previousEnergy;
        private float _previousHydration;

        public RewardCalculator(ExperimentConfiguration config)
        {
            _config = config ?? ExperimentConfiguration.Default();
            _previousEnergy = _config.InitialEnergy;
            _previousHydration = _config.InitialHydration;
        }

        public void Reset(FlyHomeostasis homeostasis)
        {
            _previousEnergy = homeostasis != null ? homeostasis.Energy : _config.InitialEnergy;
            _previousHydration = homeostasis != null ? homeostasis.Hydration : _config.InitialHydration;
        }

        public float Compute(
            FlyHomeostasis homeostasis,
            float curiosity,
            float motorEffort,
            float instability,
            bool fell)
        {
            float energyImprovement = homeostasis.Energy - _previousEnergy;
            float hydrationImprovement = homeostasis.Hydration - _previousHydration;
            _previousEnergy = homeostasis.Energy;
            _previousHydration = homeostasis.Hydration;

            float cappedCuriosity = Mathf.Clamp(curiosity, 0f, 1f);

            float reward =
                _config.CuriosityWeight * cappedCuriosity
                + _config.EnergyWeight * energyImprovement
                + _config.HydrationWeight * hydrationImprovement
                - _config.DamageWeight * homeostasis.Damage
                - _config.EffortWeight * motorEffort
                - _config.InstabilityWeight * instability;

            if (fell)
            {
                reward -= 1f;
            }

            return reward;
        }

        public static float ComputeStatic(
            float curiosity,
            float energyImprovement,
            float hydrationImprovement,
            float damage,
            float motorEffort,
            float instability,
            ExperimentConfiguration config)
        {
            return config.CuriosityWeight * Mathf.Clamp(curiosity, 0f, 1f)
                   + config.EnergyWeight * energyImprovement
                   + config.HydrationWeight * hydrationImprovement
                   - config.DamageWeight * damage
                   - config.EffortWeight * motorEffort
                   - config.InstabilityWeight * instability;
        }
    }
}
