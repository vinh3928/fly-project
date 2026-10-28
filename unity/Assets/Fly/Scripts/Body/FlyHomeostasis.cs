using FlySimulation.Experiments;
using UnityEngine;

namespace FlySimulation.Fly.Body
{
    /// <summary>
    /// Internal biological motivation variables with configurable metabolism.
    /// </summary>
    public sealed class FlyHomeostasis : MonoBehaviour
    {
        /// <summary>Energy never drops below this, so the fly cannot starve to zero.</summary>
        public const float MinEnergy = 0.01f;

        [SerializeField] private float energy = 1f;
        [SerializeField] private float hydration = 1f;
        [SerializeField] private float fatigue;
        [SerializeField] private float damage;
        [SerializeField] private float bodyTemperature = 25f;
        [SerializeField] private float toxinExposure;
        [SerializeField] private float hunger;
        [SerializeField] private float arousal = 0.5f;

        [SerializeField] private float metabolismRate = 0.002f;
        [SerializeField] private float movementEnergyCost = 0.01f;
        [SerializeField] private float wingEnergyCost = 0.05f;
        [SerializeField] private float restRecoveryRate = 0.01f;

        public float Energy => energy;
        public float Hydration => hydration;
        public float Fatigue => fatigue;
        public float Damage => damage;
        public float BodyTemperature => bodyTemperature;
        public float ToxinExposure => toxinExposure;
        public float Hunger => hunger;
        public float Arousal => arousal;

        public bool IsDepleted => energy <= 0f || hydration <= 0f || damage >= 1f;

        public void Configure(ExperimentConfiguration config)
        {
            energy = ClampEnergy(config.InitialEnergy);
            hydration = config.InitialHydration;
            fatigue = 0f;
            damage = 0f;
            toxinExposure = 0f;
            hunger = 1f - energy;
            arousal = 0.5f;
            bodyTemperature = config.TemperatureCelsius;
            metabolismRate = config.MetabolismRate;
            movementEnergyCost = config.MovementEnergyCost;
            wingEnergyCost = config.WingEnergyCost;
        }

        public void ResetState(ExperimentConfiguration config)
        {
            Configure(config);
        }

        public void Tick(float deltaTime, float motorEffort, bool usingWings)
        {
            float burn = metabolismRate + motorEffort * movementEnergyCost;
            if (usingWings)
            {
                burn += wingEnergyCost;
            }

            energy = ClampEnergy(energy - burn * deltaTime);
            hydration = Mathf.Clamp01(hydration - metabolismRate * 0.5f * deltaTime);
            fatigue = Mathf.Clamp01(fatigue + motorEffort * deltaTime - restRecoveryRate * (1f - motorEffort) * deltaTime);
            hunger = Mathf.Clamp01(1f - energy);
            toxinExposure = Mathf.Clamp01(toxinExposure - 0.01f * deltaTime);
            damage = Mathf.Clamp01(damage);
        }

        public void ConsumeSugar(float amount)
        {
            energy = ClampEnergy(energy + amount);
            hunger = Mathf.Clamp01(1f - energy);
        }

        public static float ClampEnergy(float value)
        {
            return Mathf.Clamp(value, MinEnergy, 1f);
        }

        public void ConsumeWater(float amount)
        {
            hydration = Mathf.Clamp01(hydration + amount);
        }

        public void ApplyBitterOrToxin(float amount)
        {
            damage = Mathf.Clamp01(damage + amount);
            toxinExposure = Mathf.Clamp01(toxinExposure + amount);
            arousal = Mathf.Clamp01(arousal + amount);
        }

        public void ApplyCollisionDamage(float amount)
        {
            damage = Mathf.Clamp01(damage + amount);
        }

        public void CopyNormalized(float[] buffer, int offset = 0)
        {
            buffer[offset] = energy;
            buffer[offset + 1] = hydration;
            buffer[offset + 2] = fatigue;
            buffer[offset + 3] = damage;
            buffer[offset + 4] = Mathf.Clamp01((bodyTemperature - 10f) / 30f);
            buffer[offset + 5] = toxinExposure;
            buffer[offset + 6] = hunger;
            buffer[offset + 7] = arousal;
        }

        public static float ComputeEnergyDelta(float previous, float current)
        {
            return current - previous;
        }
    }
}
