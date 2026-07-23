using UnityEngine;

namespace FlySimulation.Fly.Sensors
{
    /// <summary>
    /// Eval-only food-search metrics. Not packed into ML-Agents vector observations.
    /// </summary>
    public sealed class FoodSearchMetrics : MonoBehaviour
    {
        public float SugarConsumed { get; private set; }
        public int BitterContacts { get; private set; }
        public float EnergyDelta { get; private set; }

        private float _energyAtReset = 1f;

        public void ResetMetrics(float currentEnergy)
        {
            SugarConsumed = 0f;
            BitterContacts = 0;
            EnergyDelta = 0f;
            _energyAtReset = currentEnergy;
        }

        public void RegisterSugarConsume(float amount)
        {
            SugarConsumed += Mathf.Max(0f, amount);
        }

        public void RegisterBitterContact()
        {
            BitterContacts++;
        }

        public void UpdateEnergyDelta(float currentEnergy)
        {
            EnergyDelta = currentEnergy - _energyAtReset;
        }
    }
}
