using FlySimulation.Environment;
using FlySimulation.Experiments;
using FlySimulation.Fly.Body;
using FlySimulation.Fly.Motors;
using FlySimulation.Fly.Recording;
using FlySimulation.Fly.Sensors;
using UnityEngine;
#if UNITY_MLAGENTS
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
#endif

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Primary fly agent. Compiles with ML-Agents when UNITY_MLAGENTS is defined;
    /// otherwise behaves as a standalone MonoBehaviour for editor prototyping.
    /// </summary>
#if UNITY_MLAGENTS
    public sealed class FlyAgent : Agent
#else
    public sealed class FlyAgent : MonoBehaviour
#endif
    {
        [SerializeField] private ExperimentConfiguration configuration;
        [SerializeField] private FlyBodyController body;
        [SerializeField] private FlyHomeostasis homeostasis;
        [SerializeField] private FlyMotorController motors;
        [SerializeField] private FlyObservationCollector observations;
        [SerializeField] private FlyEpisodeManager episodeManager;
        [SerializeField] private ExperimentRecorder recorder;
        [SerializeField] private bool useHighLevelActions = true;
        [SerializeField] private bool useWalkingDemo = true;

        private RewardCalculator _rewardCalculator;
        private float _decisionAccumulator;
        private float _previousEnergy;
        private bool _decisionRequested;

        public ExperimentConfiguration Configuration => configuration;
        public long SimulationStepCount =>
            episodeManager != null ? episodeManager.SimulationStep : 0;

#if UNITY_MLAGENTS
        public override void Initialize()
        {
            BaseInitialize();
        }

        public override void OnEpisodeBegin()
        {
            episodeManager?.StartEpisode();
            _rewardCalculator?.Reset(homeostasis);
            _previousEnergy = homeostasis != null ? homeostasis.Energy : 1f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            float dt = Time.fixedDeltaTime;
            int length = observations != null ? observations.Collect(dt) : 0;
            float[] vector = observations != null ? observations.GetVectorObservation() : null;
            if (vector == null)
            {
                return;
            }

            for (int i = 0; i < length; i++)
            {
                sensor.AddObservation(vector[i]);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Play-mode WalkingDemoDriver owns motors; ignore zero/default ML-Agents actions.
            if (!useWalkingDemo)
            {
                ApplyActions(actions.ContinuousActions.Array);
            }

            float reward = ComputeStepReward(0f);
            AddReward(reward);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuous = actionsOut.ContinuousActions;
            if (useWalkingDemo && motors != null)
            {
                continuous[0] = motors.WalkForward;
                continuous[1] = motors.TurnBias;
                if (continuous.Length > 2) continuous[2] = 0.5f;
                if (continuous.Length > 3) continuous[3] = 0.7f;
                if (continuous.Length > 4) continuous[4] = 0f;
                return;
            }

            continuous[0] = Input.GetAxis("Vertical");
            continuous[1] = Input.GetAxis("Horizontal");
            if (continuous.Length > 2) continuous[2] = 0.5f;
            if (continuous.Length > 3) continuous[3] = 0.5f;
            if (continuous.Length > 4) continuous[4] = Input.GetKey(KeyCode.Space) ? 1f : 0f;
        }
#else
        private void Awake()
        {
            BaseInitialize();
        }

        private void Start()
        {
            episodeManager?.StartEpisode();
            _rewardCalculator?.Reset(homeostasis);
        }
#endif

        private void BaseInitialize()
        {
            if (configuration == null)
            {
                configuration = ExperimentConfiguration.Default();
            }

            _rewardCalculator = new RewardCalculator(configuration);
            if (episodeManager != null)
            {
                episodeManager.Configure(configuration);
                episodeManager.EpisodeEnded += OnEpisodeEnded;
            }
        }

        private void FixedUpdate()
        {
            if (body != null)
            {
                body.FixedSensorUpdate();
            }

            if (homeostasis != null && motors != null)
            {
                homeostasis.Tick(Time.fixedDeltaTime, motors.LastMotorEffort, usingWings: false);
            }

            TryConsumeFood();

#if !UNITY_MLAGENTS
            _decisionAccumulator += Time.fixedDeltaTime;
            float interval = 1f / Mathf.Max(1f, configuration.DecisionHz);
            if (_decisionAccumulator >= interval)
            {
                _decisionAccumulator = 0f;
                observations?.Collect(Time.fixedDeltaTime);
                // WalkingDemoDriver owns high-level commands in Play mode when enabled.
                if (!useWalkingDemo)
                {
                    motors?.SetHighLevelCommands(0f, 0f, 0.5f, 0.5f);
                    motors?.SetProboscisExtension(0f);
                }

                ComputeStepReward(0f);
            }
#endif

            if (recorder != null && episodeManager != null && episodeManager.EpisodeActive)
            {
                int recordPeriod = Mathf.Max(
                    1,
                    Mathf.RoundToInt(configuration.PhysicsHz / Mathf.Max(1f, configuration.RecordingHz)));
                if (episodeManager.SimulationStep % recordPeriod == 0)
                {
                    observations?.Collect(Time.fixedDeltaTime);
                    recorder.RecordStep(episodeManager, observations, motors, homeostasis, body);
                }
            }
        }

        public void ApplyActions(float[] continuousActions)
        {
            if (useWalkingDemo)
            {
                return;
            }

            if (motors == null || continuousActions == null || continuousActions.Length == 0)
            {
                return;
            }

            if (useHighLevelActions)
            {
                float forward = continuousActions.Length > 0 ? continuousActions[0] : 0f;
                float turn = continuousActions.Length > 1 ? continuousActions[1] : 0f;
                float height = continuousActions.Length > 2 ? continuousActions[2] : 0.5f;
                float frequency = continuousActions.Length > 3 ? continuousActions[3] : 0.5f;
                float proboscis = continuousActions.Length > 4 ? continuousActions[4] : 0f;
                motors.SetHighLevelCommands(forward, turn, height, frequency);
                motors.SetProboscisExtension(Mathf.Clamp01(proboscis));
            }
            else
            {
                motors.SetDirectJointTargets(continuousActions);
                if (continuousActions.Length > 18)
                {
                    motors.SetProboscisExtension(Mathf.Clamp01(continuousActions[18]));
                }
            }
        }

        private float ComputeStepReward(float curiosity)
        {
            if (_rewardCalculator == null || homeostasis == null || body == null || motors == null)
            {
                return 0f;
            }

            float reward = _rewardCalculator.Compute(
                homeostasis,
                curiosity,
                motors.LastMotorEffort,
                body.BodyInstability(),
                body.HasFallen);

            _previousEnergy = homeostasis.Energy;
            return reward;
        }

        private void TryConsumeFood()
        {
            if (homeostasis == null || observations == null)
            {
                return;
            }

            var proboscis = observations.Sensors != null ? observations.Sensors.Proboscis : null;
            if (proboscis == null || !proboscis.IsContacting)
            {
                return;
            }

            // Require meaningful extension before feeding / toxin uptake.
            if (proboscis.ExtensionNormalized < 0.35f)
            {
                return;
            }

            // Taste is valid only during contact (Phase 3 gating).
            ChemicalChannels taste = proboscis.GatedTaste;
            float dt = Time.fixedDeltaTime;
            if (taste.Sugar > 0.2f)
            {
                float amount = 0.08f * dt;
                homeostasis.ConsumeSugar(amount);
                var metrics = GetComponent<FoodSearchMetrics>();
                metrics?.RegisterSugarConsume(amount);
                recorder?.LogEvent(
                    SimulationStepCount,
                    episodeManager != null ? episodeManager.EpisodeTime : 0f,
                    "food_consumed");
            }

            if (taste.Water > 0.2f)
            {
                homeostasis.ConsumeWater(0.05f * dt);
            }

            if (taste.Bitter > 0.2f || taste.Toxin > 0.1f)
            {
                float toxin = 0.04f * dt;
                homeostasis.ApplyBitterOrToxin(toxin);
                var metrics = GetComponent<FoodSearchMetrics>();
                metrics?.RegisterBitterContact();
                recorder?.LogEvent(
                    SimulationStepCount,
                    episodeManager != null ? episodeManager.EpisodeTime : 0f,
                    "fly_touched_bitter");
            }
        }

        private void OnEpisodeEnded(string reason)
        {
#if UNITY_MLAGENTS
            EndEpisode();
#endif
        }

        private void OnDestroy()
        {
            if (episodeManager != null)
            {
                episodeManager.EpisodeEnded -= OnEpisodeEnded;
            }
        }
    }
}
