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
        [Tooltip("When true AND WalkingDemoDriver is enabled, demo owns motors. Leave false for live mind / Python.")]
        [SerializeField] private bool useWalkingDemo = false;
        [Tooltip("Phase 9 architecture B: append mind/motor goals to vector observations for PPO.")]
        [SerializeField] private bool includeMotorGoalsInObs;
        [Tooltip("Phase 9: add locomotion/goal-tracking reward terms.")]
        [SerializeField] private bool useLocomotionReward;

        private RewardCalculator _rewardCalculator;
        private float _decisionAccumulator;
        private float _previousEnergy;
        private bool _decisionRequested;
        private readonly float[] _motorGoal = { 0f, 0f, 0.5f, 0.5f, 0f };
        private Vector3 _previousPosition;
        private float _previousYaw;

        public ExperimentConfiguration Configuration => configuration;
        public long SimulationStepCount =>
            episodeManager != null ? episodeManager.SimulationStep : 0;
        /// <summary>When true, WalkingDemoDriver owns motors; Python/ML-Agents actions are ignored.</summary>
        public bool UseWalkingDemo => useWalkingDemo;

        /// <summary>
        /// Demo owns motors only when the flag is on <em>and</em> WalkingDemoDriver is enabled.
        /// Avoids a stuck fly when Use Walking Demo is true but the driver component is disabled.
        /// </summary>
        public bool DemoOwnsMotors
        {
            get
            {
                if (!useWalkingDemo)
                {
                    return false;
                }

                var demo = GetComponent<WalkingDemoDriver>();
                return demo != null && demo.isActiveAndEnabled;
            }
        }


#if UNITY_MLAGENTS
        public override void Initialize()
        {
            EnsureEyeCameraSensors();
            BaseInitialize();
            if (useWalkingDemo && !DemoOwnsMotors)
            {
                Debug.LogWarning(
                    "FlyAgent: Use Walking Demo is on but WalkingDemoDriver is disabled — " +
                    "applying ML-Agents/Python actions. Enable WalkingDemoDriver for the stand/walk/turn demo, " +
                    "or leave Use Walking Demo off for live mind.");
            }
            else if (!useWalkingDemo)
            {
                bool trainer =
                    Unity.MLAgents.Academy.IsInitialized
                    && Unity.MLAgents.Academy.Instance.IsCommunicatorOn;
                if (trainer)
                {
                    Debug.Log(
                        "FlyAgent: live control — connected to Python trainer; actions drive motors.");
                }
                else
                {
                    Debug.LogError(
                        "FlyAgent: NO Python trainer on port 5004 — Academy is in inference mode. "
                        + "Actions will be ~zero and the fly will stand still. "
                        + "Stop Play, start: python -m scripts.run_live_mind --editor … "
                        + "wait until it prints Listening on port 5004, THEN press Play.");
                }
            }
        }

        /// <summary>
        /// Register ML-Agents camera sensors for eye frames before Agent sensor init completes.
        /// </summary>
        private void EnsureEyeCameraSensors()
        {
            var bridge = GetComponent<FlyEyeMlAgentsBridge>();
            if (bridge == null)
            {
                bridge = gameObject.AddComponent<FlyEyeMlAgentsBridge>();
            }

            bridge.EnsureCameraSensors();
        }

        public override void OnEpisodeBegin()
        {
            episodeManager?.StartEpisode();
            _rewardCalculator?.Reset(homeostasis);
            _previousEnergy = homeostasis != null ? homeostasis.Energy : 1f;
            if (body != null)
            {
                _previousPosition = body.transform.position;
                _previousYaw = body.transform.eulerAngles.y;
            }

            // Sample a stand/walk/turn goal for motor PPO when goals are in the obs.
            if (includeMotorGoalsInObs)
            {
                SampleTrainingGoal();
            }
        }

        /// <summary>Phase 9: mind / Python sets the motor goal setpoint.</summary>
        public void SetMotorGoal(float forward, float turn, float height, float frequency, float proboscis)
        {
            _motorGoal[0] = Mathf.Clamp(forward, -1f, 1f);
            _motorGoal[1] = Mathf.Clamp(turn, -1f, 1f);
            _motorGoal[2] = Mathf.Clamp01(height);
            _motorGoal[3] = Mathf.Clamp01(frequency);
            _motorGoal[4] = Mathf.Clamp01(proboscis);
        }

        public float[] GetMotorGoal() => (float[])_motorGoal.Clone();

        public override void CollectObservations(VectorSensor sensor)
        {
            const int expected = SensorManager.ExpectedVectorSize;
            const int internalOffset =
                SensorManager.JointCount * 4 +
                SensorManager.FootCount * 5 +
                SensorManager.FootCount * SensorManager.ChemicalChannels +
                SensorManager.ChemicalChannels * 2 +
                (2 + SensorManager.ChemicalChannels);
            float dt = Time.fixedDeltaTime;

            // Recover bindings if the scene reference was lost at runtime.
            if (observations != null && observations.Sensors == null)
            {
                var manager = GetComponent<SensorManager>();
                if (manager != null)
                {
                    observations.Bind(manager);
                }
            }

            int length = observations != null ? observations.Collect(dt) : 0;
            float[] vector = observations != null ? observations.GetVectorObservation() : null;

            if (vector == null || length <= 0)
            {
                vector = new float[expected];
                length = expected;
                Debug.LogError(
                    "FlyAgent.CollectObservations: sensor collect failed; " +
                    "building a minimal vector from homeostasis only.");
            }

            // Always stamp internals from the live FlyHomeostasis component.
            // This is the source of truth the Inspector shows; SensorManager alone
            // has been observed to write zeros across to Python.
            FlyHomeostasis homeo = homeostasis != null ? homeostasis : GetComponent<FlyHomeostasis>();
            if (homeo != null && length > internalOffset + SensorManager.InternalStateCount - 1)
            {
                homeo.CopyNormalized(vector, internalOffset);
            }

            int n = Mathf.Min(length, expected);
            for (int i = 0; i < n; i++)
            {
                sensor.AddObservation(vector[i]);
            }

            for (int i = n; i < expected; i++)
            {
                sensor.AddObservation(0f);
            }

            if (includeMotorGoalsInObs)
            {
                for (int i = 0; i < LocomotionReward.GoalDim; i++)
                {
                    sensor.AddObservation(_motorGoal[i]);
                }
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Apply Python/ML-Agents actions unless an enabled WalkingDemoDriver owns motors.
            if (!DemoOwnsMotors)
            {
                ApplyActions(actions.ContinuousActions.Array);
            }

            float reward = ComputeStepReward(0f);
            if (useLocomotionReward && body != null)
            {
                Vector3 pos = body.transform.position;
                Vector3 delta = pos - _previousPosition;
                Vector3 forward = new Vector3(
                    Mathf.Sin(_previousYaw * Mathf.Deg2Rad),
                    0f,
                    Mathf.Cos(_previousYaw * Mathf.Deg2Rad));
                float forwardProgress = Vector3.Dot(delta, forward);
                float yawProgress = Mathf.DeltaAngle(_previousYaw, body.transform.eulerAngles.y) * Mathf.Deg2Rad;
                float[] actionArr = actions.ContinuousActions.Array;
                reward += LocomotionReward.Compute(
                    _motorGoal,
                    actionArr,
                    forwardProgress,
                    yawProgress,
                    upright: body.HasFallen ? 0f : 1f,
                    fell: body.HasFallen);
                _previousPosition = pos;
                _previousYaw = body.transform.eulerAngles.y;
            }

            AddReward(reward);
        }

        private void SampleTrainingGoal()
        {
            // Curriculum-ish random setpoints for goal-conditioned motor training.
            float mode = Random.value;
            if (mode < 0.25f)
            {
                SetMotorGoal(0f, 0f, 0.5f, 0.5f, 0f); // stand
            }
            else if (mode < 0.7f)
            {
                SetMotorGoal(Random.Range(0.4f, 1f), Random.Range(-0.2f, 0.2f), 0.5f, 0.7f, 0f);
            }
            else
            {
                SetMotorGoal(Random.Range(0.1f, 0.5f), Random.Range(-1f, 1f), 0.5f, 0.7f, 0f);
            }
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
            // WalkingDemoDriver owns high-level commands in Play mode when actively enabled.
            if (!DemoOwnsMotors)
            {
                // Without ML-Agents there is no external action stream; hold idle.
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
            if (DemoOwnsMotors)
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
                motors.SetHighLevelCommands(
                    forward, LocomotionReward.MindTurnToMotorBias(turn), height, frequency);
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
