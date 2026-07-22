using FlySimulation.Fly.Motors;
using UnityEngine;

namespace FlySimulation.Fly.Brain
{
    /// <summary>
    /// Play-mode Phase 2 demo: cycles stand → walk → turn so exit criteria are visible.
    /// Re-applies commands every FixedUpdate so ML-Agents default-zero actions cannot stick.
    /// </summary>
    public sealed class WalkingDemoDriver : MonoBehaviour
    {
        [SerializeField] private FlyMotorController motors;
        [SerializeField] private bool autoStart = true;
        [SerializeField] private float standSeconds = 1f;
        [SerializeField] private float walkSeconds = 5f;
        [SerializeField] private float turnSeconds = 4f;

        private float _timer;
        private int _stage;
        private string _lastLoggedStage;

        public void Bind(FlyMotorController motorController)
        {
            motors = motorController;
        }

        private void OnEnable()
        {
            _timer = 0f;
            _stage = 0;
            _lastLoggedStage = null;
            ApplyStage(forceLog: true);
        }

        private void FixedUpdate()
        {
            if (!autoStart || motors == null)
            {
                return;
            }

            _timer += Time.fixedDeltaTime;
            float stageDuration = _stage switch
            {
                0 => standSeconds,
                1 => walkSeconds,
                _ => turnSeconds
            };

            if (_timer >= stageDuration)
            {
                _timer = 0f;
                _stage = (_stage + 1) % 3;
            }

            // Always reassert — ML-Agents may emit zero actions between our updates.
            ApplyStage(forceLog: false);
        }

        private void ApplyStage(bool forceLog)
        {
            if (motors == null)
            {
                return;
            }

            switch (_stage)
            {
                case 0:
                    motors.SetHighLevelCommands(0f, 0f, 0.5f, 0.5f);
                    LogStage("stand (idle)", forceLog);
                    break;
                case 1:
                    motors.SetHighLevelCommands(1f, 0f, 0.5f, 0.8f);
                    LogStage("walk forward", forceLog);
                    break;
                default:
                    motors.SetHighLevelCommands(0.4f, 1f, 0.5f, 0.8f);
                    LogStage("turn", forceLog);
                    break;
            }
        }

        private void LogStage(string label, bool forceLog)
        {
            if (!forceLog && _lastLoggedStage == label)
            {
                return;
            }

            _lastLoggedStage = label;
            Debug.Log($"WalkingDemoDriver → {label}");
        }
    }
}
