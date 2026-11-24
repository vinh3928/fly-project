using FlySimulation.Fly.Body;
using FlySimulation.Fly.Motors;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
#if UNITY_MLAGENTS
using FlySimulation.Fly.Brain;
using FlySimulation.Fly.DebugViz;
#endif

namespace FlySimulation.Environment
{
    /// <summary>
    /// Display 4: fly-centered review cam + promote queue panel
    /// (Accept / Test / Ask LLM / Refuse). World freezes only while the user is
    /// viewing Display 4 for HITL categorization. Away from Display 4, the sim
    /// keeps running and Ask LLM auto-fires immediately with no motion interrupt.
    /// Manual Ask LLM thaws <c>timeScale</c> (SideChannels need FixedUpdate) while
    /// keeping the fly locked until Python SHOWs/HIDEs again.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public sealed class SkillPromoteDisplay : MonoBehaviour
    {
        public const string ObjectName = "SkillPromoteDisplay";
        public const int TargetDisplayIndex = 3; // Unity Display 4
        public const KeyCode TestKey = KeyCode.T;

        [SerializeField] private Transform target;
        // Portrait framing: sit behind/above and look at body center.
        [SerializeField] private Vector3 localOffset = new Vector3(0f, 0.035f, -0.14f);
        [SerializeField] private Vector3 lookAtOffset = new Vector3(0f, 0.012f, 0f);
        [SerializeField] private float fieldOfView = 40f;
        [SerializeField] private float positionLerp = 22f;
        [SerializeField] private bool activateDisplay4 = true;
        [SerializeField] private float motorTestSeconds = 2.5f;
        [Tooltip("Seconds away from Display 4 before auto Ask LLM. 0 = immediate.")]
        [SerializeField] private float autoJudgeAwaySeconds = 0f;
        [SerializeField] private Color panelColor = new Color(0.07f, 0.09f, 0.12f, 0.94f);
        [SerializeField] private Color textColor = new Color(0.92f, 0.94f, 0.88f, 1f);
        [SerializeField] private Color accentColor = new Color(0.7f, 0.85f, 0.55f, 1f);

        private Camera _camera;
        private Canvas _canvas;
        private GameObject _hudRoot;
        private Text _hudText;
        private GameObject _panelRoot;
        private Text _title;
        private Text _meta;
        private Text _detail;
        private InputField _nameInput;
        private Button _accept;
        private Button _askLlm;
        private Button _refuse;
        private Button _test;
        private bool _visible;
        private bool _displayActivated;
        private bool _reviewFrozen;
        private bool _awaitingLlmJudge;
        private bool _lockMotionForJudge;
        private bool _allowLlm = true;
        private bool _lastViewingReported;
        private bool _hasViewingReport;
        private float _awaySeconds;
        private float _savedTimeScale = 1f;
        private int _pendingRequestId;
        private float _motorTestRemaining;
        private FlyMotorController _motors;
        private bool _hadTarget;

#if UNITY_MLAGENTS
        private SkillPromoteDebugChannel _channel;
#endif

        public bool IsVisible => _visible;
        public bool IsAwaitingLlmJudge => _awaitingLlmJudge;
        public string TitleText => _title != null ? _title.text : "";
        public string MetaText => _meta != null ? _meta.text : "";
        public string NameFieldText => _nameInput != null ? _nameInput.text : "";
        public Camera Camera => _camera != null ? _camera : (_camera = GetComponent<Camera>());

        /// <summary>
        /// Chamber.unity may predate Display 4 — create the chase cam at play if missing.
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void BootstrapAfterSceneLoad()
        {
            EnsureInScene();
        }

        /// <summary>Find-or-create Display 4 fly camera + promote UI host.</summary>
        public static SkillPromoteDisplay EnsureInScene(Transform fly = null)
        {
            var existing = Object.FindFirstObjectByType<SkillPromoteDisplay>();
            if (existing != null)
            {
                if (fly != null)
                {
                    existing.Bind(fly);
                }

                Place(existing.Camera, existing.fieldOfView);
                return existing;
            }

            var named = GameObject.Find(ObjectName);
            GameObject go = named != null ? named : new GameObject(ObjectName);
            var cam = go.GetComponent<Camera>() ?? go.AddComponent<Camera>();
            var display = go.GetComponent<SkillPromoteDisplay>() ?? go.AddComponent<SkillPromoteDisplay>();
            if (fly != null)
            {
                display.Bind(fly);
            }
            else
            {
                display.ResolveTarget();
            }

            Place(cam, display.fieldOfView);
            return display;
        }

        private void Awake()
        {
            _camera = GetComponent<Camera>();
            Place(_camera, fieldOfView);
            EnsureEventSystem();
            EnsureUi();
            TryActivateDisplay4();
            ResolveTarget();
            SnapToTarget();
            Hide();
        }

        private void OnEnable()
        {
            Place(Camera, fieldOfView);
            EnsureEventSystem();
            EnsureUi();
            TryActivateDisplay4();
            ResolveTarget();
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.ShowReceived += OnShow;
                _channel.HideReceived += OnHide;
                if (_channel.IsVisible)
                {
                    OnShow();
                }
            }
#endif
        }

        private void OnDisable()
        {
#if UNITY_MLAGENTS
            if (_channel != null)
            {
                _channel.ShowReceived -= OnShow;
                _channel.HideReceived -= OnHide;
                _channel = null;
            }
#endif
        }

        private void Update()
        {
            if (Input.GetKeyDown(TestKey))
            {
                RunMotorTest();
            }

            TickMotorTest(Time.unscaledDeltaTime);
            TickAwayAutoJudge(Time.unscaledDeltaTime);
            RefreshHud();
        }

        /// <summary>
        /// True when the Editor Game view is on Display 4, or a physical Display 4
        /// is active in a player build. Otherwise the user is "away".
        /// </summary>
        public static bool IsUserViewingDisplay4()
        {
#if UNITY_EDITOR
            // 0 = Display 1 … 3 = Display 4 (Game view Display dropdown).
            return Display.activeEditorGameViewTarget == TargetDisplayIndex;
#else
            return Display.displays.Length > TargetDisplayIndex;
#endif
        }

        private void TickAwayAutoJudge(float dt)
        {
            if (!_visible || _awaitingLlmJudge || !_allowLlm)
            {
                _awaySeconds = 0f;
                return;
            }

            bool viewing = IsUserViewingDisplay4();
            ReportViewingIfChanged(viewing);
            if (viewing)
            {
                _awaySeconds = 0f;
                // User arrived on Display 4 — freeze for HITL review if not already.
                if (!_reviewFrozen && _motorTestRemaining <= 0f)
                {
                    SetReviewFrozen(true);
                }

                return;
            }

            // Away from Display 4: keep sim running and auto Ask LLM (default: immediate).
            if (_reviewFrozen)
            {
                SetReviewFrozen(false);
            }

            _awaySeconds += Mathf.Max(0f, dt);
            float grace = Mathf.Max(0f, autoJudgeAwaySeconds);
            if (_awaySeconds >= grace)
            {
                if (_title != null)
                {
                    _title.text = "Away → Ask LLM judge…";
                }

                // Do not freeze / suspend motors — only HITL on Display 4 stops the fly.
                SubmitAskLlm(lockMotion: false);
            }
        }

        private void ReportViewingIfChanged(bool viewing)
        {
            if (_hasViewingReport && viewing == _lastViewingReported)
            {
                return;
            }

            _hasViewingReport = true;
            _lastViewingReported = viewing;
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.SendViewing(_pendingRequestId, viewing);
            }
#endif
        }

        private void LateUpdate()
        {
            // Keep camera on Display 4 even while review freezes timeScale.
            Place(Camera, fieldOfView);
            if (target == null)
            {
                ResolveTarget();
                if (target == null)
                {
                    return;
                }
            }

            // HITL freeze / locked judge: centered portrait. Away auto-judge: keep following.
            if ((_reviewFrozen || (_awaitingLlmJudge && _lockMotionForJudge))
                && _motorTestRemaining <= 0f)
            {
                SnapToTarget();
                _hadTarget = true;
                return;
            }

            if (!_hadTarget)
            {
                SnapToTarget();
                _hadTarget = true;
            }

            Follow(Time.unscaledDeltaTime);
        }

        private void OnDestroy()
        {
            if (_reviewFrozen)
            {
                SetReviewFrozen(false);
            }
        }

#if UNITY_MLAGENTS
        private void EnsureChannel()
        {
            if (_channel != null)
            {
                return;
            }

            try
            {
                _channel = SkillPromoteDebugChannel.Register();
            }
            catch (System.Exception)
            {
                _channel = new SkillPromoteDebugChannel();
            }
        }

        private void OnShow()
        {
            if (_channel == null)
            {
                return;
            }

            Show(
                _channel.LastRequestId,
                _channel.QueueCount,
                _channel.SkillId,
                _channel.SkillName,
                _channel.Tags,
                _channel.Score,
                _channel.StepsCount,
                _channel.SuggestedName,
                _channel.AllowLlm);
        }

        private void OnHide()
        {
            Hide();
        }
#endif

        /// <summary>EditMode helper without SideChannel.</summary>
        public void ShowForTests(
            int requestId,
            int queueCount,
            string skillId,
            string name,
            string tags,
            string score,
            string stepsCount,
            string suggestedName = "",
            bool allowLlm = true)
        {
            Show(requestId, queueCount, skillId, name, tags, score, stepsCount, suggestedName, allowLlm);
        }

        public void Show(
            int requestId,
            int queueCount,
            string skillId,
            string name,
            string tags,
            string score,
            string stepsCount,
            string suggestedName,
            bool allowLlm)
        {
            EnsureUi();
            _visible = true;
            _awaitingLlmJudge = false;
            _lockMotionForJudge = false;
            _allowLlm = allowLlm;
            _awaySeconds = 0f;
            _hasViewingReport = false;
            _pendingRequestId = requestId;
            ResolveTarget();
            SnapToTarget();
            bool viewing = IsUserViewingDisplay4();
            ReportViewingIfChanged(viewing);
            // Freeze only when the user is on Display 4 for HITL; otherwise keep sim live
            // and TickAwayAutoJudge fires Ask LLM immediately (no motion interrupt).
            SetReviewFrozen(viewing);
            if (!viewing)
            {
                ReleaseMotorGate();
            }
            if (_panelRoot != null)
            {
                _panelRoot.SetActive(true);
            }

            if (_title != null)
            {
                _title.text = viewing
                    ? $"Skill promote · {Mathf.Max(0, queueCount)} ready"
                    : $"Skill promote · {Mathf.Max(0, queueCount)} ready (auto LLM)";
            }

            if (_meta != null)
            {
                _meta.text = string.IsNullOrEmpty(skillId)
                    ? "candidate: (none)"
                    : $"id: {skillId}";
            }

            if (_detail != null)
            {
                _detail.text =
                    $"name: {name}\n" +
                    $"tags: {(string.IsNullOrEmpty(tags) ? "-" : tags)}\n" +
                    $"score: {score}   steps: {stepsCount}";
            }

            if (_nameInput != null)
            {
                string seed = !string.IsNullOrEmpty(suggestedName)
                    ? suggestedName
                    : (!string.IsNullOrEmpty(name) ? name : "");
                if (seed.StartsWith("candidate_"))
                {
                    seed = seed.Substring("candidate_".Length);
                }

                _nameInput.text = seed;
                _nameInput.ActivateInputField();
            }

            if (_askLlm != null)
            {
                _askLlm.gameObject.SetActive(allowLlm);
                _askLlm.interactable = allowLlm;
            }
        }

        public void Hide()
        {
            _visible = false;
            _awaitingLlmJudge = false;
            _lockMotionForJudge = false;
            _awaySeconds = 0f;
            _motorTestRemaining = 0f;
            if (_panelRoot != null)
            {
                _panelRoot.SetActive(false);
            }

            SetReviewFrozen(false);
            // Ask-LLM path suspends actions while _reviewFrozen is already false;
            // SetReviewFrozen(false) early-returns and would leave the fly stuck.
            ReleaseMotorGate();
        }

        /// <summary>Clear kinematic / action suspend unless profile menu still owns the gate.</summary>
        private void ReleaseMotorGate()
        {
            if (!FlyProfileMenu.IsSessionReady)
            {
                return;
            }

            SetBodiesKinematic(false);
            if (_motorTestRemaining <= 0f)
            {
                SuspendPythonActions(false);
            }
        }

        public void Bind(Transform fly)
        {
            target = fly;
            _hadTarget = false;
            SnapToTarget();
        }

        /// <summary>EditMode helper: snap portrait framing immediately.</summary>
        public void SnapToTargetForTests()
        {
            ResolveTarget();
            SnapToTarget();
        }

        /// <summary>EditMode / UI: start a short local motor bout (also KeyCode.T).</summary>
        public void RunMotorTest()
        {
            if (!FlyProfileMenu.IsSessionReady)
            {
                if (_hudText != null)
                {
                    _hudText.text = "Choose a fly first — then press T to test motors";
                }

                return;
            }

            ResolveMotors();
            if (_motors == null)
            {
                if (_hudText != null)
                {
                    _hudText.text = "No FlyMotorController — cannot test";
                }

                return;
            }

            // Temporarily unfreeze so the bout is visible, then freeze again.
            SetReviewFrozen(false);
            SuspendPythonActions(true);
            _motorTestRemaining = Mathf.Max(0.5f, motorTestSeconds);
            ApplyMotorTestPose(0f);

            if (_visible)
            {
                SendReply(SkillPromoteSource.Test, "");
            }
        }

        private void SetReviewFrozen(bool freeze)
        {
            if (freeze == _reviewFrozen)
            {
                if (freeze)
                {
                    SnapToTarget();
                    ZeroVelocities();
                }

                return;
            }

            _reviewFrozen = freeze;
            if (freeze)
            {
                if (Time.timeScale > 0.0001f)
                {
                    _savedTimeScale = Time.timeScale;
                }

                Time.timeScale = 0f;
                SetBodiesKinematic(true);
                ZeroVelocities();
                SuspendPythonActions(true);
                SnapToTarget();
            }
            else
            {
                // Keep frozen if the profile menu still owns the gate.
                if (!FlyProfileMenu.IsSessionReady)
                {
                    _reviewFrozen = true;
                    return;
                }

                Time.timeScale = _savedTimeScale > 0f ? _savedTimeScale : 1f;
                SetBodiesKinematic(false);
                if (_motorTestRemaining <= 0f)
                {
                    SuspendPythonActions(false);
                }
            }
        }

        private void SetBodiesKinematic(bool kinematic)
        {
            var bodies = FindObjectsByType<Rigidbody>(FindObjectsSortMode.None);
            for (int i = 0; i < bodies.Length; i++)
            {
                if (bodies[i] == null)
                {
                    continue;
                }

                bodies[i].isKinematic = kinematic;
            }
        }

        private void ZeroVelocities()
        {
            var bodies = FindObjectsByType<Rigidbody>(FindObjectsSortMode.None);
            for (int i = 0; i < bodies.Length; i++)
            {
                if (bodies[i] == null)
                {
                    continue;
                }

                bodies[i].linearVelocity = Vector3.zero;
                bodies[i].angularVelocity = Vector3.zero;
            }

            if (_motors != null)
            {
                _motors.SetHighLevelCommands(0f, 0f, 0.5f, 0.5f);
                _motors.SetProboscisExtension(0f);
            }
        }

        private static void SuspendPythonActions(bool suspend)
        {
#if UNITY_MLAGENTS
            var agents = FindObjectsByType<FlyAgent>(FindObjectsSortMode.None);
            for (int i = 0; i < agents.Length; i++)
            {
                if (agents[i] != null)
                {
                    agents[i].SetActionsSuspended(suspend);
                }
            }
#else
            _ = suspend;
#endif
        }

        private void SubmitAccept()
        {
            string name = _nameInput != null ? _nameInput.text : "";
            SendReply(SkillPromoteSource.Accept, name);
        }

        private void SubmitAskLlm(bool lockMotion = true)
        {
            if (_awaitingLlmJudge)
            {
                return;
            }

            string name = _nameInput != null ? _nameInput.text : "";
            SendReply(SkillPromoteSource.AskLlm, name);
            BeginAwaitingLlmJudge(lockMotion);
        }

        /// <summary>EditMode: mirror Ask LLM click (thaw clock, keep panel).</summary>
        public void SubmitAskLlmForTests()
        {
            SubmitAskLlm(lockMotion: true);
        }

        /// <summary>
        /// timeScale=0 stops FixedUpdate, so ML-Agents never flushes SideChannel
        /// replies. Thaw the clock for Ask LLM. When <paramref name="lockMotion"/> is
        /// true (HITL on Display 4), keep the fly kinematic until Hide/Show.
        /// Away auto-judge passes false so walking continues.
        /// </summary>
        private void BeginAwaitingLlmJudge(bool lockMotion)
        {
            _awaitingLlmJudge = true;
            _lockMotionForJudge = lockMotion;
            if (_title != null)
            {
                _title.text = lockMotion ? "Ask LLM judging…" : "Away → Ask LLM judging…";
            }

            if (_askLlm != null)
            {
                _askLlm.interactable = false;
            }

            // Thaw the clock so SideChannels flush.
            _reviewFrozen = false;
            if (Time.timeScale <= 0.0001f)
            {
                Time.timeScale = _savedTimeScale > 0f ? _savedTimeScale : 1f;
            }

            if (lockMotion)
            {
                SetBodiesKinematic(true);
                ZeroVelocities();
                SuspendPythonActions(true);
                SnapToTarget();
            }
            else
            {
                ReleaseMotorGate();
            }
        }

        private void SubmitRefuse()
        {
            SendReply(SkillPromoteSource.Refuse, "");
        }

        private void SubmitTest()
        {
            RunMotorTest();
        }

        private void SendReply(int source, string name)
        {
#if UNITY_MLAGENTS
            EnsureChannel();
            if (_channel != null)
            {
                _channel.SendReply(_pendingRequestId, source, name ?? "");
            }
#endif
            // Stay visible until Python HIDE / next SHOW (non-blocking queue).
            if (source == SkillPromoteSource.Refuse || source == SkillPromoteSource.Accept)
            {
                // Brief hide; Python will SHOW next candidate or HIDE when empty.
                Hide();
            }
        }

        private void ResolveMotors()
        {
            if (_motors != null)
            {
                return;
            }

            if (target != null)
            {
                _motors = target.GetComponentInChildren<FlyMotorController>();
            }

            if (_motors == null)
            {
                _motors = FindFirstObjectByType<FlyMotorController>();
            }
        }

        private void TickMotorTest(float dt)
        {
            if (_motorTestRemaining <= 0f)
            {
                return;
            }

            ResolveMotors();
            if (_motors == null)
            {
                _motorTestRemaining = 0f;
                return;
            }

            float elapsed = motorTestSeconds - _motorTestRemaining;
            ApplyMotorTestPose(elapsed);
            _motorTestRemaining -= Mathf.Max(0f, dt);
            if (_motorTestRemaining <= 0f)
            {
                _motors.SetHighLevelCommands(0f, 0f, 0.5f, 0.5f);
                _motors.SetProboscisExtension(0f);
                // Back to frozen centered review if the promote panel is still open.
                if (_visible)
                {
                    SetReviewFrozen(true);
                }
                else
                {
                    SuspendPythonActions(false);
                }
            }
        }

        private void ApplyMotorTestPose(float elapsed)
        {
            if (_motors == null)
            {
                return;
            }

            // Visible spin + step bout so Display 4 shows legs/body moving.
            float turn = Mathf.Sin(elapsed * 4f) * 0.85f;
            float forward = 0.35f + 0.25f * Mathf.Sin(elapsed * 6f);
            _motors.SetHighLevelCommands(forward, turn, 0.55f, 0.85f);
            _motors.SetProboscisExtension(elapsed < 1.2f ? 0.7f : 0f);
        }

        private void RefreshHud()
        {
            if (_hudText == null)
            {
                return;
            }

            string track = target != null
                ? $"fly centered · {target.name}"
                : "no fly target — waiting…";
            string freeze = _reviewFrozen && _motorTestRemaining <= 0f ? " · FROZEN" : "";
            string judge = _awaitingLlmJudge
                ? (_lockMotionForJudge ? " · LLM JUDGING" : " · AUTO LLM")
                : "";
            string away = !_awaitingLlmJudge && _visible && !IsUserViewingDisplay4()
                ? (autoJudgeAwaySeconds <= 0f
                    ? " · AWAY auto-LLM"
                    : $" · AWAY auto-LLM {Mathf.Max(0f, autoJudgeAwaySeconds - _awaySeconds):0.0}s")
                : "";
            string test = _motorTestRemaining > 0f
                ? $" · TEST {_motorTestRemaining:0.0}s"
                : " · T = test motors";
            _hudText.text = $"Display 4 · {track}{freeze}{judge}{away}{test}";
        }

        private void ResolveTarget()
        {
            if (target != null)
            {
                return;
            }

            var body = FindFirstObjectByType<FlyBodyController>();
            if (body != null)
            {
                target = body.transform;
            }
        }

        private void TryActivateDisplay4()
        {
            if (!activateDisplay4 || _displayActivated)
            {
                return;
            }

            // Editor Game view can show Display 4 without a physical monitor.
            // Activate() only when a real multi-display device is present.
            if (Display.displays.Length > TargetDisplayIndex)
            {
                try
                {
                    Display.displays[TargetDisplayIndex].Activate();
                }
                catch (System.Exception)
                {
                    // Ignore — targetDisplay still drives the Game view tab.
                }
            }

            _displayActivated = true;
        }

        private void SnapToTarget()
        {
            if (target == null || Camera == null)
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

        private Vector3 FlyCenter()
        {
            if (target == null)
            {
                return Vector3.zero;
            }

            // Prefer mesh/collider bounds so the body sits in the screen center.
            var renderers = target.GetComponentsInChildren<Renderer>();
            if (renderers != null && renderers.Length > 0)
            {
                Bounds bounds = renderers[0].bounds;
                for (int i = 1; i < renderers.Length; i++)
                {
                    if (renderers[i] != null)
                    {
                        bounds.Encapsulate(renderers[i].bounds);
                    }
                }

                return bounds.center;
            }

            return target.TransformPoint(lookAtOffset);
        }

        private Vector3 DesiredPosition()
        {
            Vector3 center = FlyCenter();
            // World-space portrait: behind the fly's facing, slightly elevated.
            Vector3 back = -target.forward;
            back.y = 0f;
            if (back.sqrMagnitude < 1e-6f)
            {
                back = Vector3.back;
            }

            back.Normalize();
            float distance = Mathf.Abs(localOffset.z);
            if (distance < 0.05f)
            {
                distance = 0.14f;
            }

            float height = Mathf.Max(0.02f, localOffset.y);
            return center + back * distance + Vector3.up * height;
        }

        private Quaternion DesiredRotation()
        {
            Vector3 lookAt = FlyCenter();
            Vector3 forward = lookAt - transform.position;
            if (forward.sqrMagnitude < 1e-8f)
            {
                forward = target != null ? target.forward : Vector3.forward;
            }

            return Quaternion.LookRotation(forward.normalized, Vector3.up);
        }

        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null)
            {
                return;
            }

            var go = new GameObject("EventSystem");
            go.AddComponent<EventSystem>();
            go.AddComponent<StandaloneInputModule>();
        }

        private void EnsureUi()
        {
            if (_canvas != null && _panelRoot != null && _hudRoot != null)
            {
                _canvas.targetDisplay = TargetDisplayIndex;
                return;
            }

            var root = new GameObject("SkillPromoteCanvas");
            root.transform.SetParent(transform, false);
            _canvas = root.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.targetDisplay = TargetDisplayIndex;
            _canvas.sortingOrder = 100;
            var scaler = root.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280f, 720f);
            root.AddComponent<GraphicRaycaster>();

            // Always-on HUD so Display 4 shows tracking status + Test hint.
            _hudRoot = new GameObject("Hud");
            _hudRoot.transform.SetParent(root.transform, false);
            var hudBg = _hudRoot.AddComponent<Image>();
            hudBg.color = new Color(0.05f, 0.06f, 0.08f, 0.65f);
            var hudRt = hudBg.rectTransform;
            hudRt.anchorMin = new Vector2(0.04f, 0.9f);
            hudRt.anchorMax = new Vector2(0.96f, 0.98f);
            hudRt.offsetMin = Vector2.zero;
            hudRt.offsetMax = Vector2.zero;
            _hudText = CreateText(_hudRoot.transform, "HudText", 16, FontStyle.Bold, TextAnchor.MiddleLeft);
            var hudTextRt = _hudText.rectTransform;
            hudTextRt.anchorMin = Vector2.zero;
            hudTextRt.anchorMax = Vector2.one;
            hudTextRt.offsetMin = new Vector2(12f, 0f);
            hudTextRt.offsetMax = new Vector2(-12f, 0f);
            _hudText.text = "Display 4 · fly centered · T = test motors";

            // Bottom panel; upper ~55% stays clear so the centered fly stays visible.
            _panelRoot = new GameObject("PromotePanel");
            _panelRoot.transform.SetParent(root.transform, false);
            var panelImage = _panelRoot.AddComponent<Image>();
            panelImage.color = panelColor;
            var panelRt = panelImage.rectTransform;
            panelRt.anchorMin = new Vector2(0.04f, 0.02f);
            panelRt.anchorMax = new Vector2(0.96f, 0.38f);
            panelRt.offsetMin = Vector2.zero;
            panelRt.offsetMax = Vector2.zero;

            _title = CreateText(_panelRoot.transform, "Title", 20, FontStyle.Bold, TextAnchor.UpperLeft);
            PlaceText(_title, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -10f), new Vector2(-14f, 30f));

            _meta = CreateText(_panelRoot.transform, "Meta", 14, FontStyle.Italic, TextAnchor.UpperLeft);
            _meta.color = accentColor;
            PlaceText(_meta, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -40f), new Vector2(-14f, 22f));

            _detail = CreateText(_panelRoot.transform, "Detail", 15, FontStyle.Normal, TextAnchor.UpperLeft);
            PlaceText(_detail, new Vector2(0f, 0.48f), new Vector2(1f, 1f), new Vector2(14f, -68f), new Vector2(-14f, -4f));
            _detail.horizontalOverflow = HorizontalWrapMode.Wrap;
            _detail.verticalOverflow = VerticalWrapMode.Overflow;

            var inputGo = new GameObject("NameInput");
            inputGo.transform.SetParent(_panelRoot.transform, false);
            var inputBg = inputGo.AddComponent<Image>();
            inputBg.color = new Color(0.04f, 0.05f, 0.07f, 1f);
            var inputRt = inputBg.rectTransform;
            inputRt.anchorMin = new Vector2(0f, 0.22f);
            inputRt.anchorMax = new Vector2(1f, 0.42f);
            inputRt.offsetMin = new Vector2(14f, 0f);
            inputRt.offsetMax = new Vector2(-14f, 0f);

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(inputGo.transform, false);
            var inputText = textGo.AddComponent<Text>();
            StyleText(inputText, 16, FontStyle.Normal);
            inputText.supportRichText = false;
            inputText.alignment = TextAnchor.MiddleLeft;
            var textRt = inputText.rectTransform;
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = new Vector2(10f, 4f);
            textRt.offsetMax = new Vector2(-10f, -4f);

            var placeholderGo = new GameObject("Placeholder");
            placeholderGo.transform.SetParent(inputGo.transform, false);
            var placeholder = placeholderGo.AddComponent<Text>();
            StyleText(placeholder, 16, FontStyle.Italic);
            placeholder.color = new Color(textColor.r, textColor.g, textColor.b, 0.45f);
            placeholder.text = "Name this skill (required to Accept)…";
            placeholder.alignment = TextAnchor.MiddleLeft;
            var phRt = placeholder.rectTransform;
            phRt.anchorMin = Vector2.zero;
            phRt.anchorMax = Vector2.one;
            phRt.offsetMin = new Vector2(10f, 4f);
            phRt.offsetMax = new Vector2(-10f, -4f);

            _nameInput = inputGo.AddComponent<InputField>();
            _nameInput.textComponent = inputText;
            _nameInput.placeholder = placeholder;
            _nameInput.lineType = InputField.LineType.SingleLine;

            _accept = CreateButton(_panelRoot.transform, "Accept", "Accept", new Vector2(0.02f, 0.04f), new Vector2(0.24f, 0.18f));
            _accept.onClick.AddListener(SubmitAccept);

            _test = CreateButton(_panelRoot.transform, "Test", "Test (T)", new Vector2(0.26f, 0.04f), new Vector2(0.48f, 0.18f));
            _test.onClick.AddListener(SubmitTest);

            _askLlm = CreateButton(_panelRoot.transform, "AskLlm", "Ask LLM judge", new Vector2(0.50f, 0.04f), new Vector2(0.74f, 0.18f));
            _askLlm.onClick.AddListener(() => SubmitAskLlm(lockMotion: true));

            _refuse = CreateButton(_panelRoot.transform, "Refuse", "Refuse", new Vector2(0.76f, 0.04f), new Vector2(0.98f, 0.18f));
            _refuse.onClick.AddListener(SubmitRefuse);
        }

        private Text CreateText(Transform parent, string name, int size, FontStyle style, TextAnchor align)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var text = go.AddComponent<Text>();
            StyleText(text, size, style);
            text.alignment = align;
            text.raycastTarget = false;
            return text;
        }

        private void StyleText(Text text, int size, FontStyle style)
        {
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (text.font == null)
            {
                text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }

            text.fontSize = size;
            text.fontStyle = style;
            text.color = textColor;
            text.supportRichText = false;
        }

        private static void PlaceText(Text text, Vector2 anchorMin, Vector2 anchorMax, Vector2 anchoredPos, Vector2 size)
        {
            var rt = text.rectTransform;
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = anchoredPos;
            rt.sizeDelta = size;
        }

        private Button CreateButton(Transform parent, string name, string label, Vector2 anchorMin, Vector2 anchorMax)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<Image>();
            image.color = new Color(0.18f, 0.3f, 0.22f, 1f);
            var rt = image.rectTransform;
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;

            var button = go.AddComponent<Button>();
            var text = CreateText(go.transform, "Label", 16, FontStyle.Bold, TextAnchor.MiddleCenter);
            var textRt = text.rectTransform;
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.offsetMin = Vector2.zero;
            textRt.offsetMax = Vector2.zero;
            text.text = label;
            text.raycastTarget = false;
            return button;
        }

        public static void Place(Camera cam, float fov = 50f)
        {
            if (cam == null)
            {
                return;
            }

            cam.enabled = true;
            cam.tag = "Untagged";
            cam.targetDisplay = TargetDisplayIndex;
            cam.clearFlags = RenderSettings.skybox != null ? CameraClearFlags.Skybox : CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.1f, 0.12f, 0.14f);
            cam.fieldOfView = fov;
            // Avoid ultra-small near clips — they can blank the Game view on Metal.
            cam.nearClipPlane = 0.01f;
            cam.farClipPlane = Mathf.Max(OutsideWorldLayout.ViewDistance, 20f);
            cam.depth = 55f;
            cam.cullingMask = ChamberWallHider.OverviewCullingMask;
            cam.targetTexture = null;
        }

        public static bool IsPromoteCamera(Camera cam)
        {
            if (cam == null)
            {
                return false;
            }

            return cam.GetComponent<SkillPromoteDisplay>() != null || cam.gameObject.name == ObjectName;
        }
    }

    internal static class SkillPromoteSource
    {
        public const int Accept = 0;
        public const int AskLlm = 1;
        public const int Refuse = 2;
        public const int Test = 3;
    }
}
