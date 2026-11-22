using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using FlySimulation.Experiments;
using UnityEngine;
#if UNITY_MLAGENTS
using FlySimulation.Fly.Brain;
#endif

namespace FlySimulation.Environment
{
    /// <summary>
    /// Pre-play roster: pick Fly 1 / 2 / 3 … or create a new profile before the run starts.
    /// Uses OnGUI so the menu is always visible in the Game view (no uGUI font/canvas issues).
    /// Shares <c>fly_profiles.json</c> with Python (<c>data/</c> + StreamingAssets mirror).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FlyProfileMenu : MonoBehaviour
    {
        public const string ObjectName = "FlyProfileMenu";
        public const string FileName = "fly_profiles.json";

        [SerializeField] private bool showOnPlay = true;

        private ExperimentConfiguration _config;
        private bool _ready;
        private bool _visible;
        private readonly List<Profile> _profiles = new List<Profile>();
        private string _activeFlyId = "fly_001";
        private string _status = "Choose a fly to begin.";
        private Vector2 _scroll;
        private GUIStyle _titleStyle;
        private GUIStyle _buttonStyle;
        private GUIStyle _statusStyle;
        private bool _stylesReady;

        [Serializable]
        private sealed class Profile
        {
            public string fly_id;
            public string name;
            public int number;
            public string created_at;
            public string last_played_at;
            public string notes;
        }

        public bool IsReady => _ready;
        public string ActiveFlyId => _activeFlyId;

        private void Awake()
        {
            _config = FindExperimentConfig();
            LoadProfiles();
            if (showOnPlay)
            {
                Show();
            }
            else
            {
                ApplyActiveAndReady();
            }
        }

        private void Show()
        {
            _ready = false;
            _visible = true;
            _status = "Choose a fly to begin.";
            PauseFlyControl(true);
            Debug.Log($"FlyProfileMenu: showing roster ({_profiles.Count} profiles), active={_activeFlyId}");
        }

        private void Hide()
        {
            _visible = false;
        }

        private void ApplyActiveAndReady()
        {
            ApplyFlyId(_activeFlyId, touch: true);
            _ready = true;
            Hide();
            PauseFlyControl(false);
        }

        private void SelectProfile(Profile profile)
        {
            if (profile == null)
            {
                return;
            }

            _activeFlyId = profile.fly_id;
            ApplyActiveAndReady();
        }

        private void CreateNewProfile()
        {
            int next = 1;
            for (int i = 0; i < _profiles.Count; i++)
            {
                next = Mathf.Max(next, _profiles[i].number + 1);
            }

            var profile = new Profile
            {
                number = next,
                fly_id = $"fly_{next:000}",
                name = $"Fly {next}",
                created_at = DateTime.UtcNow.ToString("o"),
                notes = ""
            };
            _profiles.Add(profile);
            _activeFlyId = profile.fly_id;
            SaveProfiles();
            SelectProfile(profile);
        }

        private void ApplyFlyId(string flyId, bool touch)
        {
            _activeFlyId = string.IsNullOrEmpty(flyId) ? "fly_001" : flyId;
            if (_config != null)
            {
                _config.FlyId = _activeFlyId;
            }

            for (int i = 0; i < _profiles.Count; i++)
            {
                if (_profiles[i].fly_id == _activeFlyId)
                {
                    if (touch)
                    {
                        _profiles[i].last_played_at = DateTime.UtcNow.ToString("o");
                    }

                    break;
                }
            }

            SaveProfiles();
            _status = $"Playing as {_activeFlyId}";
            Debug.Log($"FlyProfileMenu: active={_activeFlyId}");
        }

        private void PauseFlyControl(bool pause)
        {
#if UNITY_MLAGENTS
            var demo = UnityEngine.Object.FindFirstObjectByType<WalkingDemoDriver>();
            if (demo != null)
            {
                demo.enabled = !pause;
            }
#endif
            var bodies = UnityEngine.Object.FindObjectsByType<Rigidbody>(FindObjectsSortMode.None);
            for (int i = 0; i < bodies.Length; i++)
            {
                if (bodies[i] == null)
                {
                    continue;
                }

                bodies[i].isKinematic = pause;
                if (pause)
                {
                    bodies[i].linearVelocity = Vector3.zero;
                    bodies[i].angularVelocity = Vector3.zero;
                }
            }
        }

        private static ExperimentConfiguration FindExperimentConfig()
        {
            var behaviours = UnityEngine.Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
            for (int i = 0; i < behaviours.Length; i++)
            {
                var mb = behaviours[i];
                if (mb == null)
                {
                    continue;
                }

                var field = mb.GetType().GetField(
                    "configuration",
                    System.Reflection.BindingFlags.Instance
                    | System.Reflection.BindingFlags.Public
                    | System.Reflection.BindingFlags.NonPublic);
                if (field != null && field.FieldType == typeof(ExperimentConfiguration))
                {
                    return field.GetValue(mb) as ExperimentConfiguration;
                }
            }

            return null;
        }

        private void LoadProfiles()
        {
            _profiles.Clear();
            string json = null;
            string streaming = Path.Combine(Application.streamingAssetsPath, FileName);
            string dataPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "data", FileName));
            if (File.Exists(streaming))
            {
                json = File.ReadAllText(streaming);
            }
            else if (File.Exists(dataPath))
            {
                json = File.ReadAllText(dataPath);
            }

            if (!string.IsNullOrEmpty(json))
            {
                ParseProfilesJson(json);
            }

            if (_profiles.Count == 0)
            {
                _profiles.Add(new Profile
                {
                    fly_id = "fly_001",
                    name = "Fly 1",
                    number = 1,
                    created_at = DateTime.UtcNow.ToString("o")
                });
                _activeFlyId = "fly_001";
                SaveProfiles();
            }
        }

        /// <summary>
        /// Regex parse so Python JSON (nulls / pretty print) works without JsonUtility quirks.
        /// </summary>
        private void ParseProfilesJson(string json)
        {
            try
            {
                Match active = Regex.Match(json, "\"active_fly_id\"\\s*:\\s*\"([^\"]+)\"");
                if (active.Success)
                {
                    _activeFlyId = active.Groups[1].Value;
                }

                foreach (Match block in Regex.Matches(
                             json,
                             "\\{[^{}]*\"fly_id\"\\s*:\\s*\"([^\"]+)\"[^{}]*\\}",
                             RegexOptions.Singleline))
                {
                    string body = block.Value;
                    string flyId = MatchGroup(body, "\"fly_id\"\\s*:\\s*\"([^\"]+)\"");
                    if (string.IsNullOrEmpty(flyId))
                    {
                        continue;
                    }

                    string name = MatchGroup(body, "\"name\"\\s*:\\s*\"([^\"]+)\"");
                    string numberText = MatchGroup(body, "\"number\"\\s*:\\s*(\\d+)");
                    int number = 1;
                    if (!int.TryParse(numberText, out number))
                    {
                        Match numFromId = Regex.Match(flyId, @"(\d+)$");
                        number = numFromId.Success ? int.Parse(numFromId.Groups[1].Value) : _profiles.Count + 1;
                    }

                    _profiles.Add(new Profile
                    {
                        fly_id = flyId,
                        name = string.IsNullOrEmpty(name) ? $"Fly {number}" : name,
                        number = number,
                        created_at = MatchGroup(body, "\"created_at\"\\s*:\\s*\"([^\"]+)\""),
                        last_played_at = MatchGroup(body, "\"last_played_at\"\\s*:\\s*\"([^\"]+)\""),
                        notes = MatchGroup(body, "\"notes\"\\s*:\\s*\"([^\"]*)\"")
                    });
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"FlyProfileMenu: failed to parse profiles: {ex.Message}");
            }
        }

        private static string MatchGroup(string body, string pattern)
        {
            Match m = Regex.Match(body, pattern);
            return m.Success ? m.Groups[1].Value : "";
        }

        private void SaveProfiles()
        {
            _profiles.Sort((a, b) => a.number.CompareTo(b.number));
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine($"  \"active_fly_id\": \"{_activeFlyId}\",");
            sb.AppendLine("  \"profiles\": [");
            for (int i = 0; i < _profiles.Count; i++)
            {
                Profile p = _profiles[i];
                sb.AppendLine("    {");
                sb.AppendLine($"      \"fly_id\": \"{Escape(p.fly_id)}\",");
                sb.AppendLine($"      \"name\": \"{Escape(p.name)}\",");
                sb.AppendLine($"      \"number\": {p.number},");
                sb.AppendLine($"      \"created_at\": \"{Escape(p.created_at)}\",");
                if (string.IsNullOrEmpty(p.last_played_at))
                {
                    sb.AppendLine("      \"last_played_at\": null,");
                }
                else
                {
                    sb.AppendLine($"      \"last_played_at\": \"{Escape(p.last_played_at)}\",");
                }

                sb.AppendLine($"      \"notes\": \"{Escape(p.notes)}\"");
                sb.Append("    }");
                sb.AppendLine(i + 1 < _profiles.Count ? "," : "");
            }

            sb.AppendLine("  ]");
            sb.AppendLine("}");
            string json = sb.ToString();

            try
            {
                Directory.CreateDirectory(Application.streamingAssetsPath);
                File.WriteAllText(Path.Combine(Application.streamingAssetsPath, FileName), json);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"FlyProfileMenu: could not write StreamingAssets: {ex.Message}");
            }

            try
            {
                string dataDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "data"));
                Directory.CreateDirectory(dataDir);
                File.WriteAllText(Path.Combine(dataDir, FileName), json);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"FlyProfileMenu: could not write data/: {ex.Message}");
            }
        }

        private static string Escape(string value)
        {
            return (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"");
        }

        private void EnsureStyles()
        {
            if (_stylesReady)
            {
                return;
            }

            _titleStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 28,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                normal = { textColor = Color.white }
            };
            _statusStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 16,
                alignment = TextAnchor.MiddleCenter,
                wordWrap = true,
                normal = { textColor = new Color(0.85f, 0.9f, 0.95f, 1f) }
            };
            _buttonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 20,
                fontStyle = FontStyle.Bold,
                fixedHeight = 48f,
                alignment = TextAnchor.MiddleCenter
            };
            _stylesReady = true;
        }

        private void OnGUI()
        {
            if (!_visible || Event.current == null)
            {
                return;
            }

            EnsureStyles();

            // Dim the whole game view so the menu is obvious.
            Color prev = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.72f);
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.whiteTexture);
            GUI.color = prev;

            float width = Mathf.Min(520f, Screen.width - 40f);
            float height = Mathf.Min(420f, Screen.height - 40f);
            var box = new Rect((Screen.width - width) * 0.5f, (Screen.height - height) * 0.5f, width, height);
            GUI.Box(box, GUIContent.none);

            GUILayout.BeginArea(new Rect(box.x + 20f, box.y + 16f, box.width - 40f, box.height - 32f));
            GUILayout.Label("Choose your fly", _titleStyle);
            GUILayout.Space(6f);
            GUILayout.Label(_status, _statusStyle);
            GUILayout.Space(12f);

            _profiles.Sort((a, b) => a.number.CompareTo(b.number));
            _scroll = GUILayout.BeginScrollView(_scroll, GUILayout.ExpandHeight(true));
            for (int i = 0; i < _profiles.Count; i++)
            {
                Profile profile = _profiles[i];
                bool active = profile.fly_id == _activeFlyId;
                string label = $"{profile.name}  ({profile.fly_id})" + (active ? "  ★" : "");
                if (GUILayout.Button(label, _buttonStyle))
                {
                    SelectProfile(profile);
                }

                GUILayout.Space(8f);
            }

            if (GUILayout.Button("+ Create new fly", _buttonStyle))
            {
                CreateNewProfile();
            }

            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }
    }
}
