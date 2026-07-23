"""Phase 3 exit-criteria constants mirrored from Unity Phase3ExitCriteria."""

OBSERVATIONS_FILE = "observations.csv"
ACTIONS_FILE = "actions.csv"
INTERNAL_STATE_FILE = "internal_state.csv"
JOINTS_FILE = "joints.csv"
CONTACTS_FILE = "contacts.csv"
TASTE_FILE = "taste.csv"
ODOR_FILE = "odor.csv"
POSE_FILE = "pose.csv"
EVENTS_FILE = "events.jsonl"
METADATA_FILE = "metadata.json"
ENVIRONMENT_FILE = "environment.json"
VISION_MANIFEST_FILE = "vision/manifest.jsonl"
EPISODE_SUMMARY_FILE = "episode_summary.json"

COMMON_CSV_HEADER = "experiment_id,episode_id,step,sim_time,seed,fly_id"

SCHEMA_VERSION = 3
DEFAULT_VISION_WIDTH = 64
DEFAULT_VISION_HEIGHT = 64

REQUIRED_STREAM_FILES = (
    OBSERVATIONS_FILE,
    ACTIONS_FILE,
    INTERNAL_STATE_FILE,
    JOINTS_FILE,
    CONTACTS_FILE,
    TASTE_FILE,
    ODOR_FILE,
    POSE_FILE,
    EVENTS_FILE,
    METADATA_FILE,
)
