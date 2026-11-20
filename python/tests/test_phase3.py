from __future__ import annotations

import json
from pathlib import Path

from flybrain.phase3 import (
    ACTIONS_FILE,
    COMMON_CSV_HEADER,
    CONTACTS_FILE,
    EVENTS_FILE,
    JOINTS_FILE,
    METADATA_FILE,
    OBSERVATIONS_FILE,
    ODOR_FILE,
    POSE_FILE,
    SCHEMA_VERSION,
    TASTE_FILE,
    VISION_MANIFEST_FILE,
)
from flybrain.recording import EpisodeReplay


def _write_minimal_episode(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "vision").mkdir(exist_ok=True)

    (root / METADATA_FILE).write_text(
        json.dumps(
            {
                "experiment_id": "phase3_py",
                "episode_id": 1,
                "seed": 7,
                "fly_id": "fly_0",
                "recording_hz": 50.0,
                "vision_hz": 30.0,
                "schema_version": SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )

    header = COMMON_CSV_HEADER + ",pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,fwd_x,fwd_y,fwd_z,"
    header += "left_eye_fwd_x,left_eye_fwd_y,left_eye_fwd_z,right_eye_fwd_x,right_eye_fwd_y,right_eye_fwd_z"
    (root / POSE_FILE).write_text(
        header
        + "\n"
        + "phase3_py,1,0,0.000000,7,fly_0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1\n"
        + "phase3_py,1,4,0.020000,7,fly_0,0.01,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1\n",
        encoding="utf-8",
    )

    taste_header = COMMON_CSV_HEADER
    for i in range(6):
        for ch in ("sugar", "bitter", "salt", "acid", "water", "protein", "toxin", "odor"):
            taste_header += f",foot{i}_{ch}"
    taste_header += ",proboscis_contact"
    for ch in ("sugar", "bitter", "salt", "acid", "water", "protein", "toxin", "odor"):
        taste_header += f",proboscis_{ch}"
    zeros = ",".join(["0"] * (6 * 8))
    (root / TASTE_FILE).write_text(
        taste_header
        + "\n"
        + f"phase3_py,1,0,0.000000,7,fly_0,{zeros},0,0,0,0,0,0,0,0,0\n"
        + f"phase3_py,1,4,0.020000,7,fly_0,{zeros},1,0.5,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    for name in (
        OBSERVATIONS_FILE,
        ACTIONS_FILE,
        "internal_state.csv",
        JOINTS_FILE,
        CONTACTS_FILE,
        ODOR_FILE,
    ):
        (root / name).write_text(
            COMMON_CSV_HEADER + "\nphase3_py,1,0,0.000000,7,fly_0\n", encoding="utf-8"
        )

    (root / EVENTS_FILE).write_text(
        '{"experiment_id":"phase3_py","episode_id":1,"step":0,"timestamp":0.000000,"event":"episode_started"}\n'
        '{"experiment_id":"phase3_py","episode_id":1,"step":4,"timestamp":0.020000,"event":"episode_ended"}\n',
        encoding="utf-8",
    )

    (root / VISION_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "step": 0,
                "timestamp": 0.0,
                "frame_index": 0,
                "left_file": "vision/left_000000.png",
                "right_file": "vision/right_000000.png",
                "body_forward": [0.0, 0.0, 1.0],
                "left_eye_forward": [0.0, 0.0, 1.0],
                "right_eye_forward": [0.0, 0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase3_constants_mirror_unity() -> None:
    assert SCHEMA_VERSION == 3
    assert TASTE_FILE == "taste.csv"
    assert VISION_MANIFEST_FILE == "vision/manifest.jsonl"
    assert "sim_time" in COMMON_CSV_HEADER


def test_episode_replay_loads_timestamped_streams(tmp_path: Path) -> None:
    episode_dir = tmp_path / "ep0001"
    _write_minimal_episode(episode_dir)

    replay = EpisodeReplay.from_directory(episode_dir)
    replay.validate_schema()

    pose_rows = list(replay.iter_pose())
    assert len(pose_rows) == 2
    assert pose_rows[1]["sim_time"] == "0.020000"

    events = replay.events
    assert events[0].event == "episode_started"
    assert events[1].timestamp == 0.02

    vision = list(replay.iter_vision())
    assert len(vision) == 1
    assert vision[0].left_file.endswith("left_000000.png")
    assert replay.vision_matches_pose_orientation()
    assert replay.taste_nonzero_only_with_contact()


def test_taste_gating_helper_rejects_ungated_proboscis(tmp_path: Path) -> None:
    episode_dir = tmp_path / "bad_taste"
    _write_minimal_episode(episode_dir)
    # Corrupt: non-zero taste without contact.
    lines = (episode_dir / TASTE_FILE).read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace(",1,0.5,", ",0,0.5,")
    (episode_dir / TASTE_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    replay = EpisodeReplay.from_directory(episode_dir)
    assert not replay.taste_nonzero_only_with_contact()
