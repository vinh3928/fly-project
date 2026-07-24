"""Console logging helpers for live mind."""

from __future__ import annotations

from pathlib import Path

from flybrain.pipeline.live_mind import (
    format_live_mind_detail,
    format_live_step_log,
    log_live_step,
)


def test_format_live_step_log_always_shows_mind_state() -> None:
    mind_line = format_live_step_log(
        0,
        [0.5, -0.1, 0.5, 0.5, 0.0],
        {
            "mind_tick": True,
            "motor_goal": [0.8, 0.0, 0.5, 0.5, 0.0],
            "motor_tracking_error": 0.12,
            "energy": 0.9,
            "refined_thought": "Seek sugar carefully.",
            "devil_critique": "Counter: fall risk.",
            "intent": {
                "future": {
                    "chosen_name": "explore",
                    "layer1_values": [
                        {"name": "explore", "value": 0.9},
                        {"name": "stop", "value": 0.8},
                        {"name": "forward", "value": 0.7},
                    ],
                }
            },
            "affect": {"survival_risk": 0.1, "fear": 0.2},
            "vision_cues": {"wall_hazard": 0.3, "target_bearing": -0.1},
            "curiosity_gated": 0.05,
        },
    )
    assert "[live 0000 MIND*]" in mind_line
    assert "goal=[+0.80" in mind_line
    assert "mind: Seek sugar" in mind_line
    assert "devil:" in mind_line
    assert "plan=explore" in mind_line
    assert "L1[explore=0.90, stop=0.80, forward=0.70]" in mind_line
    assert "wall=0.30" in mind_line
    assert "fear=0.20" in mind_line

    # Motor ticks still print the held mind snapshot.
    motor_line = format_live_step_log(
        3,
        [0.7, 0.0, 0.5, 0.5, 0.0],
        {
            "mind_tick": False,
            "motor_goal": [0.8, 0.0],
            "motor_tracking_error": 0.05,
            "refined_thought": "Seek sugar carefully.",
            "devil_critique": "Counter: fall risk.",
            "intent": {"future": {"chosen_name": "explore"}},
        },
    )
    assert "[live 0003 motor]" in motor_line
    assert "mind: Seek sugar" in motor_line
    assert "devil:" in motor_line


def test_format_live_mind_detail_includes_full_thought() -> None:
    detail = format_live_mind_detail(
        4,
        {
            "refined_thought": "Present: energy=1.00. Future hypothesis: act=stop.",
            "devil_critique": "Counter: freezing wastes time.",
            "intent": {
                "forward": 0.0,
                "turn": -0.2,
                "confidence": 0.6,
                "future": {
                    "chosen_name": "stop",
                    "layer1_values": [{"name": "stop", "value": 0.92}],
                },
            },
            "vision_cues": {"wall_hazard": 0.81},
            "affect": {"survival_risk": 0.21, "fear": 0.44},
        },
    )
    assert "[live 0004 DETAIL]" in detail
    assert "Present: energy=1.00" in detail
    assert '"plan": "stop"' in detail
    assert "0.81" in detail


def test_log_live_step_appends_to_file(tmp_path: Path) -> None:
    path = tmp_path / "live_mind.log"
    log_live_step(
        1,
        [0.2, 0.1, 0.5, 0.5, 0.0],
        {
            "mind_tick": True,
            "motor_goal": [0.2, 0.1],
            "refined_thought": "Walk forward.",
            "intent": {"future": {"chosen_name": "walk"}},
        },
        log_path=path,
        verbose=True,
    )
    text = path.read_text(encoding="utf-8")
    assert "[live 0001 MIND*]" in text
    assert "[live 0001 DETAIL]" in text
    assert "plan=walk" in text
    assert "mind: Walk forward." in text
