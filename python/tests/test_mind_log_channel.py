"""Mind/deep-thought Display 3 SideChannel pack + human-readable panel."""

from __future__ import annotations

from flybrain.debug.mind_log_channel import (
    KIND_DEEP,
    KIND_MIND,
    MindLogDebugChannel,
    format_mind_panel,
    pack_mind_log_message,
    parse_mind_log_message,
)


def test_format_mind_panel_includes_thought_and_devil() -> None:
    panel = format_mind_panel(
        42,
        {
            "refined_thought": "Seek sugar via remembered tunnel.",
            "devil_critique": "Wall ahead; prefer opening.",
            "energy": 0.4,
            "hunger": 0.6,
            "intent": {"future": {"chosen_name": "feed"}},
            "judgement": {"decision": "accept", "blend_weight": 0.7},
            "route_home": {"mode": "route", "hops_left": 3, "reason": "plume_trail"},
        },
    )
    assert "[MIND 42]" in panel
    assert "plan=feed" in panel
    assert "E=0.40" in panel and "H=0.60" in panel
    assert "thought: Seek sugar via remembered tunnel." in panel
    assert "devil: Wall ahead; prefer opening." in panel
    assert "route=route hops=3 (plume_trail)" in panel
    assert "judge=accept(w=0.70)" in panel


def test_format_deep_panel_tags_cue() -> None:
    panel = format_mind_panel(
        99,
        {
            "refined_thought": "Low energy — head home.",
            "interrupt_meta": {"cue": "low_energy"},
            "deep_thought": True,
        },
        deep=True,
    )
    assert "[DEEP 99]" in panel
    assert "cue=low_energy" in panel


def test_pack_parse_roundtrip_ascii() -> None:
    text = "[MIND 1]\nplan=explore\nthought: hello"
    raw = pack_mind_log_message(text, kind=KIND_MIND, step=1)
    parsed = parse_mind_log_message(raw)
    assert parsed["version"] == 1
    assert parsed["kind"] == KIND_MIND
    assert parsed["step"] == 1
    assert parsed["text"] == text


def test_non_ascii_replaced_for_sidechannel() -> None:
    panel = format_mind_panel(1, {"refined_thought": "café → home"})
    assert "thought: caf? ? home" in panel
    assert "→" not in panel
    ch = MindLogDebugChannel()
    out = ch.send_panel(1, {"refined_thought": "café"}, deep=False)
    assert out.isascii()
    assert ch.last_kind == KIND_MIND


def test_send_deep_sets_kind() -> None:
    ch = MindLogDebugChannel()
    ch.send_panel(7, {"refined_thought": "reframe"}, deep=True)
    assert ch.last_kind == KIND_DEEP
    assert ch.last_step == 7
    assert "[DEEP 7]" in ch.last_text
