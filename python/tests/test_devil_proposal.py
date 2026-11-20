"""Structured devil proposals."""

from __future__ import annotations

from flybrain.thinking.devils_advocate import (
    PROPOSAL_SCHEMA,
    BackgroundLlmClient,
    HeuristicDevilClient,
    parse_proposal,
    propose_critique,
)
from flybrain.thinking.judgement import DevilProposal


def test_parse_proposal_from_noisy_llm_text() -> None:
    text = 'Sure: {"risk":"wall","forward_scale":0.4,"turn_delta":-0.3,"confidence":0.7,"reason":"wall ahead"} ok'
    prop = parse_proposal(text)
    assert prop.risk == "wall"
    assert abs(prop.forward_scale - 0.4) < 1e-6
    assert prop.turn_delta < 0
    assert prop.source == "llm"
    assert prop.ran


def test_parse_proposal_unparsable() -> None:
    prop = parse_proposal("no json here")
    assert prop.source == "unparsable"
    assert prop.confidence == 0.0


def test_heuristic_propose_critique_wall() -> None:
    prop = propose_critique(
        HeuristicDevilClient(),
        draft="Considering step=1 L1=explore",
        affect={"survival_risk": 0.2},
        signals={"wall_hazard": 0.8, "energy": 0.9},
    )
    assert isinstance(prop, DevilProposal)
    assert prop.ran and prop.source == "heuristic"
    assert prop.risk == "wall"


def test_heuristic_propose_critique_prefers_heightened_odor() -> None:
    """Deep-thought ×10 odor should beat ambient wall in the heuristic."""
    prop = propose_critique(
        HeuristicDevilClient(),
        draft="Considering step=1 L1=explore odor=3.7",
        affect={"survival_risk": 1.0},
        signals={
            "wall_hazard": 0.81,
            "energy": 0.9,
            "odor": 3.7,
            "odor_left": 3.7,
            "odor_right": 3.71,
        },
    )
    assert prop.risk == "odor"
    assert "odor" in prop.reason
    assert prop.turn_delta > 0  # right plume slightly stronger


def test_heuristic_no_false_odor_or_wall_from_draft_keywords() -> None:
    """Draft always prints 'odor=' and 'L1=explore'; must not force odor/wall."""
    prop = propose_critique(
        HeuristicDevilClient(),
        draft="Considering step=1 sugar=0.0 bitter=0.0 odor=0.0 L1=explore tendency=curiosity",
        affect={"survival_risk": 0.1},
        signals={
            "wall_hazard": 0.2,
            "energy": 0.9,
            "odor": 0.0,
            "odor_left": 0.0,
            "odor_right": 0.0,
        },
    )
    assert prop.risk == "none"


def test_heuristic_deep_thought_owns_smell_interrupt() -> None:
    """Smell interrupt: reason must name the cue even with wall noise present."""
    prop = propose_critique(
        HeuristicDevilClient(),
        draft="Considering step=1 L1=explore odor=3.7 | INTERRUPT cue=smell_novelty",
        affect={"survival_risk": 1.0},
        signals={
            "wall_hazard": 0.81,
            "odor": 3.7,
            "odor_left": 3.7,
            "odor_right": 3.71,
            "stimulus_interrupt": True,
            "deep_thought": True,
            "interrupt_remaining": 40,
            "interrupt_cue": "smell_novelty",
            "interrupt_reason": "novel_vs_smell",
        },
    )
    assert prop.risk == "odor"
    assert "interrupt" in prop.reason
    assert "smell_novelty" in prop.reason


def test_heuristic_deep_thought_visual_cue_without_plume() -> None:
    """Visual interrupt with zero odor should not fabricate an odor counter."""
    prop = propose_critique(
        HeuristicDevilClient(),
        draft="Considering step=1 L1=explore odor=0.0 | INTERRUPT cue=visual_novelty",
        affect={"survival_risk": 0.2},
        signals={
            "wall_hazard": 0.2,
            "odor": 0.0,
            "odor_left": 0.0,
            "odor_right": 0.0,
            "stimulus_interrupt": True,
            "deep_thought": True,
            "interrupt_remaining": 40,
            "interrupt_cue": "visual_novelty",
            "interrupt_reason": "empty_joint_stm",
        },
    )
    assert prop.risk != "odor"
    assert "visual_novelty" in prop.reason

    assert prop.forward_scale < 1.0


def test_parse_proposal_repairs_unquoted_keys() -> None:
    """llama3.2:3b-style slip: only the first key gets quoted."""
    prop = parse_proposal(
        '{"risk":"toxin", forward_scale: 0.1, turn_delta: -0.5, confidence: 0.8, reason: "bitter"}'
    )
    assert prop.source == "llm"
    assert prop.risk == "toxin"
    assert abs(prop.forward_scale - 0.1) < 1e-6
    assert abs(prop.confidence - 0.8) < 1e-6


def test_parse_proposal_repairs_truncated_object() -> None:
    prop = parse_proposal('{"risk":"wall","forward_scale":0.2,"turn_delta":0.4,"confidence":0.6,')
    assert prop.source == "llm"
    assert prop.risk == "wall"
    assert abs(prop.confidence - 0.6) < 1e-6


def test_parse_proposal_repairs_truncation_mid_value() -> None:
    """Token limit cuts the reason string in half."""
    prop = parse_proposal(
        '{"risk":"toxin","forward_scale":0.3,"turn_delta":-0.2,"reason":"bitter patch ah'
    )
    assert prop.source == "llm"
    assert prop.risk == "toxin"
    assert abs(prop.turn_delta + 0.2) < 1e-6


def test_parse_proposal_coerces_bad_numbers() -> None:
    prop = parse_proposal(
        '{"risk":"wall","forward_scale":"slow","turn_delta":null,"confidence":0.5,"reason":"x"}'
    )
    assert prop.forward_scale == 1.0
    assert prop.turn_delta == 0.0
    assert prop.risk == "wall"


class _SchemaSpy:
    """Records whether the structured-output schema reached the transport."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.response_format: dict | None = None
        self.calls = 0

    def complete(self, system: str, user: str, response_format: dict | None = None) -> str:
        self.calls += 1
        self.response_format = response_format
        return self.reply


class _LegacyClient:
    """A client predating response_format; must still work."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        return self.reply


def test_propose_critique_requests_json_schema() -> None:
    spy = _SchemaSpy(
        '{"risk":"wall","forward_scale":0.3,"turn_delta":0.5,"confidence":0.9,"reason":"r"}'
    )
    prop = propose_critique(spy, draft="d", affect={}, signals={"wall_hazard": 0.9})
    assert spy.response_format == PROPOSAL_SCHEMA
    schema = PROPOSAL_SCHEMA["json_schema"]["schema"]
    assert schema["properties"]["risk"]["enum"] == [
        "wall",
        "toxin",
        "odor",
        "fall",
        "energy",
        "none",
    ]
    assert schema["additionalProperties"] is False
    assert prop.source == "llm" and prop.risk == "wall"


def test_propose_critique_tolerates_client_without_response_format() -> None:
    prop = propose_critique(
        _LegacyClient(
            '{"risk":"fall","forward_scale":0.2,"turn_delta":0,"confidence":0.4,"reason":"r"}'
        ),
        draft="d",
        affect={},
        signals={},
    )
    assert prop.source == "llm" and prop.risk == "fall"


def test_background_client_forwards_response_format() -> None:
    spy = _SchemaSpy(
        '{"risk":"none","forward_scale":1,"turn_delta":0,"confidence":0.2,"reason":"r"}'
    )
    client = BackgroundLlmClient(spy, executor=None)
    propose_critique(client, draft="d", affect={}, signals={})
    assert spy.response_format == PROPOSAL_SCHEMA


def test_heuristic_hungry_prefers_energy_over_odor() -> None:
    """Status/hunger reframe: devil must not chase odor while hungry."""
    prop = propose_critique(
        HeuristicDevilClient(),
        draft="Considering step=1 L1=explore odor=3.7 | INTERRUPT cue=hunger",
        affect={"survival_risk": 0.5},
        signals={
            "wall_hazard": 0.3,
            "energy": 0.25,
            "odor": 3.7,
            "odor_left": 3.7,
            "odor_right": 3.71,
            "stimulus_interrupt": True,
            "deep_thought": True,
            "interrupt_remaining": 40,
            "interrupt_cue": "hunger",
            "interrupt_reason": "status_reframe:hunger",
        },
    )
    assert prop.risk == "energy"
    assert "hunger" in prop.reason or "food" in prop.reason or "hungry" in prop.reason
