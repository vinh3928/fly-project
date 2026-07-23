"""Phase 0–8 deterministic benchmark suite tests."""

from __future__ import annotations

from flybrain.benchmark import (
    DEFAULT_EXPECTED_PATH,
    BenchmarkConfig,
    all_gates_passed,
    compare_reports,
    load_report,
    run_phase_benchmark,
)


def test_phase_benchmark_gates_and_matches_expected() -> None:
    report = run_phase_benchmark(BenchmarkConfig(seed=42))
    assert all_gates_passed(report)
    assert DEFAULT_EXPECTED_PATH.exists()
    expected = load_report(DEFAULT_EXPECTED_PATH)
    diffs = compare_reports(report, expected)
    assert diffs == [], diffs


def test_phase_benchmark_is_deterministic() -> None:
    cfg = BenchmarkConfig(
        seed=42,
        p4_steps=20,
        p5_steps=30,
        p6_steps=30,
        p7_wm_steps=20,
        p8_steps=25,
    )
    a = run_phase_benchmark(cfg)
    b = run_phase_benchmark(cfg)
    diffs = compare_reports(a, b)
    assert diffs == [], diffs


def test_compare_reports_detects_regression() -> None:
    expected = {
        "seed": 42,
        "phases": {"p0_smoke": {"obs_dim": 197, "action_dim": 5, "ok": True}},
        "gates": {"p0_shapes_ok": True},
    }
    actual = {
        "seed": 42,
        "phases": {"p0_smoke": {"obs_dim": 1, "action_dim": 5, "ok": True}},
        "gates": {"p0_shapes_ok": True},
    }
    diffs = compare_reports(actual, expected)
    assert any("obs_dim" in d for d in diffs)
