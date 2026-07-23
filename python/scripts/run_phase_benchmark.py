"""
Deterministic Phase 0–8 fake-env benchmark (seeded, repeatable).

Examples:
  python -m scripts.run_phase_benchmark
  python -m scripts.run_phase_benchmark --write
  python -m scripts.run_phase_benchmark --check
  python -m scripts.run_phase_benchmark --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flybrain.benchmark import (
    DEFAULT_EXPECTED_PATH,
    BenchmarkConfig,
    all_gates_passed,
    compare_reports,
    load_report,
    run_phase_benchmark,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write",
        type=str,
        nargs="?",
        const=str(DEFAULT_EXPECTED_PATH),
        default=None,
    )
    parser.add_argument(
        "--check",
        type=str,
        nargs="?",
        const=str(DEFAULT_EXPECTED_PATH),
        default=None,
        help="Compare against expected JSON (default: benchmarks/expected_seed42.json)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run twice and require identical phase metrics (determinism)",
    )
    parser.add_argument("--p4-steps", type=int, default=40)
    parser.add_argument("--p5-steps", type=int, default=80)
    parser.add_argument("--p6-steps", type=int, default=80)
    parser.add_argument("--p7-wm-steps", type=int, default=40)
    parser.add_argument("--p8-steps", type=int, default=60)
    args = parser.parse_args(argv)

    cfg = BenchmarkConfig(
        seed=args.seed,
        p4_steps=args.p4_steps,
        p5_steps=args.p5_steps,
        p6_steps=args.p6_steps,
        p7_wm_steps=args.p7_wm_steps,
        p8_steps=args.p8_steps,
    )
    report = run_phase_benchmark(cfg)
    print(json.dumps(report, indent=2, sort_keys=True))

    if not all_gates_passed(report):
        print("phase_benchmark_gates_failed", report.get("gates"), flush=True)
        return 2

    if args.self_check:
        again = run_phase_benchmark(cfg)
        diffs = compare_reports(again, report)
        if diffs:
            print("phase_benchmark_not_deterministic", flush=True)
            for line in diffs:
                print(f"  {line}", flush=True)
            return 3
        print("phase_benchmark_deterministic_ok", flush=True)

    if args.write is not None:
        write_report(report, args.write)
        print(f"wrote {args.write}", flush=True)

    if args.check is not None:
        expected_path = Path(args.check)
        if not expected_path.exists():
            print(f"missing expected report: {expected_path}", flush=True)
            return 4
        expected = load_report(expected_path)
        diffs = compare_reports(report, expected)
        if diffs:
            print("phase_benchmark_regression", flush=True)
            for line in diffs:
                print(f"  {line}", flush=True)
            return 5
        print("phase_benchmark_check_ok", flush=True)

    print("phase_benchmark_ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
