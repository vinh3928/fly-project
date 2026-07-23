# Agent instructions

## Testing

- All new and changed code must have unit test coverage.
- Prefer tests next to the area under test (Unity EditMode tests under `unity/Assets/Tests/`, Python tests under `python/tests/`).
- When code changes are made, all tests must be run before finishing the task.
  - Python: `cd python && pytest`
  - Phase 0–8 fake benchmark (optional but preferred when touching mind/env):  
    `python -m scripts.run_phase_benchmark --check`
  - Unity: EditMode tests in the Test Runner (or equivalent batch run when available)
- Do not consider work done until those test runs pass.

## Design

- Follow DRY (Don't Repeat Yourself): extract shared logic into reusable helpers, components, or modules instead of copying code.
- Prefer a single source of truth for constants, configuration, and repeated algorithms.
