# Test Layout

- `tests/` contains the supported open-source regression suite. These tests run
  in CI and cover the SDK surface that ships publicly.
- `legacy_tests/` preserves the original internal suite. Those tests exercise
  tightly coupled infrastructure (remote services, etc.) and are excluded
  from the default run. Execute them manually via `pytest legacy_tests` when the
  required services are available.

Fixtures live in `tests/conftest.py` and are shared by both suites.
