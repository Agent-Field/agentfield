<!--
Thanks for contributing to AgentField! Please keep the sections below in
the order they appear. You can delete the comment blocks and any sections
that do not apply.
-->

## Summary

<!-- 1-3 sentences describing what this PR changes and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Docs only
- [ ] Tests only
- [ ] CI / tooling
- [ ] Breaking change

## Test plan

<!-- Concrete list of commands / steps that verify this PR. Prefer
automated tests; note any manual verification you did. Example:
- `cd control-plane && go test ./internal/services/...`
- `cd control-plane/web/client && npx vitest run src/test/pages/RunsPage.test.tsx`
-->

- [ ]
- [ ]

## Test coverage

This repo enforces a **coverage gate** on every PR (see
[`.github/workflows/coverage.yml`](.github/workflows/coverage.yml) and
[`docs/COVERAGE.md`](docs/COVERAGE.md)).

- The gate runs `./scripts/coverage-summary.sh` and compares per-surface
  numbers against `coverage-baseline.json`.
- It **fails** if any surface drops more than **1.0 pp** below its baseline,
  if any surface is below **80%**, or if the weighted aggregate falls below
  **85%**.

Before asking for review, please confirm:

- [ ] I ran tests for the surface(s) I changed locally.
- [ ] New code paths are covered by tests in this PR (no bare additions).
- [ ] If I removed code, I updated `coverage-baseline.json` in
      this PR *only if* the removal caused a legitimate regression and I
      called it out in the summary above.
- [ ] The coverage gate check is green in CI before requesting review.

> **For AI coding agents:** if the `Coverage Summary` GitHub Actions job
> fails, read the sticky PR comment titled "📊 Coverage report" — it lists
> the specific surface that regressed, the delta, and the exact command to
> reproduce locally. Add tests for the uncovered lines in this same PR
> before requesting review again. Do *not* lower baselines to silence the
> gate unless the regression is intentional and explicitly called out in
> the PR summary.

## Checklist

- [ ] I have read [CONTRIBUTING.md](CONTRIBUTING.md) (if present) and
      [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
- [ ] Commits are signed and follow conventional-commits style.
- [ ] I have linked any related issues (Fixes #123 / Closes #456).

## Related issues / PRs

<!-- Optional. Leave blank if none. -->
