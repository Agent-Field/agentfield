# Coverage Guide

AgentField assumes most of its code is written, tested, and reviewed by
AI coding agents. The coverage infrastructure is therefore designed
around two goals:

1. Give an agent a single, unambiguous answer to "is my change allowed to
   land?" — via a CI gate with a machine-readable status file.
2. Give the same agent the exact shell command to fix any regression it
   causes, without needing a human to explain the repo layout.

This file is the ground truth. If anything below contradicts a README
badge or an Actions step summary, this file wins.

## The two entry points

- `./scripts/test-all.sh` — broad local regression pass (all surfaces, no
  coverage reporting). Use this when you just want to know "did the
  tests still pass after my change?"
- `./scripts/coverage-summary.sh` — measures coverage on every tracked
  surface, writes artifacts to `test-reports/coverage/`, and is what the
  `Coverage Summary` GitHub Actions workflow runs on every PR.

## Surfaces

Coverage is tracked per runtime surface, not rolled up into a single
magic number, because the repo is a polyglot monorepo:

| Surface | Root | Toolchain |
| --- | --- | --- |
| `control-plane`  | `control-plane/internal/...`       | Go  (`go test -coverprofile`) |
| `sdk-go`         | `sdk/go/...`                       | Go  (`go test -coverprofile`) |
| `sdk-python`     | `sdk/python/agentfield/...`        | Python (`pytest --cov`) |
| `sdk-typescript` | `sdk/typescript/src/...`           | TypeScript (`vitest --coverage`) |
| `web-ui`         | `control-plane/web/client/src/...` | TypeScript (`vitest --coverage`) |

`./scripts/coverage-summary.sh` writes five per-surface numbers **and** a
weighted aggregate. The aggregate is the number the README badge uses; it
is weighted by source size so a tiny helper package cannot move the
needle.

## The coverage gate

Every PR runs `./scripts/coverage-gate.py`, which compares the current
numbers to a baseline checked into the repo:

- Config:   [`.coverage-gate.toml`](../.coverage-gate.toml)
- Baseline: [`coverage-baseline.json`](../coverage-baseline.json)

The gate enforces five rules, all loaded from `.coverage-gate.toml`:

| Rule | Current value | Enforced by |
| --- | --- | --- |
| `min_surface` — absolute floor for every single surface | **86.0%** | `scripts/coverage-gate.py` |
| `min_aggregate` — absolute floor for the weighted aggregate | **88.5%** | `scripts/coverage-gate.py` |
| `max_surface_drop` — biggest per-surface regression allowed vs baseline | **1.0 pp** | `scripts/coverage-gate.py` |
| `max_aggregate_drop` — biggest aggregate regression allowed vs baseline | **0.5 pp** | `scripts/coverage-gate.py` |
| `min_patch` — coverage required on lines the PR actually touches, per surface | **80.0%** | `scripts/patch-coverage-gate.sh` (diff-cover vs `origin/main`) |

The aggregate rules protect the repo from slow drift; the **patch** rule is
the single most effective regression signal and matches the default used by
codecov, vitest, rust-lang, and grafana — aggregates move slowly, but
untested *new* code shows up immediately.

If any rule fails, the `Coverage Summary` job fails, two sticky PR comments
titled **"📊 Coverage gate"** and **"📐 Patch coverage gate"** are posted
with the tables and exact reproduce commands, and both JSON verdicts are
written to `test-reports/coverage/` (`gate-status.json` and
`patch-gate-status.json`) for programmatic consumers.

## Branch protection

The required-check configuration for `main` is version-controlled in
[`.github/rulesets/main.json`](../.github/rulesets/main.json) and pushed to
GitHub by the [Sync Rulesets](../.github/workflows/sync-rulesets.yml)
workflow. The ruleset requires:

- `Coverage Summary / coverage-summary` must pass before merge
- The branch must be up to date with `main` (strict mode)
- 1 approving review, stale reviews dismissed on push
- No force-pushes, no deletions

Any change to branch protection happens in a PR that edits
`.github/rulesets/main.json`, reviewed like any other code change.

## For AI coding agents

This section is written for you, not for a human.

When the `Coverage Summary` check is red on your PR, here is the loop:

1. **Read the canonical status file, not the PR comment.** The comment is
   a rendering; the ground truth is `test-reports/coverage/gate-status.json`
   from the workflow artifact `coverage-summary`. Download it with:

   ```bash
   gh run download --name coverage-summary --dir /tmp/coverage
   cat /tmp/coverage/gate-status.json
   ```

2. **Iterate over `violations[]`.** Each entry has a `rule`, a `surface`,
   the `value` that failed, the `threshold`, a human-readable `message`,
   and most importantly a `reproduce` shell command.

3. **Run the reproduce command for that surface locally.** It is
   designed to print the lowest-coverage functions/files first so you
   can target your tests.

4. **Add tests in the same PR that cover the specific uncovered code
   paths you just identified.** Tests must exercise real behaviour. A
   test that only imports the module to bump a coverage counter will be
   detected and rejected on review.

5. **Re-run the gate locally before pushing** so you know whether your
   next CI run will pass:

   ```bash
   ./scripts/coverage-summary.sh
   ./scripts/coverage-gate.py \
       --summary  test-reports/coverage/summary.json \
       --baseline coverage-baseline.json \
       --config   .coverage-gate.toml
   echo $?   # 0 = passed, 1 = violations, 2 = usage error
   ```

6. **Do not lower thresholds or baseline numbers to silence the gate.**
   `.coverage-gate.toml` and `coverage-baseline.json` are
   the contract between this repo and every agent working in it. If you
   believe a regression is legitimate (for example, dead code was
   deleted and the file shrank), say so explicitly in the PR description
   and update the relevant file as a separate, reviewable change —
   ideally in its own commit with a message like
   `chore(coverage): lower web-ui baseline to 78.2% after page X removal`.

7. **If you are blocked**, request human review by commenting on the PR.
   Do not merge around the gate.

The gate never runs `git commit` or `git push` for you. It only reads
files and writes a report.

## Badge

The README coverage badge is a shields.io endpoint backed by a GitHub
Gist that `coverage.yml` updates on every push to `main`. The endpoint
JSON lives at `test-reports/coverage/badge.json` during a run and is
copied into the gist only from `main`, so the badge always reflects
`main`'s current aggregate.

If you need to configure the gist for a fork or a self-hosted mirror,
the two required secrets are documented in `.github/workflows/coverage.yml`:

- `GIST_TOKEN`
- `COVERAGE_GIST_ID`

## Why a single blended number at all

AgentField historically reported only per-surface numbers because a
single monorepo percentage is easy to game. That is still the canonical
view — the table in every PR comment lists every surface with a colour
and an arrow.

The aggregate exists on top of the per-surface table for exactly two
reasons:

1. A single number is the only thing a shields.io badge can display.
2. An AI coding agent needs a single global "am I making the repo
   better or worse?" signal it can sort PRs by.

The weighting in `.coverage-gate.toml` intentionally matches the source
size of each surface so the aggregate cannot be inflated by adding
exhaustive tests to a 50-line helper package.

## Non-goals

- Branch coverage, path coverage, or MC/DC: we only enforce statement
  coverage today because that is what every surface's toolchain reports
  out of the box. If a surface starts reporting branch coverage we will
  add a separate rule for it.
- Patch (diff) coverage: tracked as `min_patch` in
  `.coverage-gate.toml` but **not yet enforced** because per-language
  diff-cover tooling is not wired up on every surface. Safe to turn on
  once the Python and Go surfaces have diff-cover in CI.
- Functional / end-to-end tests: those live in
  `.github/workflows/functional-tests.yml` and are never counted toward
  the statement coverage numbers above. They provide cross-service
  confidence that statement coverage cannot.
