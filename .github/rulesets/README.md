# GitHub Rulesets (checked in)

This directory holds the **canonical, version-controlled definition** of this
repo's branch-protection rules. GitHub Rulesets are normally configured through
the web UI, which means (a) there is no audit trail, (b) there is no code
review on protection changes, and (c) recreating the repo drops them on the
floor. Committing them here fixes all three problems — the same pattern used
by `grafana/grafana`, `opentelemetry-collector`, and other mature OSS repos.

## Files

| File | Purpose |
|---|---|
| [`main.json`](main.json) | Protection rules applied to the default branch. Any change here **must** be made in a PR and reviewed like code. |

## How rules get applied

The [`.github/workflows/sync-rulesets.yml`](../workflows/sync-rulesets.yml)
workflow runs on every push to `main` that touches this directory. It calls
the GitHub REST API (`POST /repos/{owner}/{repo}/rulesets`) to create or
update the matching ruleset so the JSON file here is the single source of
truth.

To apply by hand (for example, on the very first rollout):

```bash
./scripts/sync-rulesets.sh
```

The script uses `gh api` and needs a token with `repo` scope (the default
for `gh auth login`).

## What `main.json` currently enforces

1. **No branch deletion** — nobody can `git push --delete main`.
2. **No force-push** — `git push --force-with-lease` to `main` is rejected.
3. **Merge queue** — squash-only, ALLGREEN grouping, 1–5 entries per batch,
   5-minute wait, 60-minute check timeout.
4. **Pull request required** — 1 approving review, **CODEOWNERS review
   required**, stale reviews dismissed on push, review threads must be
   resolved, squash-only merges.
5. **Required status check** — `Coverage Summary / coverage-summary` must
   pass before merge, and the branch must be up to date with `main`
   (strict mode).
6. **Bypass actors** — org admins, deploy keys, and the Maintain repo role
   can bypass. Everyone else is bound by the rules above.

## Adding a new required check

1. Edit `main.json`, add a new entry under
   `rules[].parameters.required_status_checks`:
   ```json
   { "context": "Your Workflow / job-id" }
   ```
2. Open a PR. The sync workflow will apply it on merge.

> The `context` value must match what shows up in the "Checks" tab of a PR
> exactly — `<workflow name> / <job id or name>`.

## Why this layout (vs. `.github/branch-protection.json`, rulesets-as-terraform, etc.)

- A single `rulesets/*.json` directory scales to multiple rulesets (main,
  release branches, tags) without inventing a schema.
- The shape is **the literal request body** of GitHub's Rulesets REST API,
  so there's nothing to translate and the file can be round-tripped with
  `gh api /repos/{o}/{r}/rulesets/{id}`.
- No new build tooling (Terraform, Pulumi) for a repo that doesn't need it.
