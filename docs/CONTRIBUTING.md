# Contributing Guide

Thank you for your interest in contributing to AgentField! This guide outlines how to propose changes, report issues, and participate in the community.

## Ground Rules

- Be kind and respectful. See `CODE_OF_CONDUCT.md`.
- Create issues before large changes to align on direction.
- Keep pull requests focused and small. Large refactors should be split.
- Follow existing coding style and conventions.
- Ensure tests pass locally before opening a pull request.

## Development Environment

1. Fork the repository and clone your fork.
2. Install dependencies:
   ```bash
   ./scripts/install.sh
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```

## Commit Guidelines

- Use [Conventional Commits](https://www.conventionalcommits.org) when possible (`feat:`, `fix:`, `chore:`, etc.).
- Keep commit messages concise yet descriptive.
- Reference related issues with `Fixes #<id>` or `Refs #<id>` when applicable.

## Pull Requests

Before submitting:

1. Run `./scripts/test-all.sh`.
2. Run `make fmt tidy` to keep code formatted and dependencies tidy.
3. Update documentation and changelog entries where relevant.
4. Ensure CI workflows pass.

When opening a PR:

- Provide context in the description.
- Highlight user-facing changes and migration steps.
- Include screenshots for UI changes.
- Link to the issue being resolved.

### CLA Requirements

This repository currently uses the hosted `cla-assistant.io` integration for the
`license/cla` status check.

- Human contributors must sign the CLA through the link shown on the pull request.
- Bot-authored pull requests do not have a human signer, so maintainers must
  allowlist approved bot accounts in the hosted CLA Assistant repository
  settings.
- Recommended bot allowlist entries:
  - `dependabot[bot]`
  - `github-actions[bot]`
  - `renovate[bot]`

Important: this CLA gate is not managed from `.github/workflows`. Updating
repository workflows alone will not clear a pending `license/cla` status from
the hosted integration.

## Issue Reporting

- Search existing issues to avoid duplicates.
- Use the provided issue templates (`bug`, `feature`, `question`).
- Include reproduction steps, logs, or stack traces when possible.

## Documentation

- Keep docs precise and actionable.
- Update `docs/DEVELOPMENT.md` for tooling or workflow changes.
- Update `docs/ARCHITECTURE.md` for structural changes.

## Release Workflow

Releases are automated via GitHub Actions. The workflow now uses
[git-cliff](https://github.com/orhun/git-cliff) to render changelog entries from the
commit history as part of the version bump step. Maintainers typically only need to:

1. Ensure commits follow the conventional prefixes (`feat:`, `fix:`, etc.) so they are
   categorized correctly.
2. Trigger the `Release` workflow with the desired SemVer component/channel.
3. Let the workflow bump versions, rebuild SDKs, update `CHANGELOG.md`, and publish
   artifacts automatically.

To preview the generated changelog locally install `git-cliff`
(`cargo install git-cliff` or grab a release binary) and run:

```bash
python scripts/update_changelog.py --version 0.1.8 --dry-run
```

If you install via Cargo make sure `~/.cargo/bin` is on your `PATH` so `git cliff`
invokes the plugin binary correctly.

## AI-Assisted Contributions

We welcome contributions made with AI assistance (Claude, GitHub Copilot, Cursor, etc.).

### Guidelines

1. **Quality matters, not method** - We evaluate PRs on code quality, test coverage, and adherence to standards. How you wrote the code doesn't affect acceptance.

2. **Commit attribution** - If using an AI assistant, you may include it as a co-author:
   ```
   Co-Authored-By: Claude <noreply@anthropic.com>
   ```
   This is optional but helps with transparency.

3. **You own the submission** - You're responsible for reviewing AI-generated code before submitting. Ensure it:
   - Follows project conventions
   - Includes appropriate tests
   - Doesn't introduce security issues
   - Makes sense in context

### Testing Requirements for AI-Assisted Code

AI-generated code requires the same test coverage as human-written code. When using AI:

**Before Submitting:**
1. Run `./scripts/test-all.sh` and verify all tests pass
2. For Go code: `go test ./...` in the relevant package
3. For Python code: `pytest` with coverage
4. For Web UI: `npm test` (once vitest is set up)

**Writing Tests with AI - Prompt Strategies:**

When asking AI to write tests, be explicit about what to test:

```
Write unit tests for [function/component] that cover:
1. Happy path - normal expected inputs
2. Edge cases - empty inputs, null values, boundary conditions
3. Error cases - invalid inputs, network failures, timeouts
4. Integration points - verify mocks are called correctly
```

Example prompt for better test coverage:
```
Add tests for the retry handler. Include:
- Test successful retry on transient error
- Test max retries exceeded
- Test context cancellation during retry
- Test exponential backoff timing
- Mock the underlying service to control failures
```

**Common AI Testing Pitfalls to Avoid:**
- Tests that only check "no error" without verifying behavior
- Missing edge case coverage (empty arrays, null, negative numbers)
- Mocking everything instead of testing integration
- Tests that pass but don't actually assert anything meaningful
- Copy-paste tests that don't adapt to specific scenarios

**Test Quality Checklist:**
- [ ] Tests actually fail when the code is broken
- [ ] Each test has clear assertions, not just "runs without error"
- [ ] Edge cases are covered (null, empty, boundary values)
- [ ] Error paths are tested, not just happy paths
- [ ] Tests are independent and can run in any order
- [ ] Test names clearly describe what is being tested

### Tips for Using AI with AgentField

- Point your AI to `CLAUDE.md` for project context
- Share relevant file paths from the issue description
- Ask the AI to run tests before considering the task complete
- Review diffs carefully - AI sometimes over-engineers or misses edge cases
- If the issue has a "Files" section, give those paths to your AI
- Ask AI to explain its test strategy before writing tests

## Questions?

Open a `question` issue or start a discussion in the repository. We're excited to build with you!
