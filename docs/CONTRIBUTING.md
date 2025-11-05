# Contributing Guide

Thank you for your interest in contributing to Haxen! This guide outlines how to propose changes, report issues, and participate in the community.

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

## Issue Reporting

- Search existing issues to avoid duplicates.
- Use the provided issue templates (`bug`, `feature`, `question`).
- Include reproduction steps, logs, or stack traces when possible.

## Documentation

- Keep docs precise and actionable.
- Update `docs/DEVELOPMENT.md` for tooling or workflow changes.
- Update `docs/ARCHITECTURE.md` for structural changes.

## Release Workflow

Releases are automated via GitHub Actions:

1. Update `CHANGELOG.md` with notable changes.
2. Bump versions (Go modules, Python package) as needed.
3. Tag the release following `vMAJOR.MINOR.PATCH`.
4. The `release.yml` workflow builds artifacts and publishes SDKs.

## Questions?

Open a `question` issue or start a discussion in the repository. We’re excited to build with you!

