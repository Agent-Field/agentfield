# Branch Protection & CI Requirements

This document explains the CI requirements for merging pull requests and what to do when checks fail.

## Required Checks Before Merging

All pull requests to `main` must pass the following CI checks:

### Control Plane CI
- **Tests**: All Go tests must pass
- **Build**: Code must compile successfully
- **Linting**: Code must pass golangci-lint checks
- **Cross-compilation**: Must build for Linux, macOS, and Windows

### Go SDK CI
- **Build**: SDK must compile
- **Tests**: All tests must pass
- **Dependencies**: `go mod tidy` must not produce changes

### Python SDK CI
- **Linting**: Code must pass `ruff` checks (both linting and formatting)
- **Tests**: All pytest tests must pass

### Docker Build
- **Image Build**: Docker image must build successfully
- **Smoke Test**: Container must start and respond to `--help`

## What Happens When Checks Fail

If any required check fails, you **cannot merge** your PR until:
1. You fix the issue and push new commits, OR
2. An admin approves an emergency override (rare, see below)

## Common Failure Scenarios & Fixes

### Linting Failures
```bash
# Run linting locally before pushing
cd control-plane
golangci-lint run

# For Python SDK
cd sdk/python
ruff check .
ruff format --check .
```

### Test Failures
```bash
# Run tests locally
cd control-plane
go test ./...

cd sdk/go
go test ./...

cd sdk/python
pytest
```

### Build Failures
```bash
# Test compilation locally
cd control-plane
go build ./...

# Test cross-compilation
GOOS=linux GOARCH=amd64 go build ./cmd/agentfield-server
GOOS=darwin GOARCH=amd64 go build ./cmd/agentfield-server
GOOS=windows GOARCH=amd64 go build ./cmd/agentfield-server
```

### Docker Build Failures
```bash
# Test Docker build locally
docker build -f deployments/docker/Dockerfile.control-plane -t agentfield-control-plane:test .
docker run --rm agentfield-control-plane:test --help
```

## Emergency Merges

In rare cases (critical hotfixes, CI infrastructure issues), an admin may approve merging despite failing checks.

**If you need an emergency merge:**
1. Clearly explain why in the PR description
2. Tag an admin for review
3. Create a follow-up issue to fix the skipped checks
4. Document what was bypassed and why

**Valid reasons for emergency merge:**
- Production is down and the fix is urgent
- CI infrastructure is broken (not your code)
- Circular dependency that requires breaking changes

**Invalid reasons:**
- "I don't want to fix the linting errors"
- "Tests are flaky" (fix the flaky tests instead)
- "I'm in a hurry" (plan better next time)

## Best Practices

### Before Opening a PR
1. Run all tests locally: `./scripts/test-all.sh`
2. Run linting for your changes
3. Ensure your branch is up to date with `main`
4. Test Docker builds if you changed Dockerfiles

### During PR Review
1. Monitor CI status in the PR
2. Fix failures promptly
3. Don't push new commits while CI is running (wait for results)
4. Keep your branch up to date if `main` changes

### When CI Fails
1. Read the error message carefully
2. Reproduce the failure locally
3. Fix the issue
4. Test locally before pushing
5. Push the fix and wait for CI to re-run

## Understanding the "All Required Checks Passed" Job

Each workflow has a summary job called "All Required Checks Passed" that:
- Waits for all critical jobs to complete
- Fails if any critical job fails
- Provides a single status check for branch protection

This makes it easier to see at a glance if your PR is ready to merge.

## Workflow Triggers

Workflows run automatically when:
- You open a PR
- You push new commits to a PR branch
- Someone manually triggers them via GitHub Actions UI

Workflows only run if you changed relevant files:
- `control-plane/**` → Control Plane CI
- `sdk/go/**` → Go SDK CI
- `sdk/python/**` → Python SDK CI
- `deployments/docker/**` or `control-plane/**` → Docker Build

## Troubleshooting

### "Merge" button is disabled
- Check that all required status checks are green
- Ensure you have at least one approval (if required)
- Verify all conversations are resolved
- Make sure your branch is up to date with `main`

### Checks are stuck "In Progress"
- Wait a few minutes (builds can take time)
- Check GitHub Actions tab for details
- If stuck for >30 minutes, cancel and re-run

### Checks passed but now failing after rebase
- You may have introduced a conflict
- Re-run tests locally after rebasing
- Push a fix if needed

### "Required status check is missing"
- Your changes may not have triggered the workflow
- Manually trigger the workflow from GitHub Actions tab
- Or make a trivial change to trigger it (e.g., add a comment)

## Getting Help

If you're stuck with failing CI:
1. Check the workflow logs in GitHub Actions
2. Try to reproduce locally
3. Ask in the team chat with a link to the failing run
4. Tag a maintainer if it seems like a CI infrastructure issue

## Related Documentation

- [GitHub Actions Workflows](../.github/workflows/)
- [Contributing Guide](../docs/CONTRIBUTING.md)
- [Development Guide](../docs/DEVELOPMENT.md)
