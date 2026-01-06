# Release Process

This document describes how to create releases for AgentField using the **two-tier release system**.

## Overview

AgentField uses a two-tier release model that separates **staging** (prerelease) from **production** releases:

| Environment | Version Format | Python Registry | npm Tag | Docker Tag | GitHub Release | Trigger |
|-------------|----------------|-----------------|---------|------------|----------------|---------|
| **Staging** | `0.1.19-rc.1` | PyPI (prerelease) | `@next` | `staging-X.Y.Z-rc.N` | Pre-release | **Automatic** (push to main) |
| **Production** | `0.1.19` | PyPI (stable) | `@latest` | `vX.Y.Z` + `latest` | Release | **Manual** (workflow dispatch) |

### Key Points

- **Staging releases are automatic**: Every push to `main` that changes code triggers a staging release
- **Production releases are manual**: Only triggered via GitHub Actions workflow dispatch
- **No version gaps**: Production versions are clean sequential numbers (0.1.18, 0.1.19, 0.1.20...)

### Version Flow Example

```
Current production: 0.1.18

Development cycle for 0.1.19:
  -> PR merged to main    -> Auto: 0.1.19-rc.1  (PyPI prerelease, npm @next)
  -> Bug fix merged       -> Auto: 0.1.19-rc.2  (automatic increment)
  -> Another fix merged   -> Auto: 0.1.19-rc.3  (automatic increment)
  -> Manual trigger       -> Prod: 0.1.19       (PyPI, npm @latest)

Next cycle:
  -> PR merged            -> Auto: 0.1.20-rc.1
  -> More changes         -> Auto: 0.1.20-rc.2...
  -> Manual trigger       -> Prod: 0.1.20

Result: Clean sequential production versions with no gaps!
```

---

## Prerequisites

- Ensure all changes are merged to the main branch
- All tests are passing
- Documentation is up to date

### Required Secrets

The following secrets must be configured in GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `PYPI_API_TOKEN` | PyPI token (for all Python releases) |
| `NPM_TOKEN` | npm registry token |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions |

---

## Release Types

### Staging Release (Pre-release) - AUTOMATIC

Staging releases are **automatically triggered** when code is pushed to `main`. This includes:
- Direct pushes to main
- Merged pull requests

**Trigger paths** (changes to these files trigger a staging release):
- `control-plane/**` - Control plane changes
- `sdk/**` - SDK changes (Python, TypeScript, Go)
- `VERSION` - Version file changes
- `.github/workflows/release.yml` - Release workflow changes

**What happens automatically:**
1. Version bumps to next `-rc.N` (e.g., `0.1.19-rc.1` -> `0.1.19-rc.2`)
2. Publishes to PyPI as prerelease (excluded from `pip install` by default per [PEP 440](https://peps.python.org/pep-0440/))
3. Publishes to npm with `@next` tag
4. Pushes Docker image with `staging-` prefix
5. Creates GitHub pre-release

**Artifacts published to:**
- Python: PyPI as prerelease (`pip install --pre agentfield`)
- TypeScript: npm with `@next` tag
- Docker: `agentfield/control-plane:staging-X.Y.Z-rc.N`
- Binaries: GitHub Pre-release

**Manual staging release (optional):**

If you need to manually trigger a staging release (e.g., with different options):
1. Go to Actions -> [Release workflow](https://github.com/Agent-Field/agentfield/actions/workflows/release.yml)
2. Click "Run workflow"
3. Select `release_environment: staging`
4. Optionally change `release_component` for minor/major bumps
5. Click "Run workflow"

**Testing staging releases:**

```bash
# Binary (using --staging flag)
curl -fsSL https://agentfield.ai/install.sh | bash -s -- --staging

# Or directly from GitHub
curl -fsSL https://raw.githubusercontent.com/Agent-Field/agentfield/main/scripts/install.sh | bash -s -- --staging

# Python (prerelease - requires --pre flag)
pip install --pre agentfield

# TypeScript
npm install @agentfield/sdk@next

# Docker
docker pull agentfield/control-plane:staging-0.1.28-rc.4
```

### Production Release - MANUAL

Production releases are **manually triggered** via GitHub Actions workflow dispatch.

**When to use:**
- After staging releases have been tested
- Ready for public release
- You've verified the staging version works correctly

**Steps:**
1. Ensure staging release(s) have been tested
2. Go to Actions -> [Release workflow](https://github.com/Agent-Field/agentfield/actions/workflows/release.yml)
3. Click "Run workflow"
4. Fill in the form:
   - **release_environment:** `production` (default for manual triggers)
   - **release_component:** `patch` (usually - will finalize from prerelease)
   - **publish_pypi:** Check to publish to PyPI
   - **publish_npm:** Check to publish with `@latest` tag
   - **publish_docker:** Check to push Docker image
5. Click "Run workflow"

**What happens:**
1. Version finalizes from `0.1.19-rc.N` to `0.1.19`
2. Publishes to PyPI (production)
3. Publishes to npm with `@latest` tag
4. Pushes Docker image with version tag + `latest`
5. Creates GitHub release (not pre-release)

**Artifacts published to:**
- Python: PyPI (https://pypi.org)
- TypeScript: npm with `@latest` tag
- Docker: `agentfield/control-plane:vX.Y.Z` + `:latest`
- Binaries: GitHub Release (public)

**Installing production releases:**

```bash
# Binary (recommended)
curl -fsSL https://agentfield.ai/install.sh | bash

# Python
pip install agentfield

# TypeScript
npm install @agentfield/sdk

# Docker
docker pull agentfield/control-plane:latest
```

---

## Release Artifacts

### GitHub Release Assets
```
agentfield-darwin-amd64          # macOS Intel binary
agentfield-darwin-arm64          # macOS Apple Silicon binary
agentfield-linux-amd64           # Linux x86_64 binary
agentfield-linux-arm64           # Linux ARM64 binary
checksums.txt                    # SHA256 checksums for all binaries
agentfield-X.Y.Z-py3-none-any.whl   # Python wheel
agentfield-X.Y.Z.tar.gz             # Python source distribution
```

### Registry Packages

| Registry | Staging | Production |
|----------|---------|------------|
| PyPI | `pip install --pre agentfield` | `pip install agentfield` |
| npm | `@agentfield/sdk@next` | `@agentfield/sdk@latest` |
| Docker | `agentfield/control-plane:staging-*` | `agentfield/control-plane:v*` |

---

## Install Script Compatibility

### Production Install

```bash
# Latest stable version
curl -fsSL https://agentfield.ai/install.sh | bash

# Specific version
VERSION=v0.1.28 curl -fsSL https://agentfield.ai/install.sh | bash
```

### Staging Install

```bash
# Latest prerelease version (using --staging flag)
curl -fsSL https://agentfield.ai/install.sh | bash -s -- --staging

# Or using environment variable
STAGING=1 curl -fsSL https://agentfield.ai/install.sh | bash

# Specific prerelease version
VERSION=v0.1.28-rc.4 curl -fsSL https://agentfield.ai/install.sh | bash -s -- --staging
```

**Key differences when using `--staging`:**
- Installs to `~/.agentfield-staging/bin` (separate from production)
- Creates `af-staging` symlink instead of `af`
- Fetches the latest prerelease from GitHub API

---

## Version Numbering

Follow semantic versioning: `vMAJOR.MINOR.PATCH[-PRERELEASE]`

- **MAJOR:** Breaking changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)
- **PRERELEASE:** Staging identifier (`-rc.1`, `-beta.1`, `-alpha.1`)

Examples:
- `v0.1.18` - Current production
- `v0.1.19-rc.1` - First staging release for 0.1.19
- `v0.1.19-rc.2` - Second staging release (bug fix)
- `v0.1.19` - Production release (finalizes from rc)
- `v0.2.0-rc.1` - Staging for minor version bump

---

## Testing Releases

### Verify Automatic Staging Release

After merging a PR or pushing to `main`:

1. Check the [Actions tab](https://github.com/Agent-Field/agentfield/actions/workflows/release.yml) for the automatic run
2. Verify:
   - [ ] Workflow triggered automatically
   - [ ] Version bumped to `X.Y.Z-rc.N` (e.g., `0.1.19-rc.1`)
   - [ ] Python prerelease package appears on PyPI
   - [ ] `npm install @agentfield/sdk@next` installs new version
   - [ ] GitHub release marked as "Pre-release"
   - [ ] Docker image tagged `staging-X.Y.Z-rc.N`
3. Test staging install:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/Agent-Field/agentfield/main/scripts/install.sh | bash -s -- --staging
   ~/.agentfield-staging/bin/agentfield --version
   ```

### Multiple Staging Releases

Each push to `main` automatically increments the rc number:
- First push: `0.1.19-rc.1`
- Second push: `0.1.19-rc.2`
- Third push: `0.1.19-rc.3`
- etc.

All previous staging artifacts remain available.

### Test Production Release

1. After staging validation, trigger with `release_environment: production`
2. Verify:
   - [ ] Version finalizes to `X.Y.Z` (e.g., `0.1.19`, no `-rc.N` suffix)
   - [ ] Python package appears on PyPI
   - [ ] `npm install @agentfield/sdk` gets new version
   - [ ] GitHub release NOT marked as "Pre-release"
   - [ ] Docker image tagged `vX.Y.Z` and `latest`
3. Test `install.sh`:
   ```bash
   curl -fsSL https://agentfield.ai/install.sh | bash
   ~/.agentfield/bin/agentfield --version
   ```

---

## Rollback Procedures

### Staging Rollback

| Component | Procedure |
|-----------|-----------|
| PyPI prerelease | Cannot re-upload same version; must yank + bump rc number |
| npm @next | `npm unpublish @agentfield/sdk@X.Y.Z-rc.N` (within 72 hours) or publish new rc |
| Docker staging | Delete image tag from Docker Hub via web UI or CLI |
| GitHub | Delete the prerelease from Releases page |

### Production Rollback

| Component | Procedure |
|-----------|-----------|
| PyPI | Cannot re-upload same version; must yank + bump patch version |
| npm @latest | `npm deprecate` or publish new patch; unpublish within 72 hours |
| Docker latest | Push previous version as `:latest` |
| GitHub | Mark release as prerelease or delete |

**To delete a release and tag:**
```bash
# Delete the git tag
git tag -d v0.1.19
git push origin :refs/tags/v0.1.19

# Then delete from GitHub Releases UI
```

---

## Checklist

### Before Staging Release
- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Changes are documented
- [ ] No critical security issues

### Before Production Release
- [ ] Staging release has been tested
- [ ] SDK installation verified (`pip install --pre`, npm @next)
- [ ] Binary installation verified (`install.sh --staging`)
- [ ] Docker image tested
- [ ] CHANGELOG.md is updated
- [ ] README.md examples work

---

## Hosting Install Scripts

The install scripts need to be accessible at:
- `https://agentfield.ai/install.sh` (handles both production and staging via `--staging` flag)
- `https://agentfield.ai/uninstall.sh`

**Options:**

1. **GitHub Raw URLs (Temporary):**
   ```
   https://raw.githubusercontent.com/Agent-Field/agentfield/main/scripts/install.sh
   ```

2. **Website Rewrites (Recommended):**
   Configure your web server to serve these files from the repo or redirect to GitHub raw URLs.

3. **CDN (Production):**
   Host on a CDN for reliability and speed.

---

## Troubleshooting

### No Prerelease Found

**Error:** `install.sh --staging` reports "No prerelease version found"
**Solution:** There are no staging releases yet. Create one first using the workflow.

### Checksums Don't Match

**Error:** Install script reports checksum mismatch
**Solution:**
1. Re-download `checksums.txt` from the release
2. Verify it matches the binary hash:
   ```bash
   sha256sum agentfield-linux-amd64
   ```
3. If mismatched, delete the release and re-run the workflow

### npm Publish Fails

**Error:** `npm ERR! 403` when publishing
**Solution:**
1. Verify `NPM_TOKEN` is valid
2. Check you have publish permissions for the package
3. Ensure the version doesn't already exist on npm

---

## Support

For release issues, contact:
- GitHub Issues: https://github.com/Agent-Field/agentfield/issues
