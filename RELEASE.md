# Release Process

This document describes how to create a new release of AgentField.

## Prerequisites

- Ensure all changes are merged to the main branch
- All tests are passing
- Documentation is up to date
- Changelog is updated

## Release Methods

You have **two options** to create a release:

### Option 1: Manual Workflow Trigger (Recommended for Testing)

Use the GitHub Actions UI to manually trigger a release:

1. Go to: https://github.com/Agent-Field/agentfield/actions/workflows/release.yml
2. Click "Run workflow"
3. Fill in the form:
   - **Branch:** Select `main` (or your release branch)
   - **Version:** Enter version (e.g., `v0.1.0`) - **REQUIRED if publishing**
   - **Publish to GitHub Releases:** ✅ Check this to create a public release
   - **Publish Python SDK to PyPI:** ✅ Check if you want to publish to PyPI
   - **Push Docker image:** ✅ Check if you want to push Docker image
4. Click "Run workflow"

**What happens:**
- ✅ Builds binaries for all platforms (macOS Intel/ARM, Linux amd64/arm64, Windows)
- ✅ Generates `checksums.txt` with SHA256 hashes
- ✅ Creates GitHub release with all binaries
- ✅ Publishes to PyPI (if checked)
- ✅ Pushes Docker image (if checked)
- ✅ Makes the install script work: `curl -fsSL https://agentfield.ai/install.sh | bash`

### Option 2: Git Tag Push (Recommended for Production)

Create and push a git tag:

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create an annotated tag
git tag -a v0.1.0 -m "Release v0.1.0"

# Push the tag to trigger the release workflow
git push origin v0.1.0
```

**What happens:**
- ✅ GitHub Actions automatically detects the tag
- ✅ Builds and publishes everything (binaries, PyPI, Docker)
- ✅ Creates GitHub release

## Release Artifacts

When a release is published, the following artifacts are created:

### GitHub Release Assets
```
agentfield-darwin-amd64          # macOS Intel binary
agentfield-darwin-arm64          # macOS Apple Silicon binary
agentfield-linux-amd64           # Linux x86_64 binary
agentfield-linux-arm64           # Linux ARM64 binary
agentfield-windows-amd64.exe     # Windows 64-bit binary
checksums.txt                    # SHA256 checksums for all binaries
agentfield-X.Y.Z-py3-none-any.whl   # Python wheel
agentfield-X.Y.Z.tar.gz             # Python source distribution
```

### PyPI Package
- Package: `agentfield`
- URL: https://pypi.org/project/agentfield/

### Docker Image
- Image: `ghcr.io/agent-field/agentfield-control-plane:vX.Y.Z`
- Registry: GitHub Container Registry

## Install Script Compatibility

After a release is published, users can install using:

**macOS/Linux:**
```bash
# Latest version
curl -fsSL https://agentfield.ai/install.sh | bash

# Specific version
VERSION=v0.1.0 curl -fsSL https://agentfield.ai/install.sh | bash
```

**Windows:**
```powershell
# Latest version
iwr -useb https://agentfield.ai/install.ps1 | iex

# Specific version
$env:VERSION="v0.1.0"; iwr -useb https://agentfield.ai/install.ps1 | iex
```

The install scripts:
- Download binaries from GitHub releases
- Verify SHA256 checksums
- Install to `~/.agentfield/bin`
- Configure PATH automatically

## Version Numbering

Follow semantic versioning: `vMAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

Examples:
- `v0.1.0` - Initial release
- `v0.2.0` - New features added
- `v0.2.1` - Bug fixes
- `v1.0.0` - First stable release

## Testing a Release

### Test Manual Workflow (No Publish)

1. Go to: https://github.com/Agent-Field/agentfield/actions/workflows/release.yml
2. Run workflow with:
   - **Publish to GitHub Releases:** ❌ UNCHECKED
   - **Publish Python SDK to PyPI:** ❌ UNCHECKED
   - **Push Docker image:** ❌ UNCHECKED
3. Download artifacts from the workflow run to test locally

This builds everything without publishing.

### Test Published Release

After publishing a release, test the installation:

```bash
# Test install script
VERSION=v0.1.0 bash scripts/install.sh

# Verify installation
agentfield --version

# Test uninstall
bash scripts/uninstall.sh
```

## Rollback

If a release has issues:

### Delete GitHub Release
1. Go to: https://github.com/Agent-Field/agentfield/releases
2. Click on the problematic release
3. Click "Delete release"
4. Delete the git tag:
   ```bash
   git tag -d v0.1.0
   git push origin :refs/tags/v0.1.0
   ```

### Unpublish from PyPI
**Warning:** PyPI does not allow re-uploading the same version. You must:
1. Yank the version (marks as unavailable): https://pypi.org/manage/project/agentfield/releases/
2. Release a new patch version (e.g., `v0.1.1`)

### Remove Docker Image
```bash
# Delete from GitHub Container Registry
gh api -X DELETE /user/packages/container/agentfield-control-plane/versions/VERSION_ID
```

## Checklist

Before releasing, ensure:

- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are bumped
- [ ] No TODO or FIXME comments for critical features
- [ ] README.md examples work
- [ ] All security advisories addressed

## First Release (v0.1.0)

For the first release, use the **manual workflow trigger** method:

1. Merge this PR (the installation system)
2. Go to Actions → Release workflow
3. Fill in:
   - **Version:** `v0.1.0`
   - **Publish to GitHub Releases:** ✅ CHECK
   - **Publish Python SDK to PyPI:** ✅ CHECK (if ready)
   - **Push Docker image:** ✅ CHECK (if ready)
4. Run workflow
5. Wait for workflow to complete
6. Verify release: https://github.com/Agent-Field/agentfield/releases/tag/v0.1.0
7. Test install script: `curl -fsSL https://agentfield.ai/install.sh | bash`

## Hosting Install Scripts

The install scripts need to be accessible at:
- `https://agentfield.ai/install.sh`
- `https://agentfield.ai/install.ps1`
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

## Troubleshooting

### GoReleaser Fails

**Error:** `tag doesn't exist`
**Solution:** Make sure the tag is created before GoReleaser runs. The workflow now creates the tag automatically when using manual trigger with `publish_release=true`.

### Checksums Don't Match

**Error:** Install script reports checksum mismatch
**Solution:**
1. Re-download `checksums.txt` from the release
2. Verify it matches the binary hash:
   ```bash
   sha256sum agentfield-linux-amd64
   ```
3. If mismatched, delete the release and re-run the workflow

### Install Script 404

**Error:** `Failed to download binary`
**Solution:**
1. Verify the release exists: https://github.com/Agent-Field/agentfield/releases
2. Check binary naming matches: `agentfield-{os}-{arch}`
3. Ensure workflow completed successfully

## Support

For release issues, contact:
- GitHub Issues: https://github.com/Agent-Field/agentfield/issues
- Maintainers: @[your-team]
