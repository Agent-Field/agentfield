# AgentField Server Release Scripts

This directory contains automation scripts for building and releasing AgentField Server binaries to GitHub.

## Overview

The release system provides:
- **Automated version management** with auto-incrementing alpha builds
- **Cross-platform binary building** (Linux, macOS)
- **GitHub release creation** with pre-release tagging
- **Easy user installation** via shell script

## Scripts

### `version-manager.sh`
Manages version numbers and tracking for releases.

**Usage:**
```bash
./version-manager.sh current        # Show current version
./version-manager.sh next           # Show next version
./version-manager.sh increment      # Increment version and update file
./version-manager.sh info           # Show detailed version info
```

**Examples:**
```bash
$ ./version-manager.sh current
0.1.0-alpha.1

$ ./version-manager.sh increment
[SUCCESS] Version incremented to: v0.1.0-alpha.2
0.1.0-alpha.2

$ ./version-manager.sh info
Current Version: v0.1.0-alpha.1
Next Version:    v0.1.0-alpha.2
Last Release:    Never
Git Commit:      Unknown
Version File:    /path/to/.version
```

### `release.sh`
Main release automation script that builds binaries and creates GitHub releases.

**Usage:**
```bash
./release.sh                # Full release (build + GitHub release)
./release.sh dry-run        # Check what would be done
./release.sh build-only     # Build binaries only
./release.sh help           # Show help
```

**What it does:**
1. Checks prerequisites (GitHub CLI, jq, git)
2. Verifies GitHub authentication
3. Auto-increments version number
4. Builds cross-platform binaries using `../build-single-binary.sh`
5. Generates release notes from git commits
6. Creates GitHub release with pre-release flag
7. Uploads all binary assets and metadata

**Prerequisites:**
- GitHub CLI (`gh`) installed and authenticated
- `jq` for JSON processing
- Git repository with remote origin
- Build script executable

### `.version`
JSON file tracking current version state:
```json
{
  "major": 0,
  "minor": 1,
  "patch": 0,
  "alpha_build": 1,
  "last_release": "2025-01-16T20:08:00Z",
  "git_commit": "abc123"
}
```

## Installation Scripts

### `ops/scripts/install.sh` (Root Level)
User-friendly installation script for end users.

**Features:**
- Auto-detects platform (Linux/macOS, amd64/arm64)
- Downloads latest pre-release from GitHub
- Verifies checksums
- User-specific installation by default (`~/.local/bin`)
- Optional system-wide installation (`/usr/local/bin`)
- Automatic PATH management

**Usage:**
```bash
# Quick install (user-specific, no sudo)
curl -sSL https://raw.githubusercontent.com/your-org/agentfield/main/scripts/install.sh | bash

# System-wide install (requires sudo)
curl -sSL https://raw.githubusercontent.com/your-org/agentfield/main/scripts/install.sh | bash -s -- --system

# Custom directory
curl -sSL https://raw.githubusercontent.com/your-org/agentfield/main/scripts/install.sh | bash -s -- --dir ~/bin
```

## Workflow

### Creating a Release

1. **Prepare for release:**
   ```bash
   cd apps/platform/agentfield/scripts
   ./release.sh dry-run    # Check prerequisites and preview
   ```

2. **Create release:**
   ```bash
   ./release.sh            # Full automated release
   ```

3. **The script will:**
   - Increment version (e.g., v0.1.0-alpha.1 → v0.1.0-alpha.2)
   - Build binaries for all platforms
   - Create GitHub release with pre-release flag
   - Upload all assets (binaries, checksums, metadata)
   - Provide installation instructions

### Version Management

**Auto-incrementing versions:**
- Format: `v0.1.0-alpha.{build_number}`
- Each release increments the alpha build number
- Tracks git commits and release timestamps

**Manual version control:**
```bash
./version-manager.sh set 0.2.0-alpha.1    # Set specific version
./version-manager.sh increment             # Increment current version
```

### Build Output

Each release creates these assets:
- `agentfield-linux-amd64` - Linux Intel/AMD binary
- `agentfield-linux-arm64` - Linux ARM binary  
- `agentfield-darwin-amd64` - macOS Intel binary
- `agentfield-darwin-arm64` - macOS Apple Silicon binary
- `checksums.txt` - SHA256 checksums
- `build-info.txt` - Build metadata
- `README.md` - Installation instructions

## Configuration

### GitHub Repository
The scripts are configured for: `your-org/agentfield`

To change the repository, set the environment variable:
```bash
export GITHUB_REPO="your-org/your-repo"
./release.sh
```

### Build Configuration
The release script uses the existing `../build-single-binary.sh` with these settings:
- Embedded UI included
- Universal path management (stores data in `~/.agentfield/`)
- Cross-platform binaries
- Single binary deployment

## Prerequisites Setup

### Install GitHub CLI
```bash
# macOS
brew install gh

# Ubuntu/Debian
sudo apt-get install gh

# Or download from: https://cli.github.com/
```

### Authenticate GitHub CLI
```bash
gh auth login
```

### Install jq
```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq
```

## Troubleshooting

### Common Issues

**"GitHub CLI not authenticated"**
```bash
gh auth login
gh auth status  # Verify authentication
```

**"jq not found"**
```bash
# Install jq for your platform
brew install jq        # macOS
apt-get install jq     # Ubuntu/Debian
```

**"Build script failed"**
- Ensure `../build-single-binary.sh` is executable
- Check Node.js and Go are installed
- Verify UI build dependencies

**"Tag already exists"**
```bash
# Check existing tags
git tag -l

# Delete local tag if needed
git tag -d v0.1.0-alpha.X

# Delete remote tag if needed
git push origin :refs/tags/v0.1.0-alpha.X
```

### Manual Cleanup

If a release fails partway through:
```bash
# Delete local git tag
git tag -d v0.1.0-alpha.X

# Delete remote git tag
git push origin :refs/tags/v0.1.0-alpha.X

# Delete GitHub release
gh release delete v0.1.0-alpha.X
```

## Examples

### Complete Release Workflow
```bash
# 1. Check current state
cd apps/platform/agentfield/scripts
./version-manager.sh info

# 2. Test the release process
./release.sh dry-run

# 3. Create the release
./release.sh

# 4. Verify release was created
gh release list
```

### User Installation Testing
```bash
# Test the installation script
curl -sSL https://raw.githubusercontent.com/your-org/agentfield/main/scripts/install.sh | bash

# Verify installation
agentfield --help
```

### Build Only (No Release)
```bash
# Just build binaries without creating GitHub release
./release.sh build-only

# Check build output
ls -la ../dist/releases/
```

## File Structure

```
apps/platform/agentfield/scripts/
├── README.md              # This documentation
├── .version              # Version tracking file
├── version-manager.sh    # Version management script
└── release.sh           # Main release automation

../
├── build-single-binary.sh  # Platform build script
├── dist/releases/          # Build output directory
└── ops/scripts/install.sh             # User installation script (root level)
```

## Support

For issues with the release system:
1. Check prerequisites are installed and configured
2. Verify GitHub authentication: `gh auth status`
3. Test with dry-run mode: `./release.sh dry-run`
4. Check build script works: `../build-single-binary.sh`

For user installation issues:
- Verify platform support (Linux, macOS)
- Check network connectivity to GitHub
- Ensure curl is available
- Try manual download from GitHub releases page
