#!/bin/bash

# AgentField Release Automation Script
# Builds binaries and creates GitHub releases with auto-incrementing versions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_MANAGER="$SCRIPT_DIR/version-manager.sh"
BUILD_SCRIPT="$PROJECT_ROOT/build-single-binary.sh"
DIST_DIR="$PROJECT_ROOT/dist/releases"

# GitHub configuration
GITHUB_REPO="your-org/agentfield"
GITHUB_OWNER="your-org"
GITHUB_REPO_NAME="agentfield"

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local missing_deps=()
    
    # Check GitHub CLI
    if ! command_exists gh; then
        missing_deps+=("GitHub CLI (gh) - https://cli.github.com/")
    else
        print_success "GitHub CLI found: $(gh --version | head -n1)"
    fi
    
    # Check jq
    if ! command_exists jq; then
        missing_deps+=("jq - JSON processor")
    else
        print_success "jq found: $(jq --version)"
    fi
    
    # Check git
    if ! command_exists git; then
        missing_deps+=("git")
    else
        print_success "git found: $(git --version | head -n1)"
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        missing_deps+=("Must be run from within a git repository")
    fi
    
    # Check version manager
    if [ ! -f "$VERSION_MANAGER" ]; then
        missing_deps+=("Version manager script not found: $VERSION_MANAGER")
    elif [ ! -x "$VERSION_MANAGER" ]; then
        missing_deps+=("Version manager script not executable: $VERSION_MANAGER")
    else
        print_success "Version manager found"
    fi
    
    # Check build script
    if [ ! -f "$BUILD_SCRIPT" ]; then
        missing_deps+=("Build script not found: $BUILD_SCRIPT")
    elif [ ! -x "$BUILD_SCRIPT" ]; then
        missing_deps+=("Build script not executable: $BUILD_SCRIPT")
    else
        print_success "Build script found"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        print_info "Installation instructions:"
        print_info "GitHub CLI: https://cli.github.com/"
        print_info "jq: brew install jq (macOS) or apt-get install jq (Ubuntu)"
        exit 1
    fi
    
    print_success "All prerequisites satisfied!"
}

# Check GitHub authentication
check_github_auth() {
    print_header "Checking GitHub Authentication"
    
    if ! gh auth status >/dev/null 2>&1; then
        print_error "GitHub CLI not authenticated"
        print_info "Please run: gh auth login"
        exit 1
    fi
    
    print_success "GitHub CLI authenticated"
}

# Get version information
get_version_info() {
    print_header "Version Information"
    
    # Get current version info
    CURRENT_VERSION=$("$VERSION_MANAGER" current)
    CURRENT_TAG=$("$VERSION_MANAGER" current-tag)
    NEXT_VERSION=$("$VERSION_MANAGER" next)
    NEXT_TAG=$("$VERSION_MANAGER" next-tag)
    
    print_status "Current version: $CURRENT_TAG"
    print_status "Next version:    $NEXT_TAG"
    
    # Check if tag already exists
    if git tag -l | grep -q "^$NEXT_TAG$"; then
        print_error "Tag $NEXT_TAG already exists"
        print_info "Use '$VERSION_MANAGER set <version>' to set a different version"
        exit 1
    fi
    
    # Check if GitHub release already exists
    if gh release view "$NEXT_TAG" >/dev/null 2>&1; then
        print_error "GitHub release $NEXT_TAG already exists"
        exit 1
    fi
    
    print_success "Version $NEXT_TAG is available for release"
}

# Build binaries
build_binaries() {
    print_header "Building Binaries"
    
    # Set version for build script
    export VERSION="$NEXT_VERSION"
    
    # Navigate to project root and run build script
    cd "$PROJECT_ROOT"
    
    print_status "Running build script with version: $VERSION"
    if ! "$BUILD_SCRIPT"; then
        print_error "Build script failed"
        exit 1
    fi
    
    # Verify build outputs
    if [ ! -d "$DIST_DIR" ]; then
        print_error "Build output directory not found: $DIST_DIR"
        exit 1
    fi
    
    # Check for expected binaries
    local expected_binaries=(
        "agentfield-linux-amd64"
        "agentfield-linux-arm64"
        "agentfield-darwin-amd64"
        "agentfield-darwin-arm64"
    )
    
    local missing_binaries=()
    local available_binaries=()
    for binary in "${expected_binaries[@]}"; do
        if [ ! -f "$DIST_DIR/$binary" ]; then
            missing_binaries+=("$binary")
        else
            available_binaries+=("$binary")
        fi
    done
    
    if [ ${#missing_binaries[@]} -ne 0 ]; then
        print_warning "Missing binaries:"
        for binary in "${missing_binaries[@]}"; do
            echo "  - $binary"
        done
        print_warning "Continuing with available binaries..."
    fi
    
    if [ ${#available_binaries[@]} -eq 0 ]; then
        print_error "No binaries were built successfully"
        exit 1
    fi
    
    print_success "Build completed with ${#available_binaries[@]} of ${#expected_binaries[@]} binaries"
    
    # Show build summary
    print_status "Built files:"
    ls -la "$DIST_DIR" | grep -E "(agentfield-|checksums|build-info|README)"
}

# Generate release notes
generate_release_notes() {
    print_header "Generating Release Notes"
    
    local release_notes_file="$DIST_DIR/release-notes.md"
    
    # Get git log since last tag
    local last_tag=""
    if git tag -l | grep -E "^v[0-9]+\.[0-9]+\.[0-9]+-alpha\.[0-9]+$" | sort -V | tail -n1 | read -r tag; then
        last_tag="$tag"
    fi
    
    cat > "$release_notes_file" << EOF
# AgentField $NEXT_TAG (Pre-release)

This is a pre-release version of AgentField for testing purposes.

## 🚀 What's New

EOF
    
    if [ -n "$last_tag" ]; then
        echo "### Changes since $last_tag" >> "$release_notes_file"
        echo "" >> "$release_notes_file"
        git log --pretty=format:"- %s" "$last_tag..HEAD" >> "$release_notes_file" 2>/dev/null || true
        echo "" >> "$release_notes_file"
    else
        echo "### Initial Alpha Release" >> "$release_notes_file"
        echo "" >> "$release_notes_file"
        echo "- Initial release of AgentField Server" >> "$release_notes_file"
        echo "- Single binary deployment with embedded UI" >> "$release_notes_file"
        echo "- Universal path management (stores data in ~/.agentfield/)" >> "$release_notes_file"
        echo "- Cross-platform support (Linux, macOS)" >> "$release_notes_file"
    fi
    
    cat >> "$release_notes_file" << 'EOF'

## 📦 Installation

### Quick Install (Recommended)
```bash
    curl -sSL https://raw.githubusercontent.com/your-org/agentfield/main/scripts/install.sh | bash
```

### Manual Download
1. Download the appropriate binary for your platform from the assets below
2. Make it executable: `chmod +x agentfield-*`
3. Run: `./agentfield-linux-amd64` (or your platform's binary)
4. Open http://localhost:8080 in your browser

## 🏗️ Available Binaries

- **agentfield-linux-amd64** - Linux (Intel/AMD 64-bit)
- **agentfield-linux-arm64** - Linux (ARM 64-bit)  
- **agentfield-darwin-amd64** - macOS (Intel)
- **agentfield-darwin-arm64** - macOS (Apple Silicon)

## 🔧 Features

- **Single Binary**: Everything bundled in one executable
- **Universal Storage**: All data stored in `~/.agentfield/` directory
- **Embedded UI**: Web interface included in binary
- **Cross-Platform**: Works on Linux and macOS
- **Portable**: Run from anywhere, data stays consistent

## 📁 Data Directory

All AgentField data is stored in `~/.agentfield/`:
```
~/.agentfield/
├── data/
│   ├── agentfield.db              # Main database
│   ├── agentfield.bolt            # Cache
│   ├── keys/                 # Cryptographic keys
│   ├── did_registries/       # DID registries
│   └── vcs/                  # Verifiable credentials
├── agents/                   # Installed agents
├── logs/                     # Application logs
└── config/                   # User configurations
```

## ⚠️ Pre-release Notice

This is an alpha pre-release intended for testing and development. Not recommended for production use.

## 🐛 Issues & Support

Report issues at: https://github.com/your-org/agentfield/issues

EOF
    
    print_success "Release notes generated: $release_notes_file"
}

# Create GitHub release
create_github_release() {
    print_header "Creating GitHub Release"
    
    local release_notes_file="$DIST_DIR/release-notes.md"
    
    # Increment version
    print_status "Incrementing version..."
    "$VERSION_MANAGER" increment >/dev/null
    
    # Create git tag
    print_status "Creating git tag: $NEXT_TAG"
    git tag -a "$NEXT_TAG" -m "Release $NEXT_TAG"
    
    # Push tag to remote
    print_status "Pushing tag to remote..."
    git push origin "$NEXT_TAG"
    
    # Create GitHub release
    print_status "Creating GitHub release..."
    gh release create "$NEXT_TAG" \
        --title "AgentField $NEXT_TAG" \
        --notes-file "$release_notes_file" \
        --prerelease \
        --repo "$GITHUB_REPO"
    
    print_success "GitHub release created: $NEXT_TAG"
}

# Upload release assets
upload_assets() {
    print_header "Uploading Release Assets"
    
    cd "$DIST_DIR"
    
    # List of assets to upload
    local assets=(
        "agentfield-linux-amd64"
        "agentfield-linux-arm64"
        "agentfield-darwin-amd64"
        "agentfield-darwin-arm64"
        "checksums.txt"
        "build-info.txt"
        "README.md"
    )
    
    # Upload each asset
    for asset in "${assets[@]}"; do
        if [ -f "$asset" ]; then
            print_status "Uploading $asset..."
            gh release upload "$NEXT_TAG" "$asset" --repo "$GITHUB_REPO"
            print_success "Uploaded $asset"
        else
            print_warning "Asset not found: $asset"
        fi
    done
    
    print_success "All assets uploaded"
}

# Show release summary
show_summary() {
    print_header "Release Summary"
    
    print_success "🎉 Release $NEXT_TAG created successfully!"
    echo ""
    print_status "Release URL: https://github.com/$GITHUB_REPO/releases/tag/$NEXT_TAG"
    print_status "Version: $NEXT_TAG"
    print_status "Type: Pre-release"
    
    if [ -d "$DIST_DIR" ]; then
        local total_size=$(du -sh "$DIST_DIR" | cut -f1)
        print_status "Total package size: $total_size"
    fi
    
    echo ""
    print_status "Users can install with:"
    echo "  curl -sSL https://raw.githubusercontent.com/$GITHUB_REPO/main/ops/scripts/install.sh | bash"
    echo ""
    print_status "Or download manually from:"
    echo "  https://github.com/$GITHUB_REPO/releases/tag/$NEXT_TAG"
}

# Clean up function
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Release process failed"
        print_info "You may need to clean up manually:"
        print_info "- Delete git tag: git tag -d $NEXT_TAG"
        print_info "- Delete remote tag: git push origin :refs/tags/$NEXT_TAG"
        print_info "- Delete GitHub release: gh release delete $NEXT_TAG"
    fi
}

# Main release function
main() {
    print_header "AgentField Server Release Automation"
    
    echo "This script will:"
    echo "  • Check prerequisites and authentication"
    echo "  • Auto-increment version number"
    echo "  • Build cross-platform binaries"
    echo "  • Create GitHub release with assets"
    echo "  • Tag as pre-release"
    echo ""
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run release steps
    check_prerequisites
    check_github_auth
    get_version_info
    build_binaries
    generate_release_notes
    create_github_release
    upload_assets
    show_summary
    
    # Remove cleanup trap on success
    trap - EXIT
}

# Handle command line arguments
case "${1:-}" in
    "dry-run")
        print_header "Dry Run Mode"
        check_prerequisites
        check_github_auth
        get_version_info
        print_status "Would build version: $NEXT_TAG"
        print_status "Would create release with binaries"
        print_success "Dry run completed - no changes made"
        ;;
    "build-only")
        print_header "Build Only Mode"
        check_prerequisites
        get_version_info
        build_binaries
        print_success "Build completed - no release created"
        ;;
    "help"|"-h"|"--help")
        echo "AgentField Release Automation Script"
        echo ""
        echo "Usage:"
        echo "  $0                Create full release (build + GitHub release)"
        echo "  $0 dry-run        Check prerequisites and show what would be done"
        echo "  $0 build-only     Build binaries only, no GitHub release"
        echo "  $0 help           Show this help"
        echo ""
        echo "Prerequisites:"
        echo "  - GitHub CLI (gh) installed and authenticated"
        echo "  - jq installed"
        echo "  - git repository with remote origin"
        echo "  - Build script executable"
        echo ""
        echo "Environment Variables:"
        echo "  GITHUB_REPO       Override GitHub repository (default: $GITHUB_REPO)"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        print_info "Use '$0 help' for usage information"
        exit 1
        ;;
esac
