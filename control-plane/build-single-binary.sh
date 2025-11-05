#!/bin/bash

# AgentField Single Binary Builder
# This script creates a single, portable binary that includes:
# - Go backend with universal path management
# - Embedded UI
# - All dependencies bundled

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_SOURCE_DIR="$SCRIPT_DIR/web/client"
UI_DIST_DIR="$UI_SOURCE_DIR/dist"
OUTPUT_DIR="$SCRIPT_DIR/dist/releases"

# Build configuration
VERSION="${VERSION:-$(date +%Y%m%d-%H%M%S)}"
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
LDFLAGS="-s -w -X main.version=$VERSION -X main.buildTime=$BUILD_TIME -X main.gitCommit=$GIT_COMMIT"

# Function to print colored output
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    local missing_deps=()

    # Check Go
    if ! command_exists go; then
        missing_deps+=("Go (https://golang.org/dl/)")
    else
        go_version=$(go version | grep -o 'go[0-9]\+\.[0-9]\+' | sed 's/go//')
        major_version=$(echo "$go_version" | cut -d. -f1)
        minor_version=$(echo "$go_version" | cut -d. -f2)

        if [ "$major_version" -lt 1 ] || ([ "$major_version" -eq 1 ] && [ "$minor_version" -lt 16 ]); then
            print_error "Go 1.16+ required for embed support. Current: $go_version"
            missing_deps+=("Go 1.16+ (current: $go_version)")
        else
            print_success "Go version: $go_version"
        fi
    fi

    # Check Node.js
    if ! command_exists node; then
        missing_deps+=("Node.js (https://nodejs.org/)")
    else
        node_version=$(node --version)
        print_success "Node.js version: $node_version"
    fi

    # Check npm
    if ! command_exists npm; then
        missing_deps+=("npm (comes with Node.js)")
    else
        npm_version=$(npm --version)
        print_success "npm version: $npm_version"
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi

    print_success "All prerequisites satisfied!"
}

# Function to clean previous builds
clean_build() {
    print_header "Cleaning Previous Builds"

    # Remove previous UI build
    if [ -d "$UI_DIST_DIR" ]; then
        print_status "Removing previous UI build..."
        rm -rf "$UI_DIST_DIR"
    fi

    # Remove previous binaries
    if [ -d "$OUTPUT_DIR" ]; then
        print_status "Removing previous binaries..."
        rm -rf "$OUTPUT_DIR"
    fi

    # Remove test file
    if [ -f "$SCRIPT_DIR/test-paths.go" ]; then
        rm -f "$SCRIPT_DIR/test-paths.go"
    fi

    print_success "Clean completed"
}

# Function to test path management system
test_path_system() {
    print_header "Testing Universal Path Management"

    print_status "Running path management tests..."

    # Create a temporary test file
    cat > "$SCRIPT_DIR/temp-test-paths.go" << 'EOF'
package main

import (
 "fmt"
 "strings"
 "github.com/Agent-Field/agentfield/control-plane/internal/utils"
)

func main() {
 dirs, err := utils.GetAgentFieldDataDirectories()
 if err != nil {
  fmt.Printf("ERROR: %v\n", err)
  return
 }

 fmt.Printf("AgentField Home: %s\n", dirs.AgentFieldHome)

 // Verify that AgentField Home points to ~/.agentfield
 if !strings.HasSuffix(dirs.AgentFieldHome, ".agentfield") {
  fmt.Printf("ERROR: AgentField Home should end with .agentfield, got: %s\n", dirs.AgentFieldHome)
  return
 }

 // Test database path
 dbPath, err := utils.GetDatabasePath()
 if err != nil {
  fmt.Printf("ERROR getting database path: %v\n", err)
  return
 }
 fmt.Printf("Database Path: %s\n", dbPath)

 // Verify database path is in ~/.agentfield/data/
 if !strings.Contains(dbPath, ".agentfield/data/agentfield.db") {
  fmt.Printf("ERROR: Database path should be in ~/.agentfield/data/, got: %s\n", dbPath)
  return
 }

 // Test directory creation
 _, err = utils.EnsureDataDirectories()
 if err != nil {
  fmt.Printf("ERROR: %v\n", err)
  return
 }

 fmt.Println("SUCCESS: Path management system working correctly - database will be stored in ~/.agentfield/")
}
EOF

    # Run the test
    cd "$SCRIPT_DIR"
    if go run temp-test-paths.go | grep -q "SUCCESS"; then
        print_success "Path management system test passed"
    else
        print_error "Path management system test failed"
        exit 1
    fi

    # Clean up test file
    rm -f temp-test-paths.go
}

# Function to build UI
build_ui() {
    print_header "Building User Interface"

    # Check if UI source exists
    if [ ! -d "$UI_SOURCE_DIR" ]; then
        print_error "UI source directory not found: $UI_SOURCE_DIR"
        exit 1
    fi

    # Check if package.json exists
    if [ ! -f "$UI_SOURCE_DIR/package.json" ]; then
        print_error "package.json not found in $UI_SOURCE_DIR"
        exit 1
    fi

    # Navigate to UI directory
    cd "$UI_SOURCE_DIR"

    # Install dependencies
    print_status "Installing UI dependencies..."
    npm install --force --silent

    # Build UI for production
    print_status "Building UI for production..."
    npm run build --silent

    # Verify build output
    if [ ! -d "$UI_DIST_DIR" ]; then
        print_error "UI build failed - dist directory not found"
        exit 1
    fi

    if [ ! -f "$UI_DIST_DIR/index.html" ]; then
        print_error "UI build failed - index.html not found"
        exit 1
    fi

    # Get build size
    if command_exists du; then
        ui_size=$(du -sh "$UI_DIST_DIR" | cut -f1)
        print_success "UI build completed - Size: $ui_size"
    else
        print_success "UI build completed"
    fi

    # Return to script directory
    cd "$SCRIPT_DIR"
}

# Function to build Go binary
build_binary() {
    local os="$1"
    local arch="$2"
    local output_name="$3"

    print_status "Building binary for $os/$arch..."

    # Set environment variables for cross-compilation
    export GOOS="$os"
    export GOARCH="$arch"
    export CGO_ENABLED=1  # Disable CGO for better portability with pure Go SQLite

    # Build the binary with embedded UI and FTS5 support
    go build \
        -ldflags "$LDFLAGS" \
        -tags "embedded sqlite_fts5" \
        -o "$OUTPUT_DIR/$output_name" \
        ./cmd/agentfield-server

    if [ $? -eq 0 ]; then
        # Get file size
        if command_exists du; then
            size=$(du -h "$OUTPUT_DIR/$output_name" | cut -f1)
            print_success "Built $output_name (Size: $size)"
        else
            print_success "Built $output_name"
        fi

        # Make executable on Unix systems
        if [ "$os" != "windows" ]; then
            chmod +x "$OUTPUT_DIR/$output_name"
        fi

        return 0
    else
        print_error "Failed to build $output_name"
        return 1
    fi
}

# Function to build all binaries
build_all_binaries() {
    print_header "Building Cross-Platform Binaries"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Navigate to project root for building
    cd "$SCRIPT_DIR"

    # Build for different platforms
    local build_success=0
    local build_total=0

    # Define platforms to build
    declare -a platforms=(
        # "linux:amd64:agentfield-linux-amd64"
        # "linux:arm64:agentfield-linux-arm64"
        # "darwin:amd64:agentfield-darwin-amd64"
        "darwin:arm64:agentfield-darwin-arm64"
    )

    for platform in "${platforms[@]}"; do
        IFS=':' read -r os arch output <<< "$platform"
        build_total=$((build_total + 1))

        if build_binary "$os" "$arch" "$output"; then
            build_success=$((build_success + 1))
        fi
    done

    print_status "Built $build_success/$build_total binaries successfully"

    if [ $build_success -eq 0 ]; then
        print_error "No binaries were built successfully"
        exit 1
    fi
}

# Function to generate checksums and metadata
generate_metadata() {
    print_header "Generating Metadata"

    cd "$OUTPUT_DIR"

    # Generate SHA256 checksums
    if command_exists sha256sum; then
        sha256sum agentfield-* > checksums.txt 2>/dev/null || true
        print_success "Generated checksums.txt"
    elif command_exists shasum; then
        shasum -a 256 agentfield-* > checksums.txt 2>/dev/null || true
        print_success "Generated checksums.txt"
    else
        print_warning "No checksum utility found, skipping checksum generation"
    fi

    # Generate build info
    cat > build-info.txt << EOF
AgentField Single Binary Build Information
====================================

Build Version: $VERSION
Build Time: $BUILD_TIME
Git Commit: $GIT_COMMIT
Builder: $(whoami)@$(hostname)
Build OS: $(uname -s)
Build Arch: $(uname -m)

Features:
- Universal Path Management (stores data in ~/.agentfield/)
- Embedded Web UI
- Cross-platform compatibility
- Single binary deployment

Usage:
  ./agentfield-<platform>           # Start AgentField server with UI
  ./agentfield-<platform> --help    # Show help
  ./agentfield-<platform> --backend-only  # Start without UI

Data Storage:
All AgentField data is stored in ~/.agentfield/ directory:
- ~/.agentfield/data/agentfield.db      # Main database
- ~/.agentfield/data/agentfield.bolt    # Cache/KV store
- ~/.agentfield/data/keys/         # DID cryptographic keys
- ~/.agentfield/data/did_registries/  # DID registries
- ~/.agentfield/data/vcs/          # Verifiable credentials
- ~/.agentfield/agents/            # Installed agents
- ~/.agentfield/logs/              # Application logs
- ~/.agentfield/config/            # User configurations

Environment Variables:
- AGENTFIELD_HOME: Override default ~/.agentfield directory
- AGENTFIELD_PORT: Override default port (8080)
- AGENTFIELD_CONFIG_FILE: Override config file location

EOF

    print_success "Generated build-info.txt"

    cd "$SCRIPT_DIR"
}

# Function to create distribution package
create_distribution() {
    print_header "Creating Distribution Package"

    # Create a README for the distribution
    cat > "$OUTPUT_DIR/README.md" << 'EOF'
# AgentField Single Binary Distribution

This package contains pre-built AgentField binaries for multiple platforms.

## Quick Start

1. Download the appropriate binary for your platform:
   - `agentfield-linux-amd64` - Linux (Intel/AMD 64-bit)
   - `agentfield-linux-arm64` - Linux (ARM 64-bit)
   - `agentfield-darwin-amd64` - macOS (Intel)
   - `agentfield-darwin-arm64` - macOS (Apple Silicon)
   - `agentfield-windows-amd64.exe` - Windows (64-bit)

2. Make the binary executable (Linux/macOS):
   ```bash
   chmod +x agentfield-*
   ```

3. Run AgentField:
   ```bash
   ./agentfield-linux-amd64
   ```

4. Open your browser to http://localhost:8080

## Features

- **Single Binary**: Everything bundled in one executable
- **Universal Storage**: All data stored in `~/.agentfield/` directory
- **Embedded UI**: Web interface included in binary
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Portable**: Run from anywhere, data stays consistent

## Configuration

AgentField can be configured via:
- Environment variables (AGENTFIELD_HOME, AGENTFIELD_PORT, etc.)
- Configuration file (`~/.agentfield/agentfield.yaml`)
- Command line flags (`--port`, `--backend-only`, etc.)

## Data Directory

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

## Support

For issues and documentation, visit: https://github.com/Agent-Field/agentfield
EOF

    print_success "Created distribution README.md"
}

# Function to display build summary
show_summary() {
    print_header "Build Summary"

    print_status "Build completed successfully!"
    print_status "Version: $VERSION"
    print_status "Build time: $BUILD_TIME"
    print_status "Git commit: $GIT_COMMIT"

    if [ -d "$OUTPUT_DIR" ]; then
        print_status "Output directory: $OUTPUT_DIR"
        print_status "Built files:"
        ls -la "$OUTPUT_DIR" | grep -E "(agentfield-|checksums|build-info|README)"

        # Calculate total size
        if command_exists du; then
            total_size=$(du -sh "$OUTPUT_DIR" | cut -f1)
            print_status "Total package size: $total_size"
        fi
    fi

    echo ""
    print_success "🎉 Single binary build completed successfully!"
    echo ""
    print_status "To test your binary:"
    echo "  cd $OUTPUT_DIR"
    echo "  ./agentfield-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | sed 's/x86_64/amd64/')"
    echo ""
    print_status "The binary includes:"
    echo "  ✅ Go backend with universal path management"
    echo "  ✅ Embedded web UI"
    echo "  ✅ All dependencies bundled"
    echo "  ✅ Cross-platform compatibility"
    echo "  ✅ Portable deployment (stores data in ~/.agentfield/)"
}

# Main build function
main() {
    print_header "AgentField Single Binary Builder"

    echo "Building AgentField single binary with:"
    echo "  • Universal path management"
    echo "  • Embedded web UI"
    echo "  • Cross-platform support"
    echo "  • Portable deployment"
    echo ""

    # Run build steps
    check_prerequisites
    clean_build
    test_path_system
    build_ui
    build_all_binaries
    generate_metadata
    create_distribution
    show_summary
}

# Handle command line arguments
case "${1:-}" in
    "clean")
        clean_build
        ;;
    "ui-only")
        build_ui
        ;;
    "test-paths")
        test_path_system
        ;;
    "help"|"-h"|"--help")
        echo "AgentField Single Binary Builder"
        echo ""
        echo "Usage:"
        echo "  $0                Build complete single binary package"
        echo "  $0 clean          Clean all build artifacts"
        echo "  $0 ui-only        Build only the UI"
        echo "  $0 test-paths     Test path management system"
        echo "  $0 help           Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  VERSION           Build version (default: timestamp)"
        echo "  GIT_COMMIT        Git commit hash (auto-detected)"
        echo ""
        echo "Output:"
        echo "  dist/releases/    Built binaries and metadata"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Use '$0 help' for usage information"
        exit 1
        ;;
esac
