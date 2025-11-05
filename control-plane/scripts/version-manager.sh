#!/bin/bash

# Haxen Version Manager
# Handles version tracking and incrementing for releases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="$SCRIPT_DIR/.version"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if jq is available
check_jq() {
    if ! command -v jq >/dev/null 2>&1; then
        print_error "jq is required but not installed. Please install jq first."
        print_info "On macOS: brew install jq"
        print_info "On Ubuntu/Debian: sudo apt-get install jq"
        print_info "On CentOS/RHEL: sudo yum install jq"
        exit 1
    fi
}

# Initialize version file if it doesn't exist
init_version_file() {
    if [ ! -f "$VERSION_FILE" ]; then
        print_info "Creating initial version file..."
        cat > "$VERSION_FILE" << 'EOF'
{
  "major": 0,
  "minor": 1,
  "patch": 0,
  "alpha_build": 1,
  "last_release": "",
  "git_commit": ""
}
EOF
        print_success "Version file created: $VERSION_FILE"
    fi
}

# Read current version
get_current_version() {
    check_jq
    init_version_file
    
    local major=$(jq -r '.major' "$VERSION_FILE")
    local minor=$(jq -r '.minor' "$VERSION_FILE")
    local patch=$(jq -r '.patch' "$VERSION_FILE")
    local alpha_build=$(jq -r '.alpha_build' "$VERSION_FILE")
    
    echo "${major}.${minor}.${patch}-alpha.${alpha_build}"
}

# Get current version tag (with v prefix)
get_current_version_tag() {
    echo "v$(get_current_version)"
}

# Get next version (increment alpha build)
get_next_version() {
    check_jq
    init_version_file
    
    local major=$(jq -r '.major' "$VERSION_FILE")
    local minor=$(jq -r '.minor' "$VERSION_FILE")
    local patch=$(jq -r '.patch' "$VERSION_FILE")
    local alpha_build=$(jq -r '.alpha_build' "$VERSION_FILE")
    
    # Increment alpha build
    alpha_build=$((alpha_build + 1))
    
    echo "${major}.${minor}.${patch}-alpha.${alpha_build}"
}

# Get next version tag (with v prefix)
get_next_version_tag() {
    echo "v$(get_next_version)"
}

# Increment version and update file
increment_version() {
    check_jq
    init_version_file
    
    local git_commit=""
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    else
        git_commit="unknown"
    fi
    
    local current_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Read current values
    local major=$(jq -r '.major' "$VERSION_FILE")
    local minor=$(jq -r '.minor' "$VERSION_FILE")
    local patch=$(jq -r '.patch' "$VERSION_FILE")
    local alpha_build=$(jq -r '.alpha_build' "$VERSION_FILE")
    
    # Increment alpha build
    alpha_build=$((alpha_build + 1))
    
    # Update version file
    jq --arg major "$major" \
       --arg minor "$minor" \
       --arg patch "$patch" \
       --arg alpha_build "$alpha_build" \
       --arg last_release "$current_time" \
       --arg git_commit "$git_commit" \
       '.major = ($major | tonumber) | 
        .minor = ($minor | tonumber) | 
        .patch = ($patch | tonumber) | 
        .alpha_build = ($alpha_build | tonumber) | 
        .last_release = $last_release | 
        .git_commit = $git_commit' \
       "$VERSION_FILE" > "$VERSION_FILE.tmp" && mv "$VERSION_FILE.tmp" "$VERSION_FILE"
    
    local new_version="${major}.${minor}.${patch}-alpha.${alpha_build}"
    print_success "Version incremented to: v${new_version}"
    echo "$new_version"
}

# Set specific version
set_version() {
    local version="$1"
    if [ -z "$version" ]; then
        print_error "Version string required"
        exit 1
    fi
    
    check_jq
    init_version_file
    
    # Parse version string (e.g., "0.1.0-alpha.5" or "v0.1.0-alpha.5")
    version=$(echo "$version" | sed 's/^v//')  # Remove v prefix if present
    
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+-alpha\.[0-9]+$ ]]; then
        print_error "Invalid version format. Expected: major.minor.patch-alpha.build (e.g., 0.1.0-alpha.1)"
        exit 1
    fi
    
    local major=$(echo "$version" | cut -d. -f1)
    local minor=$(echo "$version" | cut -d. -f2)
    local patch_alpha=$(echo "$version" | cut -d. -f3)
    local patch=$(echo "$patch_alpha" | cut -d- -f1)
    local alpha_build=$(echo "$version" | cut -d. -f4)
    
    local git_commit=""
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    else
        git_commit="unknown"
    fi
    
    local current_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Update version file
    jq --arg major "$major" \
       --arg minor "$minor" \
       --arg patch "$patch" \
       --arg alpha_build "$alpha_build" \
       --arg last_release "$current_time" \
       --arg git_commit "$git_commit" \
       '.major = ($major | tonumber) | 
        .minor = ($minor | tonumber) | 
        .patch = ($patch | tonumber) | 
        .alpha_build = ($alpha_build | tonumber) | 
        .last_release = $last_release | 
        .git_commit = $git_commit' \
       "$VERSION_FILE" > "$VERSION_FILE.tmp" && mv "$VERSION_FILE.tmp" "$VERSION_FILE"
    
    print_success "Version set to: v${version}"
    echo "$version"
}

# Show version info
show_version_info() {
    check_jq
    init_version_file
    
    local current_version=$(get_current_version)
    local next_version=$(get_next_version)
    local last_release=$(jq -r '.last_release' "$VERSION_FILE")
    local git_commit=$(jq -r '.git_commit' "$VERSION_FILE")
    
    echo "Current Version: v${current_version}"
    echo "Next Version:    v${next_version}"
    echo "Last Release:    ${last_release:-"Never"}"
    echo "Git Commit:      ${git_commit:-"Unknown"}"
    echo "Version File:    $VERSION_FILE"
}

# Main function
main() {
    case "${1:-}" in
        "current")
            get_current_version
            ;;
        "current-tag")
            get_current_version_tag
            ;;
        "next")
            get_next_version
            ;;
        "next-tag")
            get_next_version_tag
            ;;
        "increment")
            increment_version
            ;;
        "set")
            set_version "$2"
            ;;
        "info")
            show_version_info
            ;;
        "help"|"-h"|"--help")
            echo "Haxen Version Manager"
            echo ""
            echo "Usage:"
            echo "  $0 current        Show current version"
            echo "  $0 current-tag    Show current version with v prefix"
            echo "  $0 next           Show next version (without incrementing)"
            echo "  $0 next-tag       Show next version tag with v prefix"
            echo "  $0 increment      Increment version and update file"
            echo "  $0 set VERSION    Set specific version (e.g., 0.1.0-alpha.5)"
            echo "  $0 info           Show detailed version information"
            echo "  $0 help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 current        # 0.1.0-alpha.1"
            echo "  $0 current-tag    # v0.1.0-alpha.1"
            echo "  $0 increment      # Increment to 0.1.0-alpha.2"
            echo "  $0 set 0.2.0-alpha.1  # Set to specific version"
            ;;
        "")
            show_version_info
            ;;
        *)
            print_error "Unknown command: $1"
            print_info "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"
