#!/usr/bin/env bash
# AgentField CLI Installer
# Usage: curl -fsSL https://agentfield.ai/install.sh | bash
# Version pinning: VERSION=v1.0.0 curl -fsSL https://agentfield.ai/install.sh | bash

set -e

# Configuration
REPO="Agent-Field/agentfield"
INSTALL_DIR="${AGENTFIELD_INSTALL_DIR:-$HOME/.agentfield/bin}"
VERSION="${VERSION:-latest}"
VERBOSE="${VERBOSE:-0}"
SKIP_PATH_CONFIG="${SKIP_PATH_CONFIG:-0}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Temporary directory for downloads
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Print functions
print_banner() {
  echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║${NC}           ${BOLD}AgentField CLI Installer${NC}                        ${CYAN}║${NC}"
  echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

print_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_verbose() {
  if [[ "$VERBOSE" == "1" ]]; then
    echo -e "${CYAN}[VERBOSE]${NC} $1"
  fi
}

# Detect operating system
detect_os() {
  local os
  os=$(uname -s | tr '[:upper:]' '[:lower:]')

  case "$os" in
    darwin)
      echo "darwin"
      ;;
    linux)
      echo "linux"
      ;;
    mingw*|msys*|cygwin*)
      echo "windows"
      ;;
    *)
      print_error "Unsupported operating system: $os"
      print_info "Supported platforms:"
      print_info "  - darwin (macOS)"
      print_info "  - linux"
      print_info "  - windows"
      print_info ""
      print_info "Please open an issue: https://github.com/$REPO/issues"
      exit 1
      ;;
  esac
}

# Detect architecture
detect_arch() {
  local arch
  arch=$(uname -m)

  case "$arch" in
    x86_64|amd64)
      echo "amd64"
      ;;
    aarch64|arm64)
      echo "arm64"
      ;;
    *)
      print_error "Unsupported architecture: $arch"
      print_info "Supported architectures:"
      print_info "  - amd64 (x86_64)"
      print_info "  - arm64 (aarch64)"
      print_info ""
      print_info "Please open an issue: https://github.com/$REPO/issues"
      exit 1
      ;;
  esac
}

# Get latest version from GitHub API
get_latest_version() {
  print_verbose "Fetching latest version from GitHub API..."

  local latest_url="https://api.github.com/repos/$REPO/releases/latest"
  local version

  if command -v curl >/dev/null 2>&1; then
    version=$(curl -fsSL "$latest_url" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
  elif command -v wget >/dev/null 2>&1; then
    version=$(wget -qO- "$latest_url" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
  else
    print_error "Neither curl nor wget found. Please install one of them."
    exit 1
  fi

  if [[ -z "$version" ]]; then
    print_error "Failed to determine latest version from GitHub API"
    print_info "You can manually specify a version: VERSION=v1.0.0 $0"
    exit 1
  fi

  echo "$version"
}

# Download file
download_file() {
  local url="$1"
  local output="$2"

  print_verbose "Downloading: $url"
  print_verbose "To: $output"

  if command -v curl >/dev/null 2>&1; then
    if [[ "$VERBOSE" == "1" ]]; then
      curl -fSL --progress-bar "$url" -o "$output"
    else
      curl -fsSL "$url" -o "$output"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if [[ "$VERBOSE" == "1" ]]; then
      wget -O "$output" "$url"
    else
      wget -q -O "$output" "$url"
    fi
  else
    print_error "Neither curl nor wget found"
    exit 1
  fi

  if [[ ! -f "$output" ]]; then
    print_error "Download failed: $url"
    exit 1
  fi
}

# Verify checksum
verify_checksum() {
  local binary_path="$1"
  local checksums_file="$2"
  local binary_name="$3"

  print_info "Verifying checksum..."
  print_verbose "Binary: $binary_path"
  print_verbose "Checksums file: $checksums_file"
  print_verbose "Binary name: $binary_name"

  # Extract expected checksum from checksums file
  local expected_checksum
  expected_checksum=$(grep "$binary_name" "$checksums_file" | awk '{print $1}')

  if [[ -z "$expected_checksum" ]]; then
    print_error "Could not find checksum for $binary_name in checksums file"
    print_verbose "Checksums file content:"
    if [[ "$VERBOSE" == "1" ]]; then
      cat "$checksums_file"
    fi
    exit 1
  fi

  print_verbose "Expected checksum: $expected_checksum"

  # Calculate actual checksum
  local actual_checksum
  if command -v sha256sum >/dev/null 2>&1; then
    actual_checksum=$(sha256sum "$binary_path" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    actual_checksum=$(shasum -a 256 "$binary_path" | awk '{print $1}')
  else
    print_warning "No checksum utility found (sha256sum or shasum)"
    print_warning "Skipping checksum verification (not recommended)"
    return 0
  fi

  print_verbose "Actual checksum: $actual_checksum"

  if [[ "$actual_checksum" != "$expected_checksum" ]]; then
    print_error "Checksum verification failed!"
    print_error "Expected: $expected_checksum"
    print_error "Got:      $actual_checksum"
    print_error ""
    print_error "This may indicate a corrupted download or security issue."
    print_info "Please try again or report this issue:"
    print_info "  https://github.com/$REPO/issues"
    exit 1
  fi

  print_success "Checksum verified"
}

# Install binary
install_binary() {
  local binary_path="$1"
  local install_dir="$2"

  print_info "Installing to $install_dir"

  # Create install directory
  mkdir -p "$install_dir"

  # Copy binary
  cp "$binary_path" "$install_dir/agentfield"
  chmod +x "$install_dir/agentfield"

  # On macOS, remove quarantine attribute
  if [[ "$(detect_os)" == "darwin" ]]; then
    print_verbose "Removing macOS quarantine attribute..."
    xattr -d com.apple.quarantine "$install_dir/agentfield" 2>/dev/null || true
  fi

  print_success "Binary installed to $install_dir/agentfield"
}

# Configure PATH
configure_path() {
  local install_dir="$1"

  if [[ "$SKIP_PATH_CONFIG" == "1" ]]; then
    print_info "Skipping PATH configuration (SKIP_PATH_CONFIG=1)"
    return 0
  fi

  print_info "Configuring PATH..."

  # Detect shell
  local shell_name
  shell_name=$(basename "$SHELL")

  print_verbose "Detected shell: $shell_name"

  local shell_config=""
  local path_line="export PATH=\"$install_dir:\$PATH\""

  case "$shell_name" in
    bash)
      # Check which file to use (.bashrc or .bash_profile)
      if [[ -f "$HOME/.bashrc" ]]; then
        shell_config="$HOME/.bashrc"
      elif [[ -f "$HOME/.bash_profile" ]]; then
        shell_config="$HOME/.bash_profile"
      else
        shell_config="$HOME/.bashrc"
      fi
      ;;
    zsh)
      shell_config="$HOME/.zshrc"
      ;;
    fish)
      shell_config="$HOME/.config/fish/config.fish"
      path_line="set -gx PATH $install_dir \$PATH"
      mkdir -p "$(dirname "$shell_config")"
      ;;
    *)
      print_warning "Unknown shell: $shell_name"
      print_warning "Please manually add $install_dir to your PATH"
      return 0
      ;;
  esac

  print_verbose "Shell config file: $shell_config"

  # Check if PATH is already configured
  if [[ -f "$shell_config" ]] && grep -q "$install_dir" "$shell_config" 2>/dev/null; then
    print_info "PATH already configured in $shell_config"
    return 0
  fi

  # Add to PATH
  echo "" >> "$shell_config"
  echo "# AgentField CLI" >> "$shell_config"
  echo "$path_line" >> "$shell_config"

  print_success "PATH configured in $shell_config"

  # Provide instructions
  echo ""
  print_info "To use agentfield in this terminal session, run:"
  echo -e "  ${CYAN}source $shell_config${NC}"
  echo ""
  print_info "Or open a new terminal window"
}

# Verify installation
verify_installation() {
  local install_dir="$1"

  print_info "Verifying installation..."

  if [[ -x "$install_dir/agentfield" ]]; then
    print_success "Installation verified"

    # Try to get version
    if "$install_dir/agentfield" --version >/dev/null 2>&1; then
      local version_output
      version_output=$("$install_dir/agentfield" --version 2>&1)
      print_verbose "Version output: $version_output"
    fi

    return 0
  else
    print_error "Installation verification failed"
    print_error "Binary not found or not executable: $install_dir/agentfield"
    exit 1
  fi
}

# Print success message
print_success_message() {
  echo ""
  echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║${NC}  ${BOLD}AgentField CLI installed successfully!${NC}                      ${GREEN}║${NC}"
  echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "${BOLD}Next steps:${NC}"
  echo ""
  echo "  1. Reload your shell configuration:"

  local shell_name
  shell_name=$(basename "$SHELL")
  case "$shell_name" in
    bash)
      if [[ -f "$HOME/.bashrc" ]]; then
        echo -e "     ${CYAN}source ~/.bashrc${NC}"
      else
        echo -e "     ${CYAN}source ~/.bash_profile${NC}"
      fi
      ;;
    zsh)
      echo -e "     ${CYAN}source ~/.zshrc${NC}"
      ;;
    fish)
      echo -e "     ${CYAN}source ~/.config/fish/config.fish${NC}"
      ;;
    *)
      echo -e "     ${CYAN}source your shell config file${NC}"
      ;;
  esac

  echo ""
  echo "  2. Verify installation:"
  echo -e "     ${CYAN}agentfield --version${NC}"
  echo ""
  echo "  3. Initialize your first agent:"
  echo -e "     ${CYAN}agentfield init my-agent${NC}"
  echo ""
  echo -e "${BOLD}Resources:${NC}"
  echo -e "  Documentation: ${BLUE}https://agentfield.ai/docs${NC}"
  echo -e "  GitHub:        ${BLUE}https://github.com/$REPO${NC}"
  echo -e "  Support:       ${BLUE}https://github.com/$REPO/issues${NC}"
  echo ""
}

# Main installation flow
main() {
  print_banner

  # Detect platform
  local os
  local arch
  os=$(detect_os)
  arch=$(detect_arch)

  print_info "Detected platform: $os-$arch"

  # Determine version
  if [[ "$VERSION" == "latest" ]]; then
    VERSION=$(get_latest_version)
  fi

  print_info "Installing version: $VERSION"

  # Construct binary name and URL
  local binary_name="agentfield-$os-$arch"
  if [[ "$os" == "windows" ]]; then
    binary_name="agentfield-$os-$arch.exe"
  fi

  local download_url="https://github.com/$REPO/releases/download/$VERSION/$binary_name"
  local checksums_url="https://github.com/$REPO/releases/download/$VERSION/checksums.txt"

  print_verbose "Binary name: $binary_name"
  print_verbose "Download URL: $download_url"
  print_verbose "Checksums URL: $checksums_url"

  # Download binary
  print_info "Downloading binary..."
  local binary_path="$TMP_DIR/$binary_name"
  download_file "$download_url" "$binary_path"
  print_success "Binary downloaded"

  # Download checksums
  print_info "Downloading checksums..."
  local checksums_path="$TMP_DIR/checksums.txt"
  download_file "$checksums_url" "$checksums_path"
  print_success "Checksums downloaded"

  # Verify checksum
  verify_checksum "$binary_path" "$checksums_path" "$binary_name"

  # Install binary
  install_binary "$binary_path" "$INSTALL_DIR"

  # Configure PATH
  configure_path "$INSTALL_DIR"

  # Verify installation
  verify_installation "$INSTALL_DIR"

  # Print success message
  print_success_message
}

# Run main function
main "$@"
