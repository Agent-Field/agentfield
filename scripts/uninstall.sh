#!/usr/bin/env bash
# AgentField CLI Uninstaller
# Usage: curl -fsSL https://agentfield.ai/uninstall.sh | bash
# Or: bash scripts/uninstall.sh

set -e

# Configuration
INSTALL_DIR="${AGENTFIELD_INSTALL_DIR:-$HOME/.agentfield}"
VERBOSE="${VERBOSE:-0}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
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

print_banner() {
  echo ""
  echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║${NC}           ${BOLD}AgentField CLI Uninstaller${NC}                      ${CYAN}║${NC}"
  echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

# Remove AgentField directory
remove_agentfield_dir() {
  if [[ -d "$INSTALL_DIR" ]]; then
    print_info "Removing $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"
    print_success "AgentField directory removed"
  else
    print_warning "AgentField directory not found: $INSTALL_DIR"
  fi
}

# Remove PATH configuration
remove_path_config() {
  print_info "Checking shell configuration files..."

  local shell_name
  shell_name=$(basename "$SHELL")

  local shell_configs=()

  case "$shell_name" in
    bash)
      shell_configs+=("$HOME/.bashrc" "$HOME/.bash_profile")
      ;;
    zsh)
      shell_configs+=("$HOME/.zshrc")
      ;;
    fish)
      shell_configs+=("$HOME/.config/fish/config.fish")
      ;;
  esac

  local removed=false

  for config in "${shell_configs[@]}"; do
    if [[ -f "$config" ]]; then
      print_verbose "Checking $config..."

      # Check if file contains AgentField PATH
      if grep -q "\.agentfield" "$config" 2>/dev/null; then
        print_info "Found AgentField PATH in $config"

        # Create backup
        cp "$config" "$config.bak.agentfield"
        print_verbose "Created backup: $config.bak.agentfield"

        # Remove AgentField PATH entries
        # This removes lines containing .agentfield and the comment line before it
        sed -i.tmp '/# AgentField CLI/d; /\.agentfield/d' "$config"
        rm -f "$config.tmp"

        print_success "Removed PATH configuration from $config"
        removed=true
      fi
    fi
  done

  if [[ "$removed" == "false" ]]; then
    print_info "No PATH configuration found in shell config files"
  fi
}

# Main uninstall flow
main() {
  print_banner

  # Check if AgentField is installed
  if [[ ! -d "$INSTALL_DIR" ]]; then
    print_warning "AgentField does not appear to be installed at $INSTALL_DIR"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      print_info "Uninstall cancelled"
      exit 0
    fi
  fi

  # Confirm uninstall
  echo -e "${BOLD}This will remove:${NC}"
  echo "  - AgentField directory: $INSTALL_DIR"
  echo "  - PATH configuration from shell config files"
  echo ""
  read -p "Are you sure you want to uninstall AgentField? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Uninstall cancelled"
    exit 0
  fi

  echo ""

  # Remove AgentField directory
  remove_agentfield_dir

  # Remove PATH configuration
  remove_path_config

  # Print success message
  echo ""
  echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║${NC}  ${BOLD}AgentField CLI uninstalled successfully!${NC}                  ${GREEN}║${NC}"
  echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "${BOLD}Next steps:${NC}"
  echo ""
  echo "  1. Restart your terminal or reload your shell configuration"
  echo ""
  echo "  2. If you have backup files, you can restore them:"
  echo -e "     ${CYAN}ls ~/*.bak.agentfield${NC}"
  echo ""
  echo "  3. To reinstall AgentField:"
  echo -e "     ${CYAN}curl -fsSL https://agentfield.ai/install.sh | bash${NC}"
  echo ""
  echo "Thank you for using AgentField!"
  echo ""
}

# Run main function
main "$@"
