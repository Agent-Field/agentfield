#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Installing Go tools"
if command -v go >/dev/null 2>&1; then
  (cd "$ROOT_DIR/control-plane" && go mod download)
  (cd "$ROOT_DIR/sdk/go" && go mod download)
  go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
  go install github.com/pressly/goose/v3/cmd/goose@latest
else
  echo "Go not found. Install Go 1.23+ and rerun this script."
fi

echo "==> Installing Python dependencies"
if command -v python3 >/dev/null 2>&1; then
  python3 -m pip install --upgrade pip
  python3 -m pip install --upgrade build pytest ruff
  (cd "$ROOT_DIR/sdk/python" && python3 -m pip install -e .)
  if [ -f "$ROOT_DIR/sdk/python/requirements-dev.txt" ]; then
    python3 -m pip install -r "$ROOT_DIR/sdk/python/requirements-dev.txt"
  fi
else
  echo "Python 3 not found. Install Python 3.10+ and rerun this script."
fi

echo "==> Installing Node.js dependencies for control-plane UI"
if command -v corepack >/dev/null 2>&1; then
  corepack enable
fi
if command -v npm >/dev/null 2>&1; then
  (cd "$ROOT_DIR/control-plane/web/client" && npm install)
else
  echo "npm not found. Install Node.js 20+ to work on the web UI."
fi

echo "Installation complete."
