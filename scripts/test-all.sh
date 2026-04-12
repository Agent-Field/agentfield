#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v go >/dev/null 2>&1; then
  echo "go is required. Run ./scripts/install.sh first."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Run ./scripts/install.sh first."
  exit 1
fi

if ! python3 -m pytest --version >/dev/null 2>&1; then
  echo "python3 -m pytest is unavailable. Run ./scripts/install.sh first."
  exit 1
fi

echo "==> Running control plane tests"
(cd "$ROOT_DIR/control-plane" && go test ./...)

echo "==> Running Go SDK tests"
(cd "$ROOT_DIR/sdk/go" && go test ./...)

echo "==> Running Python SDK tests"
(cd "$ROOT_DIR/sdk/python" && python3 -m pytest)

if command -v npm >/dev/null 2>&1; then
  echo "==> Running TypeScript SDK tests"
  (cd "$ROOT_DIR/sdk/typescript" && CI=1 npm run test:core)

  if [[ "${AGENTFIELD_RUN_UI_LINT:-0}" == "1" ]]; then
    echo "==> Linting control plane web UI"
    (cd "$ROOT_DIR/control-plane/web/client" && CI=1 npm run lint)
  else
    echo "==> Skipping control plane web UI lint (set AGENTFIELD_RUN_UI_LINT=1 to enable)"
  fi

  echo "==> Running Web UI tests (vitest)"
  (cd "$ROOT_DIR/control-plane/web/client" && CI=1 npm run test)
else
  echo "npm not found; skipping TypeScript SDK and web UI checks."
fi

echo "All tests passed."
