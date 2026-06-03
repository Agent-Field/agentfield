#!/usr/bin/env bash
# Per-surface coverage runner for AgentField.
#
# Runs the test suite for a single surface with coverage enabled and writes
# all files the aggregator (scripts/coverage-aggregate.py) and the gates
# (scripts/coverage-gate.py, scripts/patch-coverage-gate.sh) expect into
# test-reports/coverage/.
#
# This script is the single source of truth for "what commands run for
# surface X with coverage". Both scripts/coverage-summary.sh (local
# sequential runner) and .github/workflows/coverage.yml (CI parallel matrix)
# delegate to it — so local runs and CI runs cannot drift apart.
#
# Usage:
#   ./scripts/coverage-surface.sh control-plane
#   ./scripts/coverage-surface.sh sdk-go
#   ./scripts/coverage-surface.sh sdk-python
#   ./scripts/coverage-surface.sh sdk-typescript
#   ./scripts/coverage-surface.sh web-ui
#
# Output contract (test-reports/coverage/):
#   control-plane   → control-plane.coverprofile, control-plane.cover.txt,
#                     control-plane-cobertura.xml, control-plane.total.txt
#   sdk-go          → sdk-go.coverprofile, sdk-go.cover.txt,
#                     sdk-go-cobertura.xml, sdk-go.total.txt
#   sdk-python      → sdk-python-coverage.json, sdk-python-coverage.xml
#                     (the XML is cobertura; the JSON is pytest-cov's own format)
#   sdk-typescript  → sdk-typescript-coverage-summary.json,
#                     sdk-typescript-cobertura.xml
#   web-ui          → web-ui-coverage-summary.json, web-ui-cobertura.xml
#
# The exact filenames matter — they are the contract between this producer
# and three consumers (coverage-aggregate.py, coverage-gate.py,
# patch-coverage-gate.sh). Do not rename without updating all four.

set -euo pipefail

SURFACE="${1:-}"
if [[ -z "$SURFACE" ]]; then
  echo "usage: $0 <control-plane|sdk-go|sdk-python|sdk-typescript|web-ui>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/test-reports/coverage"
mkdir -p "$REPORT_DIR"

install_gocover_cobertura() {
  if ! command -v gocover-cobertura >/dev/null 2>&1; then
    echo "==> installing gocover-cobertura"
    GO111MODULE=on go install github.com/boumenot/gocover-cobertura@latest
  fi
  export PATH="$(go env GOPATH)/bin:$PATH"
}

extract_go_total() {
  # Must cd into the module directory because `go tool cover -func` resolves
  # package paths in the coverprofile against the nearest go.mod. Running it
  # from the repo root (no go.mod) fails with:
  #   cover: no required module provides package <pkg>: go.mod file not found
  local module_dir="$1"
  local coverprofile="$2"
  (
    cd "$module_dir"
    go tool cover -func="$coverprofile" | awk '/^total:/ {print $3}' | tr -d '%'
  )
}

case "$SURFACE" in
  control-plane)
    # The control plane uses //go:embed against the web UI dist, so the UI
    # must be built before `go test` will even compile.
    echo "==> control-plane: building web UI (required for //go:embed)"
    (
      cd "$ROOT_DIR/control-plane/web/client"
      CI=1 npm ci
      CI=1 npm run build
    )
    install_gocover_cobertura
    echo "==> control-plane: go test with coverage"
    (
      cd "$ROOT_DIR/control-plane"
      go test -tags sqlite_fts5 \
        -coverprofile="$REPORT_DIR/control-plane.coverprofile" \
        ./...
      go tool cover -func="$REPORT_DIR/control-plane.coverprofile" \
        > "$REPORT_DIR/control-plane.cover.txt"
      gocover-cobertura < "$REPORT_DIR/control-plane.coverprofile" \
        > "$REPORT_DIR/control-plane-cobertura.xml"
    )
    extract_go_total "$ROOT_DIR/control-plane" "$REPORT_DIR/control-plane.coverprofile" \
      > "$REPORT_DIR/control-plane.total.txt"
    ;;

  sdk-go)
    install_gocover_cobertura
    echo "==> sdk-go: go test with coverage"
    (
      cd "$ROOT_DIR/sdk/go"
      go test -coverprofile="$REPORT_DIR/sdk-go.coverprofile" ./...
      go tool cover -func="$REPORT_DIR/sdk-go.coverprofile" \
        > "$REPORT_DIR/sdk-go.cover.txt"
      gocover-cobertura < "$REPORT_DIR/sdk-go.coverprofile" \
        > "$REPORT_DIR/sdk-go-cobertura.xml"
    )
    extract_go_total "$ROOT_DIR/sdk/go" "$REPORT_DIR/sdk-go.coverprofile" \
      > "$REPORT_DIR/sdk-go.total.txt"
    ;;

  sdk-python)
    echo "==> sdk-python: installing dev deps"
    (
      cd "$ROOT_DIR/sdk/python"
      python3 -m pip install --upgrade pip
      python3 -m pip install .[dev]
    )
    # pyproject.toml [tool.pytest.ini_options] addopts already enables
    # --cov=agentfield.* for the tracked modules. We only need to add the
    # JSON + XML report sinks here.
    echo "==> sdk-python: pytest with coverage"
    (
      cd "$ROOT_DIR/sdk/python"
      python3 -m pytest \
        --cov-report=json:"$REPORT_DIR/sdk-python-coverage.json" \
        --cov-report=xml:"$REPORT_DIR/sdk-python-coverage.xml"
    )
    ;;

  sdk-typescript)
    echo "==> sdk-typescript: installing deps"
    (
      cd "$ROOT_DIR/sdk/typescript"
      CI=1 npm ci
    )
    echo "==> sdk-typescript: vitest with coverage"
    (
      cd "$ROOT_DIR/sdk/typescript"
      CI=1 npm run test:coverage:core
    )
    cp "$ROOT_DIR/sdk/typescript/coverage/coverage-summary.json" \
       "$REPORT_DIR/sdk-typescript-coverage-summary.json"
    if [[ -f "$ROOT_DIR/sdk/typescript/coverage/cobertura-coverage.xml" ]]; then
      cp "$ROOT_DIR/sdk/typescript/coverage/cobertura-coverage.xml" \
         "$REPORT_DIR/sdk-typescript-cobertura.xml"
    fi
    ;;

  web-ui)
    echo "==> web-ui: installing deps"
    (
      cd "$ROOT_DIR/control-plane/web/client"
      CI=1 npm ci
    )
    echo "==> web-ui: vitest with coverage"
    (
      cd "$ROOT_DIR/control-plane/web/client"
      CI=1 npm run test:coverage
    )
    cp "$ROOT_DIR/control-plane/web/client/coverage/coverage-summary.json" \
       "$REPORT_DIR/web-ui-coverage-summary.json"
    if [[ -f "$ROOT_DIR/control-plane/web/client/coverage/cobertura-coverage.xml" ]]; then
      cp "$ROOT_DIR/control-plane/web/client/coverage/cobertura-coverage.xml" \
         "$REPORT_DIR/web-ui-cobertura.xml"
    fi
    ;;

  *)
    echo "unknown surface: $SURFACE" >&2
    echo "expected one of: control-plane, sdk-go, sdk-python, sdk-typescript, web-ui" >&2
    exit 2
    ;;
esac

echo "==> $SURFACE coverage artifacts written to $REPORT_DIR"
