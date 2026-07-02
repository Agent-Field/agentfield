#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_INFRA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$TEST_INFRA_DIR"

# If arguments provided, pass them to docker compose logs
# Otherwise, follow all logs
if [ $# -eq 0 ]; then
    echo "📋 Tailing logs from all services (Ctrl+C to exit)..."
    docker compose logs -f
else
    docker compose logs "$@"
fi
