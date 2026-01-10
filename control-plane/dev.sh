#!/bin/bash
# Start control plane with hot-reload using Air
#
# Usage:
#   ./dev.sh           # Start with hot-reload (SQLite mode)
#   ./dev.sh postgres  # Start with PostgreSQL (set AGENTFIELD_DATABASE_URL first)
#
# Prerequisites:
#   go install github.com/air-verse/air@v1.61.7

set -e
cd "$(dirname "$0")"

# Check if air is installed
if ! command -v air &> /dev/null; then
    echo "Air not found. Installing..."
    go install github.com/air-verse/air@v1.61.7
fi

case "${1:-}" in
  postgres|pg)
    echo "Starting control plane with PostgreSQL (hot-reload)..."
    export AGENTFIELD_STORAGE_MODE=postgresql
    air -c .air.toml
    ;;
  *)
    echo "Starting control plane with SQLite (hot-reload)..."
    export AGENTFIELD_STORAGE_MODE=local
    air -c .air.toml
    ;;
esac
