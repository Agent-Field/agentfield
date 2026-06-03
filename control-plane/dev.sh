#!/bin/bash
# Start control plane with hot-reload using Air
#
# Usage:
#   ./dev.sh              # SQLite mode with hot-reload
#   ./dev.sh postgres     # PostgreSQL mode (starts postgres via docker-compose)
#   ./dev.sh pg-stop      # Stop the dev postgres container
#   ./dev.sh pg-reset     # Stop postgres and wipe its data volume
#   ./dev.sh pg-test      # Run Go postgres tests against dev database
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

pg_up() {
    echo "Starting dev postgres..."
    docker compose -f docker-compose.dev.yml up -d --wait
    echo "Postgres ready at localhost:5432 (agentfield_dev)"
}

pg_down() {
    docker compose -f docker-compose.dev.yml down "$@"
}

case "${1:-}" in
  postgres|pg)
    pg_up
    echo "Starting control plane with PostgreSQL (hot-reload)..."
    export AGENTFIELD_STORAGE_MODE=postgres
    air -c .air.toml
    ;;
  pg-stop)
    echo "Stopping dev postgres..."
    pg_down
    ;;
  pg-reset)
    echo "Stopping dev postgres and wiping data..."
    pg_down -v
    ;;
  pg-test)
    pg_up
    echo "Running postgres storage tests..."
    POSTGRES_TEST_URL="postgres://agentfield:agentfield@localhost:5433/agentfield_dev?sslmode=disable" \
      go test ./internal/storage/ -run TestPostgres -v -count=1
    ;;
  *)
    echo "Starting control plane with SQLite (hot-reload)..."
    export AGENTFIELD_STORAGE_MODE=local
    air -c .air.toml
    ;;
esac
