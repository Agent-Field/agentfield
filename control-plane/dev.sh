#!/bin/bash
# Start control plane in Docker with hot-reload
#
# Usage:
#   ./dev.sh           # SQLite mode (default)
#   ./dev.sh postgres  # PostgreSQL mode
#   ./dev.sh down      # Stop containers
#   ./dev.sh clean     # Stop and remove volumes

set -e
cd "$(dirname "$0")"

case "${1:-}" in
  postgres|pg)
    echo "Starting control plane with PostgreSQL..."
    docker compose -f docker-compose.dev.yml --profile postgres up
    ;;
  down|stop)
    echo "Stopping containers..."
    docker compose -f docker-compose.dev.yml --profile postgres down
    ;;
  clean)
    echo "Stopping and removing volumes..."
    docker compose -f docker-compose.dev.yml --profile postgres down -v
    ;;
  *)
    echo "Starting control plane with SQLite..."
    docker compose -f docker-compose.dev.yml up
    ;;
esac
