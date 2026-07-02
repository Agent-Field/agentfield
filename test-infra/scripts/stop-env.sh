#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_INFRA_DIR="$(dirname "$SCRIPT_DIR")"

echo "🛑 Stopping AgentField test environment..."

cd "$TEST_INFRA_DIR"

# Stop and remove containers, networks, and volumes
docker compose down -v

echo "✅ Test environment stopped and cleaned up"
echo "   (All containers, networks, and volumes removed)"
