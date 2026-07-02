#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_INFRA_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting AgentField test environment..."

cd "$TEST_INFRA_DIR"

# Load environment variables if .env.test exists
if [ -f .env.test ]; then
    echo "📝 Loading environment variables from .env.test"
    export $(grep -v '^#' .env.test | xargs)
fi

# Start services
echo "🐳 Starting Docker Compose services..."
docker compose up -d --build

# Wait for health
echo "⏳ Waiting for services to be healthy..."
"$SCRIPT_DIR/wait-for-health.sh"

echo ""
echo "✅ Test environment ready!"
echo ""
echo "Services:"
echo "  🌐 Control Plane: http://localhost:8080"
echo "  🐘 PostgreSQL: localhost:5433"
echo ""
echo "Run tests with:"
echo "  cd sdk/python && pytest -m functional -v"
echo "  cd sdk/go && go test -tags functional -v"
echo ""
echo "View logs with:"
echo "  ./test-infra/scripts/logs.sh"
echo ""
echo "Stop environment with:"
echo "  ./test-infra/scripts/stop-env.sh"
