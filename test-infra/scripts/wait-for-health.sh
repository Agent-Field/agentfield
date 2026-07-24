#!/usr/bin/env bash
set -euo pipefail

CONTROL_PLANE_URL="${AGENTFIELD_SERVER:-http://localhost:8080}"
HEALTH_URL="$CONTROL_PLANE_URL/api/v1/health"
MAX_WAIT=60
WAIT_TIME=0

echo "⏳ Waiting for control plane at $HEALTH_URL..."

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "✅ Control plane is healthy"
        exit 0
    fi
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
    if [ $((WAIT_TIME % 10)) -eq 0 ]; then
        echo "   ... waiting ($WAIT_TIME/$MAX_WAIT seconds)"
    fi
done

echo "❌ Control plane failed to become healthy after $MAX_WAIT seconds"
echo "   Check logs with: docker compose -f test-infra/docker-compose.yml logs"
exit 1
