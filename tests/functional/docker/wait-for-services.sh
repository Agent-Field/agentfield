#!/bin/sh
# wait-for-services.sh - Wait for AgentField control plane to be ready

set -e

CONTROL_PLANE_URL="${AGENTFIELD_SERVER:-http://control-plane:8080}"
HEALTH_ENDPOINT="${CONTROL_PLANE_URL}/api/v1/health"
# Reduced from 60*2s=120s to 30*1s=30s - control plane typically starts in ~10-15s
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
SLEEP_INTERVAL="${SLEEP_INTERVAL:-1}"

echo "Waiting for AgentField control plane at ${CONTROL_PLANE_URL}..."

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    
    # Try to reach the health endpoint with a simple GET request
    if curl --silent --show-error --fail --max-time 2 "${HEALTH_ENDPOINT}" >/dev/null; then
        echo "✓ Control plane is ready!"
        exit 0
    fi
    
    echo "Attempt $attempt/$MAX_ATTEMPTS: Control plane not ready yet..."
    sleep $SLEEP_INTERVAL
done

echo "✗ Control plane failed to become ready after $MAX_ATTEMPTS attempts"
exit 1
