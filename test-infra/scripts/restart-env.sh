#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Restarting AgentField test environment..."

# Stop first
"$SCRIPT_DIR/stop-env.sh"

echo ""

# Then start
"$SCRIPT_DIR/start-env.sh"
