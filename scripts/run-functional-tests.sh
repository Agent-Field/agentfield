#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "══════════════════════════════════════════════════════════════════════"
echo "🧪 Running ALL AgentField Functional Tests"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Track failures
PYTHON_FAILED=0
GO_FAILED=0

# Start test infrastructure once
echo "🚀 Starting test infrastructure..."
"$REPO_ROOT/test-infra/scripts/start-env.sh"
echo ""

# Run Python tests
echo "══════════════════════════════════════════════════════════════════════"
echo "📝 Running Python SDK Functional Tests"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

cd "$REPO_ROOT/sdk/python"
if pytest -m functional -v --tb=short; then
    echo ""
    echo "✅ Python functional tests PASSED"
else
    PYTHON_FAILED=1
    echo ""
    echo "❌ Python functional tests FAILED"
fi
echo ""

# Run Go tests
echo "══════════════════════════════════════════════════════════════════════"
echo "📝 Running Go SDK Functional Tests"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

cd "$REPO_ROOT/sdk/go"
if go test -tags functional -v -timeout 10m; then
    echo ""
    echo "✅ Go functional tests PASSED"
else
    GO_FAILED=1
    echo ""
    echo "❌ Go functional tests FAILED"
fi
echo ""

# Cleanup
echo "══════════════════════════════════════════════════════════════════════"
echo "🛑 Stopping test infrastructure..."
echo "══════════════════════════════════════════════════════════════════════"
"$REPO_ROOT/test-infra/scripts/stop-env.sh"
echo ""

# Report results
echo "══════════════════════════════════════════════════════════════════════"
echo "📊 Test Results Summary"
echo "══════════════════════════════════════════════════════════════════════"

if [ "$PYTHON_FAILED" -eq 0 ]; then
    echo "✅ Python SDK: PASSED"
else
    echo "❌ Python SDK: FAILED"
fi

if [ "$GO_FAILED" -eq 0 ]; then
    echo "✅ Go SDK: PASSED"
else
    echo "❌ Go SDK: FAILED"
fi

echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Exit with error if any tests failed
if [ "$PYTHON_FAILED" -eq 1 ] || [ "$GO_FAILED" -eq 1 ]; then
    echo "❌ Some functional tests failed"
    exit 1
else
    echo "✅ All functional tests passed!"
    exit 0
fi
