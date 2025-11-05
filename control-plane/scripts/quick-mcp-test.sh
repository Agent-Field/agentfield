#!/bin/bash

# =============================================================================
# Quick MCP Endpoints Test Script
# =============================================================================
# A simplified version for quick testing of MCP endpoints
# Usage: ./haxen/scripts/quick-mcp-test.sh
# =============================================================================

HAXEN_SERVER="${HAXEN_SERVER:-http://localhost:8080}"

echo "🧠 Quick MCP Endpoints Test"
echo "=========================="
echo "Server: $HAXEN_SERVER"
echo ""

# Check if server is running
echo "1. Checking Haxen server..."
if curl -s --connect-timeout 5 "$HAXEN_SERVER/health" > /dev/null; then
    echo "✅ Haxen server is running"
else
    echo "❌ Haxen server is not responding"
    exit 1
fi

# Test overall MCP status
echo ""
echo "2. Testing overall MCP status..."
curl -s "$HAXEN_SERVER/api/ui/v1/mcp/status" | jq . 2>/dev/null || echo "❌ Failed to get MCP status"

# Get first node ID
echo ""
echo "3. Getting available nodes..."
FIRST_NODE=$(curl -s "$HAXEN_SERVER/api/ui/v1/nodes" | jq -r '.[0].id // empty' 2>/dev/null)

if [ -n "$FIRST_NODE" ] && [ "$FIRST_NODE" != "null" ]; then
    echo "✅ Found node: $FIRST_NODE"
    
    # Test node MCP health
    echo ""
    echo "4. Testing node MCP health..."
    curl -s "$HAXEN_SERVER/api/ui/v1/nodes/$FIRST_NODE/mcp/health" | jq . 2>/dev/null || echo "❌ Failed to get node MCP health"
    
    # Test developer mode
    echo ""
    echo "5. Testing developer mode..."
    curl -s "$HAXEN_SERVER/api/ui/v1/nodes/$FIRST_NODE/mcp/health?mode=developer" | jq . 2>/dev/null || echo "❌ Failed to get developer mode health"
else
    echo "⚠️  No nodes found - skipping node-specific tests"
fi

echo ""
echo "🎉 Quick test completed!"
echo ""
echo "For comprehensive testing, run:"
echo "  ./haxen/scripts/test-mcp-endpoints.sh"