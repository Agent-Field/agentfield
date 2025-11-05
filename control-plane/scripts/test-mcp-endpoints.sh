#!/bin/bash

# =============================================================================
# MCP Endpoints Testing Script
# =============================================================================
# This script tests all the new MCP health integration endpoints implemented
# in the AgentField server. It includes comprehensive testing for all 4 endpoints
# with both success and error scenarios.
#
# Prerequisites:
# - AgentField server running on http://localhost:8080 (default)
# - jq installed for JSON formatting (brew install jq on macOS)
# - curl available (should be pre-installed)
# - At least one agent node running and registered
#
# Usage:
#   chmod +x agentfield/scripts/test-mcp-endpoints.sh
#   ./agentfield/scripts/test-mcp-endpoints.sh
#
# =============================================================================

set -e  # Exit on any error

# Configuration
AGENTFIELD_SERVER="${AGENTFIELD_SERVER:-http://localhost:8080}"
VERBOSE="${VERBOSE:-false}"
SLEEP_BETWEEN_TESTS="${SLEEP_BETWEEN_TESTS:-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_separator() {
    echo -e "\n${BLUE}================================================${NC}"
}

# Check if jq is installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed. Please install it:"
        log_error "  macOS: brew install jq"
        log_error "  Ubuntu/Debian: sudo apt-get install jq"
        log_error "  CentOS/RHEL: sudo yum install jq"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed. Please install curl."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Check if AgentField server is running
check_agentfield_server() {
    log_info "Checking if AgentField server is running at $AGENTFIELD_SERVER..."
    
    if curl -s --connect-timeout 5 "$AGENTFIELD_SERVER/health" > /dev/null 2>&1; then
        log_success "AgentField server is running"
    else
        log_error "AgentField server is not responding at $AGENTFIELD_SERVER"
        log_error "Please ensure the AgentField server is running:"
        log_error "  cd agentfield/apps/platform/agentfield && go run cmd/agentfield-server/main.go"
        exit 1
    fi
}

# Get available nodes for testing
get_test_nodes() {
    log_info "Fetching available nodes for testing..."
    
    local response
    response=$(curl -s "$AGENTFIELD_SERVER/api/ui/v1/nodes" 2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        log_error "Failed to fetch nodes from AgentField server"
        return 1
    fi
    
    # Extract node IDs and display them
    local nodes
    nodes=$(echo "$response" | jq -r '.[] | "\(.id) (\(.team_id // "no-team"))"' 2>/dev/null)
    
    if [ -z "$nodes" ]; then
        log_warning "No nodes found. Some tests will be skipped."
        return 1
    fi
    
    log_success "Available nodes:"
    echo "$nodes" | while read -r line; do
        echo "  - $line"
    done
    
    # Get first node ID for testing
    FIRST_NODE_ID=$(echo "$response" | jq -r '.[0].id' 2>/dev/null)
    export FIRST_NODE_ID
    
    return 0
}

# Execute curl command with proper formatting
execute_curl() {
    local method="$1"
    local url="$2"
    local description="$3"
    local data="$4"
    local expected_status="${5:-200}"
    
    log_info "Testing: $description"
    echo "Command: curl -X $method \"$url\""
    
    local curl_args=(-s -w "\nHTTP Status: %{http_code}\nResponse Time: %{time_total}s\n")
    
    if [ "$VERBOSE" = "true" ]; then
        curl_args+=(-v)
    fi
    
    if [ -n "$data" ]; then
        curl_args+=(-H "Content-Type: application/json" -d "$data")
    fi
    
    local response
    response=$(curl -X "$method" "${curl_args[@]}" "$url" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Extract HTTP status
        local http_status
        http_status=$(echo "$response" | grep "HTTP Status:" | cut -d' ' -f3)
        
        # Extract JSON response (everything before HTTP Status line)
        local json_response
        json_response=$(echo "$response" | sed '/HTTP Status:/,$d')
        
        if [ -n "$json_response" ] && echo "$json_response" | jq . >/dev/null 2>&1; then
            log_success "Response received (HTTP $http_status):"
            echo "$json_response" | jq .
        else
            log_success "Response received (HTTP $http_status):"
            echo "$response"
        fi
        
        # Check if status matches expected
        if [ "$http_status" = "$expected_status" ]; then
            log_success "✓ Expected HTTP status $expected_status received"
        else
            log_warning "⚠ Expected HTTP $expected_status, got $http_status"
        fi
    else
        log_error "✗ Request failed with exit code $exit_code"
        echo "$response"
    fi
    
    echo ""
    sleep "$SLEEP_BETWEEN_TESTS"
}

# Test 1: Overall MCP Status
test_overall_mcp_status() {
    log_separator
    echo -e "${BLUE}TEST 1: Overall MCP Status${NC}"
    log_separator
    
    execute_curl "GET" \
        "$AGENTFIELD_SERVER/api/ui/v1/mcp/status" \
        "Get overall MCP status across all nodes" \
        "" \
        "200"
}

# Test 2: Node-specific MCP Health (User Mode)
test_node_mcp_health_user() {
    log_separator
    echo -e "${BLUE}TEST 2: Node MCP Health (User Mode)${NC}"
    log_separator
    
    if [ -z "$FIRST_NODE_ID" ]; then
        log_warning "Skipping node-specific tests - no nodes available"
        return
    fi
    
    execute_curl "GET" \
        "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/health" \
        "Get MCP health for node $FIRST_NODE_ID (user mode)" \
        "" \
        "200"
}

# Test 3: Node-specific MCP Health (Developer Mode)
test_node_mcp_health_developer() {
    log_separator
    echo -e "${BLUE}TEST 3: Node MCP Health (Developer Mode)${NC}"
    log_separator
    
    if [ -z "$FIRST_NODE_ID" ]; then
        log_warning "Skipping node-specific tests - no nodes available"
        return
    fi
    
    execute_curl "GET" \
        "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/health?mode=developer" \
        "Get MCP health for node $FIRST_NODE_ID (developer mode)" \
        "" \
        "200"
}

# Test 4: MCP Server Restart (Developer Mode)
test_mcp_server_restart() {
    log_separator
    echo -e "${BLUE}TEST 4: MCP Server Restart (Developer Mode)${NC}"
    log_separator
    
    if [ -z "$FIRST_NODE_ID" ]; then
        log_warning "Skipping MCP server restart test - no nodes available"
        return
    fi
    
    # First, try to get available MCP servers for this node
    log_info "Getting available MCP servers for node $FIRST_NODE_ID..."
    local health_response
    health_response=$(curl -s "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/health?mode=developer" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$health_response" ]; then
        local server_alias
        server_alias=$(echo "$health_response" | jq -r '.servers[0].alias // empty' 2>/dev/null)
        
        if [ -n "$server_alias" ] && [ "$server_alias" != "null" ]; then
            log_info "Found MCP server alias: $server_alias"
            execute_curl "POST" \
                "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/$server_alias/restart" \
                "Restart MCP server '$server_alias' on node $FIRST_NODE_ID" \
                "" \
                "200"
        else
            log_warning "No MCP servers found for node $FIRST_NODE_ID, testing with dummy alias"
            execute_curl "POST" \
                "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/test-server/restart" \
                "Restart MCP server 'test-server' on node $FIRST_NODE_ID (should fail)" \
                "" \
                "404"
        fi
    else
        log_warning "Could not fetch MCP servers, testing with dummy alias"
        execute_curl "POST" \
            "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/test-server/restart" \
            "Restart MCP server 'test-server' on node $FIRST_NODE_ID (should fail)" \
            "" \
            "404"
    fi
}

# Test 5: MCP Tools Listing (Developer Mode)
test_mcp_tools_listing() {
    log_separator
    echo -e "${BLUE}TEST 5: MCP Tools Listing (Developer Mode)${NC}"
    log_separator
    
    if [ -z "$FIRST_NODE_ID" ]; then
        log_warning "Skipping MCP tools listing test - no nodes available"
        return
    fi
    
    # First, try to get available MCP servers for this node
    log_info "Getting available MCP servers for node $FIRST_NODE_ID..."
    local health_response
    health_response=$(curl -s "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/health?mode=developer" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$health_response" ]; then
        local server_alias
        server_alias=$(echo "$health_response" | jq -r '.servers[0].alias // empty' 2>/dev/null)
        
        if [ -n "$server_alias" ] && [ "$server_alias" != "null" ]; then
            log_info "Found MCP server alias: $server_alias"
            execute_curl "GET" \
                "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/$server_alias/tools" \
                "Get tools for MCP server '$server_alias' on node $FIRST_NODE_ID" \
                "" \
                "200"
        else
            log_warning "No MCP servers found for node $FIRST_NODE_ID, testing with dummy alias"
            execute_curl "GET" \
                "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/test-server/tools" \
                "Get tools for MCP server 'test-server' on node $FIRST_NODE_ID (should fail)" \
                "" \
                "404"
        fi
    else
        log_warning "Could not fetch MCP servers, testing with dummy alias"
        execute_curl "GET" \
            "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/test-server/tools" \
            "Get tools for MCP server 'test-server' on node $FIRST_NODE_ID (should fail)" \
            "" \
            "404"
    fi
}

# Test 6: Error Cases
test_error_cases() {
    log_separator
    echo -e "${BLUE}TEST 6: Error Cases${NC}"
    log_separator
    
    # Test with invalid node ID
    execute_curl "GET" \
        "$AGENTFIELD_SERVER/api/ui/v1/nodes/invalid-node-id/mcp/health" \
        "Get MCP health for invalid node ID (should fail)" \
        "" \
        "404"
    
    # Test with non-existent server alias
    if [ -n "$FIRST_NODE_ID" ]; then
        execute_curl "GET" \
            "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/non-existent-server/tools" \
            "Get tools for non-existent MCP server (should fail)" \
            "" \
            "404"
        
        execute_curl "POST" \
            "$AGENTFIELD_SERVER/api/ui/v1/nodes/$FIRST_NODE_ID/mcp/servers/non-existent-server/restart" \
            "Restart non-existent MCP server (should fail)" \
            "" \
            "404"
    fi
}

# Test 7: SSE Events (if possible with curl)
test_sse_events() {
    log_separator
    echo -e "${BLUE}TEST 7: SSE Events (Limited curl test)${NC}"
    log_separator
    
    log_info "Testing SSE endpoint connection (will timeout after 5 seconds)..."
    log_info "Note: For full SSE testing, use a proper SSE client or browser"
    
    # Test SSE connection briefly
    timeout 5 curl -s -H "Accept: text/event-stream" \
        "$AGENTFIELD_SERVER/api/ui/v1/events" 2>/dev/null || true
    
    log_info "SSE connection test completed (use browser or SSE client for full testing)"
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "=============================================="
    echo "    MCP Endpoints Testing Script"
    echo "=============================================="
    echo -e "${NC}"
    
    log_info "Starting MCP endpoints testing..."
    log_info "AgentField Server: $AGENTFIELD_SERVER"
    log_info "Verbose Mode: $VERBOSE"
    echo ""
    
    # Pre-flight checks
    check_dependencies
    check_agentfield_server
    get_test_nodes
    
    # Run all tests
    test_overall_mcp_status
    test_node_mcp_health_user
    test_node_mcp_health_developer
    test_mcp_server_restart
    test_mcp_tools_listing
    test_error_cases
    test_sse_events
    
    # Summary
    log_separator
    echo -e "${GREEN}TESTING COMPLETED${NC}"
    log_separator
    
    log_success "All MCP endpoint tests have been executed"
    log_info "Review the output above for any failures or warnings"
    log_info ""
    log_info "Expected successful responses should include:"
    log_info "  - Overall MCP status: JSON with node statuses"
    log_info "  - Node MCP health: JSON with server health details"
    log_info "  - MCP tools: JSON array of available tools"
    log_info "  - Server restart: Success confirmation message"
    log_info ""
    log_info "Common troubleshooting:"
    log_info "  - Ensure AgentField server is running: go run cmd/agentfield-server/main.go"
    log_info "  - Ensure at least one agent is running and registered"
    log_info "  - Check agent MCP server configurations"
    log_info "  - Verify network connectivity to $AGENTFIELD_SERVER"
    
    echo ""
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -s|--server)
            AGENTFIELD_SERVER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose     Enable verbose curl output"
            echo "  -s, --server URL  Set AgentField server URL (default: http://localhost:8080)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  AGENTFIELD_SERVER           AgentField server URL"
            echo "  VERBOSE               Enable verbose mode (true/false)"
            echo "  SLEEP_BETWEEN_TESTS   Seconds to sleep between tests (default: 2)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            log_error "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"