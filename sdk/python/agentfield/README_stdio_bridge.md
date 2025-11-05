# MCP Stdio-to-HTTP Bridge

The `mcp_stdio_bridge.py` module provides a bridge that converts stdio-based MCP (Model Context Protocol) servers to HTTP endpoints. This allows the existing HTTP-based MCP client infrastructure to work with stdio-based MCP servers.

## Overview

Some MCP servers (like `@modelcontextprotocol/server-sequential-thinking`) use stdio transport instead of HTTP. The current AgentField SDK implementation assumes all servers are HTTP-based, causing failures when trying to communicate with stdio servers. This bridge solves that problem.

## How It Works

1. **Process Management**: Starts the stdio MCP server as a subprocess with stdin/stdout pipes
2. **HTTP Server**: Creates FastAPI endpoints that accept HTTP requests
3. **Protocol Translation**: Converts HTTP requests to JSON-RPC 2.0 format for stdio communication
4. **Request Correlation**: Uses unique IDs to match requests with responses
5. **Concurrent Handling**: Queues multiple HTTP requests for the single stdio process

## Key Features

- **HTTP Endpoints**: Provides `/health`, `/mcp/tools/list`, `/mcp/tools/call`, and `/mcp/v1` endpoints
- **JSON-RPC 2.0 Protocol**: Proper MCP protocol implementation with handshake
- **Request Correlation**: Handles multiple concurrent requests reliably
- **Error Handling**: Timeout handling, process crash recovery, proper cleanup
- **Development Mode**: Verbose logging for debugging

## Usage

### Basic Usage

```python
import asyncio
from agentfield.mcp_stdio_bridge import StdioMCPBridge

async def main():
    # Configure your stdio MCP server
    server_config = {
        "alias": "sequential-thinking",
        "run": "npx -y @modelcontextprotocol/server-sequential-thinking",
        "working_dir": ".",
        "environment": {},
        "description": "Sequential thinking MCP server"
    }
    
    # Create and start the bridge
    bridge = StdioMCPBridge(
        server_config=server_config,
        port=8200,
        dev_mode=True
    )
    
    try:
        success = await bridge.start()
        if success:
            print("Bridge started successfully!")
            # Bridge is now running and accepting HTTP requests
            await asyncio.sleep(10)  # Keep running for 10 seconds
        else:
            print("Failed to start bridge")
    finally:
        await bridge.stop()

asyncio.run(main())
```

### Making HTTP Requests

Once the bridge is running, you can make HTTP requests:

```python
import aiohttp

async def test_bridge():
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get("http://localhost:8200/health") as response:
            health = await response.json()
            print(f"Health: {health}")
        
        # List tools
        async with session.post("http://localhost:8200/mcp/tools/list") as response:
            tools = await response.json()
            print(f"Tools: {tools}")
        
        # Call a tool
        tool_request = {
            "name": "example_tool",
            "arguments": {"param": "value"}
        }
        async with session.post("http://localhost:8200/mcp/tools/call", json=tool_request) as response:
            result = await response.json()
            print(f"Result: {result}")
```

### Using with Existing MCP Client

The bridge is designed to work seamlessly with the existing `MCPClient`:

```python
from agentfield.mcp_client import MCPClient
from agentfield.mcp_stdio_bridge import StdioMCPBridge

# Start the bridge
bridge = StdioMCPBridge(server_config, port=8200)
await bridge.start()

# Use existing MCP client
client = MCPClient("sequential-thinking", port=8200, dev_mode=True)
tools = await client.list_tools()
result = await client.call_tool("tool_name", {"arg": "value"})
```

## Configuration

The `server_config` dictionary should contain:

- `alias`: Human-readable name for the server
- `run`: Command to start the stdio MCP server
- `working_dir`: Working directory for the process (optional)
- `environment`: Environment variables (optional)
- `description`: Description of the server (optional)

## HTTP Endpoints

### GET /health
Returns the health status of the bridge and stdio process.

**Response:**
```json
{
  "status": "healthy",
  "bridge": "running", 
  "process": "running"
}
```

### POST /mcp/tools/list
Lists available tools from the stdio MCP server.

**Response:**
```json
{
  "tools": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "inputSchema": {...}
    }
  ]
}
```

### POST /mcp/tools/call
Calls a specific tool on the stdio MCP server.

**Request:**
```json
{
  "name": "tool_name",
  "arguments": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

**Response:**
```json
{
  "content": [...],
  "isError": false
}
```

### POST /mcp/v1
Standard MCP JSON-RPC 2.0 endpoint.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {...}
}
```

## Error Handling

The bridge handles various error conditions:

- **Process startup failures**: Returns startup errors with stderr output
- **Request timeouts**: 30-second timeout for stdio requests
- **Process crashes**: Automatic cleanup and error reporting
- **Invalid JSON**: Proper error responses for malformed requests
- **MCP protocol errors**: Forwards MCP server errors to HTTP clients

## Development Mode

Enable development mode for verbose logging:

```python
bridge = StdioMCPBridge(server_config, port=8200, dev_mode=True)
```

This will log:
- Process startup details
- Request/response correlation
- MCP protocol messages
- Error details

## Dependencies

The bridge requires:
- `fastapi`: HTTP server framework
- `uvicorn`: ASGI server
- `asyncio`: Async process management
- Standard library modules: `json`, `subprocess`, `uuid`, `logging`

## Thread Safety

The bridge is designed for async/await usage and handles concurrent requests safely through:
- Request correlation with unique IDs
- Async queuing of stdio requests
- Proper cleanup of resources
- Thread-safe request/response matching