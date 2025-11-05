from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientTimeout

from agentfield.logger import log_debug, log_error, log_info, log_warn


class MCPClient:
    def __init__(self, base_url: str, alias: str, dev_mode: bool = False):
        self.server_alias = alias
        self.base_url = base_url
        self.dev_mode = dev_mode
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_stdio_bridge = False  # Default to direct HTTP

    # Legacy constructor support for backward compatibility
    @classmethod
    def from_port(cls, server_alias: str, port: int, dev_mode: bool = False):
        """Create MCPClient from port (legacy method for backward compatibility)"""
        base_url = f"http://localhost:{port}"
        return cls(base_url, server_alias, dev_mode)

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> bool:
        """Check if MCP server is healthy"""
        try:
            await self._ensure_session()
            if self.session is None:
                return False
            timeout = ClientTimeout(total=5)

            # Use /health endpoint for both bridge and direct HTTP
            async with self.session.get(
                f"{self.base_url}/health", timeout=timeout
            ) as response:
                return response.status == 200
        except Exception as e:
            if self.dev_mode:
                log_warn(f"Health check failed for {self.server_alias}: {e}")
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server"""
        try:
            await self._ensure_session()
            if self.session is None:
                return []

            timeout = ClientTimeout(total=10)

            if getattr(self, "_is_stdio_bridge", False):
                # Use bridge endpoint
                endpoint = "/mcp/tools/list"
                async with self.session.post(
                    f"{self.base_url}{endpoint}", timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tools = data.get("tools", [])
                        if self.dev_mode:
                            log_debug(
                                f"Found {len(tools)} tools in {self.server_alias} (stdio bridge)"
                            )
                        return tools
            else:
                # Use direct HTTP endpoint
                request_data = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }

                async with self.session.post(
                    f"{self.base_url}/mcp/v1", json=request_data, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data and "tools" in data["result"]:
                            tools = data["result"]["tools"]
                            if self.dev_mode:
                                log_debug(
                                    f"Found {len(tools)} tools in {self.server_alias} (direct HTTP)"
                                )
                            return tools

        except Exception as e:
            if self.dev_mode:
                log_error(f"Failed to list tools for {self.server_alias}: {e}")

        return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call specific tool on MCP server"""
        try:
            await self._ensure_session()
            if self.session is None:
                raise Exception("Session not available")

            if self.dev_mode:
                transport_type = (
                    "stdio bridge"
                    if getattr(self, "_is_stdio_bridge", False)
                    else "direct HTTP"
                )
                log_debug(
                    f"Calling {self.server_alias}.{tool_name} with args: {arguments} ({transport_type})"
                )

            timeout = ClientTimeout(total=30)

            if getattr(self, "_is_stdio_bridge", False):
                # Use bridge endpoint
                request_data = {"tool_name": tool_name, "arguments": arguments}

                async with self.session.post(
                    f"{self.base_url}/mcp/tools/call",
                    json=request_data,
                    timeout=timeout,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        raise Exception(
                            f"HTTP {response.status}: {await response.text()}"
                        )
            else:
                # Use direct HTTP endpoint
                request_data = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }

                async with self.session.post(
                    f"{self.base_url}/mcp/v1", json=request_data, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data:
                            return data["result"]
                        elif "error" in data:
                            raise Exception(f"MCP tool error: {data['error']}")
                    else:
                        raise Exception(
                            f"HTTP {response.status}: {await response.text()}"
                        )

        except Exception as e:
            if self.dev_mode:
                log_error(f"Tool call failed {self.server_alias}.{tool_name}: {e}")
            raise Exception(
                f"MCP tool '{self.server_alias}.{tool_name}' failed: {str(e)}"
            )


class MCPClientRegistry:
    """Registry to manage MCP clients for all servers"""

    def __init__(self, dev_mode: bool = False):
        self.clients: Dict[str, MCPClient] = {}
        self.dev_mode = dev_mode

    def register_client(self, alias: str, port: int):
        """Register MCP client for server"""
        base_url = f"http://localhost:{port}"
        client = MCPClient(base_url, alias, self.dev_mode)
        self.clients[alias] = client

        if self.dev_mode:
            log_info(f"Registered MCP client for {alias} on port {port}")

    def register_stdio_bridge_client(self, alias: str, bridge_port: int) -> None:
        """Register a client for a stdio bridge server"""
        base_url = f"http://localhost:{bridge_port}"
        client = MCPClient(base_url, alias, self.dev_mode)
        client._is_stdio_bridge = True  # Mark as bridge client
        self.clients[alias] = client
        if self.dev_mode:
            log_info(
                f"Registered stdio bridge client for {alias} on port {bridge_port}"
            )

    def get_client(self, alias: str) -> Optional[MCPClient]:
        """Get MCP client by server alias"""
        return self.clients.get(alias)

    async def close_all(self):
        """Close all MCP clients"""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
