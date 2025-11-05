import asyncio
import json
import os
import subprocess
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .logger import get_logger
from .mcp_stdio_bridge import StdioMCPBridge

logger = get_logger(__name__)


@dataclass
class MCPServerConfig:
    alias: str
    run_command: str
    working_dir: str
    environment: Dict[str, str]
    health_check: Optional[str] = None
    port: Optional[int] = None
    transport: str = "http"


@dataclass
class MCPServerProcess:
    config: MCPServerConfig
    process: Optional[subprocess.Popen] = None
    port: Optional[int] = None
    status: str = "stopped"  # stopped, starting, running, failed


class MCPManager:
    def __init__(self, agent_directory: str, dev_mode: bool = False):
        self.agent_directory = agent_directory
        self.dev_mode = dev_mode
        self.servers: Dict[str, MCPServerProcess] = {}
        self.stdio_bridges: Dict[str, StdioMCPBridge] = {}
        self.port_range_start = 8100  # Start assigning ports from 8100
        self.used_ports = set()

    def discover_mcp_servers(self) -> List[MCPServerConfig]:
        """Discover MCP servers from packages/mcp/ directory"""
        mcp_dir = os.path.join(self.agent_directory, "packages", "mcp")
        servers = []

        if not os.path.exists(mcp_dir):
            if self.dev_mode:
                logger.debug(f"No MCP directory found at {mcp_dir}")
            return servers

        for item in os.listdir(mcp_dir):
            server_dir = os.path.join(mcp_dir, item)
            config_file = os.path.join(server_dir, "config.json")

            if os.path.isdir(server_dir) and os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config_data = json.load(f)

                    config = MCPServerConfig(
                        alias=config_data.get("alias", item),
                        run_command=config_data.get("run", ""),
                        working_dir=server_dir,
                        environment=config_data.get("environment", {}),
                        health_check=config_data.get("health_check"),
                        transport=config_data.get("transport", "http"),
                    )
                    servers.append(config)

                    if self.dev_mode:
                        logger.debug(f"Discovered MCP server: {config.alias}")

                except Exception as e:
                    if self.dev_mode:
                        logger.warning(f"Failed to load config for {item}: {e}")

        return servers

    def _get_next_available_port(self) -> int:
        """Get next available port for MCP server"""
        import socket

        for port in range(self.port_range_start, self.port_range_start + 1000):
            if port not in self.used_ports:
                # Test if port is actually available
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("localhost", port))
                        self.used_ports.add(port)
                        return port
                except OSError:
                    continue

        raise RuntimeError("No available ports for MCP servers")

    def _detect_transport(self, config: MCPServerConfig) -> str:
        """Detect transport type from config"""
        return config.transport

    async def _start_stdio_server(self, config: MCPServerConfig) -> bool:
        """Start stdio MCP server using bridge"""
        try:
            # Assign port for the bridge
            port = self._get_next_available_port()
            config.port = port

            if self.dev_mode:
                logger.info(f"Starting stdio MCP server: {config.alias} on port {port}")
                logger.debug(f"Command: {config.run_command}")

            # Prepare server config for bridge
            server_config = {
                "run": config.run_command,
                "working_dir": config.working_dir,
                "environment": config.environment,
            }

            # Create and start stdio bridge
            bridge = StdioMCPBridge(
                server_config=server_config, port=port, dev_mode=self.dev_mode
            )

            success = await bridge.start()
            if success:
                self.stdio_bridges[config.alias] = bridge
                if self.dev_mode:
                    logger.info(f"Stdio MCP server {config.alias} started successfully")
                return True
            else:
                if self.dev_mode:
                    logger.error(f"Stdio MCP server {config.alias} failed to start")
                return False

        except Exception as e:
            if self.dev_mode:
                logger.error(f"Error starting stdio MCP server {config.alias}: {e}")
            return False

    async def _start_http_server(self, config: MCPServerConfig) -> bool:
        """Start HTTP MCP server (original implementation)"""
        try:
            # Assign port
            port = self._get_next_available_port()
            config.port = port

            # Prepare command with port substitution
            run_command = config.run_command.replace("{{port}}", str(port))

            # Prepare environment
            env = os.environ.copy()
            env.update(config.environment)

            if self.dev_mode:
                logger.info(f"Starting HTTP MCP server: {config.alias} on port {port}")
                logger.debug(f"Command: {run_command}")

            # Start process
            process = subprocess.Popen(
                run_command.split(),
                cwd=config.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Create server process object
            server_process = MCPServerProcess(
                config=config, process=process, port=port, status="starting"
            )

            self.servers[config.alias] = server_process

            # Wait a moment for startup
            await asyncio.sleep(2)

            # Check if process is still running
            if process.poll() is None:
                server_process.status = "running"
                if self.dev_mode:
                    logger.info(f"HTTP MCP server {config.alias} started successfully")
                return True
            else:
                server_process.status = "failed"
                if self.dev_mode:
                    logger.error(f"HTTP MCP server {config.alias} failed to start")
                return False

        except Exception as e:
            if self.dev_mode:
                logger.error(f"Error starting HTTP MCP server {config.alias}: {e}")
            if config.alias in self.servers:
                self.servers[config.alias].status = "failed"
            return False

    async def start_server(self, config: MCPServerConfig) -> bool:
        """Start individual MCP server"""
        transport = self._detect_transport(config)
        if transport == "stdio":
            return await self._start_stdio_server(config)
        else:
            return await self._start_http_server(config)

    async def start_all_servers(self) -> Dict[str, bool]:
        """Start all discovered MCP servers"""
        configs = self.discover_mcp_servers()
        results = {}

        if self.dev_mode:
            logger.info(f"Starting {len(configs)} MCP servers...")

        for config in configs:
            success = await self.start_server(config)
            results[config.alias] = success

        return results

    def get_server_status(self, alias: str) -> Optional[Dict[str, Any]]:
        """Get status of specific MCP server"""
        # Check stdio bridges first
        if alias in self.stdio_bridges:
            bridge = self.stdio_bridges[alias]
            return {
                "alias": alias,
                "transport": "stdio",
                "port": bridge.port,
                "status": "running" if bridge.running else "stopped",
                "initialized": bridge.initialized,
            }

        # Check HTTP servers
        if alias in self.servers:
            server_process = self.servers[alias]
            return {
                "alias": alias,
                "transport": "http",
                "port": server_process.port,
                "status": server_process.status,
                "config": server_process.config,
            }

        return None

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers"""
        all_status = {}

        # Add stdio bridges
        for alias, bridge in self.stdio_bridges.items():
            all_status[alias] = {
                "alias": alias,
                "transport": "stdio",
                "port": bridge.port,
                "status": "running" if bridge.running else "stopped",
                "initialized": bridge.initialized,
            }

        # Add HTTP servers
        for alias, server_process in self.servers.items():
            all_status[alias] = {
                "alias": alias,
                "transport": "http",
                "port": server_process.port,
                "status": server_process.status,
                "config": server_process.config,
            }

        return all_status

    async def stop_server(self, alias: str) -> bool:
        """Stop specific MCP server"""
        # Check if it's a stdio bridge
        if alias in self.stdio_bridges:
            bridge = self.stdio_bridges[alias]
            await bridge.stop()
            if bridge.port:
                self.used_ports.discard(bridge.port)
            del self.stdio_bridges[alias]
            if self.dev_mode:
                logger.info(f"Stopped stdio MCP server: {alias}")
            return True

        # Check if it's an HTTP server
        if alias in self.servers:
            server_process = self.servers[alias]
            if server_process.process and server_process.process.poll() is None:
                server_process.process.terminate()
                try:
                    server_process.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.process.kill()

            server_process.status = "stopped"
            if server_process.port:
                self.used_ports.discard(server_process.port)

            if self.dev_mode:
                logger.info(f"Stopped HTTP MCP server: {alias}")
            return True

        return False

    async def start_server_by_alias(self, alias: str) -> bool:
        """Start MCP server by alias"""
        # Find the config for this alias
        configs = self.discover_mcp_servers()
        for config in configs:
            if config.alias == alias:
                return await self.start_server(config)

        if self.dev_mode:
            logger.warning(f"No configuration found for MCP server: {alias}")
        return False

    async def restart_server(self, alias: str) -> bool:
        """Restart MCP server by alias"""
        # Stop first
        stop_success = await self.stop_server(alias)
        if self.dev_mode:
            logger.info(f"Stopped '{alias}' for restart: {stop_success}")

        # Wait a moment for cleanup
        await asyncio.sleep(1)

        # Start again
        return await self.start_server_by_alias(alias)

    async def shutdown_all(self) -> None:
        """Stop all MCP servers"""
        if self.dev_mode:
            logger.info("Shutting down all MCP servers...")

        # Stop all stdio bridges
        for alias in list(self.stdio_bridges.keys()):
            await self.stop_server(alias)

        # Stop all HTTP servers
        for alias in list(self.servers.keys()):
            await self.stop_server(alias)
