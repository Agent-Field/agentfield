import asyncio
import importlib.util
import os
import signal
import urllib.parse
from datetime import datetime
from typing import Optional

import uvicorn
from agentfield.agent_utils import AgentUtils
from agentfield.logger import log_debug, log_error, log_info, log_success, log_warn
from agentfield.utils import get_free_port
from fastapi import Request
from fastapi.routing import APIRoute


class AgentServer:
    """Server management functionality for AgentField Agent"""

    def __init__(self, agent_instance):
        """
        Initialize the AgentServer with a reference to the agent instance.

        Args:
            agent_instance: The Agent instance this server manages
        """
        self.agent = agent_instance

    def setup_agentfield_routes(self):
        """Setup standard routes that AgentField server expects"""

        @self.agent.get("/health")
        async def health():
            health_response = {
                "status": "healthy",
                "node_id": self.agent.node_id,
                "version": self.agent.version,
                "timestamp": datetime.now().isoformat(),
            }

            # Add MCP server status if manager is available
            if self.agent.mcp_manager:
                try:
                    all_status = self.agent.mcp_manager.get_all_status()

                    # Calculate summary statistics
                    total_servers = len(all_status)
                    running_servers = sum(
                        1
                        for server in all_status.values()
                        if server.get("status") == "running"
                    )
                    failed_servers = sum(
                        1
                        for server in all_status.values()
                        if server.get("status") == "failed"
                    )

                    # Determine overall health status
                    if failed_servers > 0:
                        health_response["status"] = "degraded"

                    # Add MCP information to health response
                    mcp_server_info = {
                        "total": total_servers,
                        "running": running_servers,
                        "failed": failed_servers,
                        "servers": {},
                    }

                    # Add individual server details
                    for alias, server_process in all_status.items():
                        process = server_process.get("process")
                        server_info = {
                            "status": server_process.get("status"),
                            "port": server_process.get("port"),
                            "pid": process.pid if process else None,
                        }
                        mcp_server_info["servers"][alias] = server_info

                    health_response["mcp_servers"] = mcp_server_info

                except Exception as e:
                    if self.agent.dev_mode:
                        log_warn(f"Error getting MCP status for health check: {e}")
                    health_response["mcp_servers"] = {
                        "error": "Failed to get MCP status",
                        "total": 0,
                        "running": 0,
                        "failed": 0,
                    }

            return health_response

        @self.agent.get("/reasoners")
        async def list_reasoners():
            return {"reasoners": self.agent.reasoners}

        @self.agent.get("/skills")
        async def list_skills():
            return {"skills": self.agent.skills}

        @self.agent.post("/shutdown")
        async def shutdown_agent(request: Request):
            """
            Graceful shutdown endpoint for the agent.

            This endpoint allows the AgentField server to request a graceful shutdown
            instead of using process signals.
            """
            try:
                # Parse request body for shutdown options
                body = (
                    await request.json()
                    if request.headers.get("content-type") == "application/json"
                    else {}
                )
                graceful = body.get("graceful", True)
                timeout_seconds = body.get("timeout_seconds", 30)

                if self.agent.dev_mode:
                    log_info(
                        f"Shutdown request received (graceful={graceful}, timeout={timeout_seconds}s)"
                    )

                # Set shutdown status
                from agentfield.agent import AgentStatus

                self.agent._shutdown_requested = True
                self.agent._current_status = AgentStatus.OFFLINE

                # Notify AgentField server of shutdown initiation
                try:
                    success = self.agent.client.notify_graceful_shutdown_sync(
                        self.agent.node_id
                    )
                    if self.agent.dev_mode:
                        state = "sent" if success else "failed"
                        log_info(f"Shutdown notification {state}")
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"Shutdown notification error: {e}")

                # Schedule graceful shutdown
                if graceful:
                    asyncio.create_task(self._graceful_shutdown(timeout_seconds))

                    return {
                        "status": "shutting_down",
                        "graceful": True,
                        "timeout_seconds": timeout_seconds,
                        "estimated_shutdown_time": datetime.now().isoformat(),
                        "message": "Graceful shutdown initiated",
                    }
                else:
                    # Immediate shutdown
                    asyncio.create_task(self._immediate_shutdown())

                    return {
                        "status": "shutting_down",
                        "graceful": False,
                        "message": "Immediate shutdown initiated",
                    }

            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Shutdown endpoint error: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to initiate shutdown: {str(e)}",
                }

        @self.agent.get("/status")
        async def get_agent_status():
            """
            Get detailed agent status information.

            This endpoint provides comprehensive status information about the agent,
            including uptime, resource usage, and current state.
            """
            try:
                import time

                import psutil

                # Get process info
                process = psutil.Process()

                # Calculate uptime
                start_time = getattr(self.agent, "_start_time", time.time())
                uptime_seconds = time.time() - start_time
                uptime_formatted = self._format_uptime(uptime_seconds)

                status_response = {
                    "status": (
                        "running"
                        if not getattr(self.agent, "_shutdown_requested", False)
                        else "stopping"
                    ),
                    "uptime": uptime_formatted,
                    "uptime_seconds": int(uptime_seconds),
                    "pid": os.getpid(),
                    "version": self.agent.version,
                    "node_id": self.agent.node_id,
                    "last_activity": datetime.now().isoformat(),
                    "resources": {
                        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                        "cpu_percent": process.cpu_percent(),
                        "threads": process.num_threads(),
                    },
                }

                # Add MCP server information if available
                if self.agent.mcp_manager:
                    try:
                        all_status = self.agent.mcp_manager.get_all_status()
                        status_response["mcp_servers"] = {
                            "total": len(all_status),
                            "running": sum(
                                1
                                for s in all_status.values()
                                if s.get("status") == "running"
                            ),
                            "servers": all_status,
                        }
                    except Exception as e:
                        if self.agent.dev_mode:
                            log_warn(f"Error getting MCP status: {e}")
                        status_response["mcp_servers"] = {"error": str(e)}

                return status_response

            except ImportError:
                # Fallback if psutil is not available
                return {
                    "status": (
                        "running"
                        if not getattr(self.agent, "_shutdown_requested", False)
                        else "stopping"
                    ),
                    "pid": os.getpid(),
                    "version": self.agent.version,
                    "node_id": self.agent.node_id,
                    "last_activity": datetime.now().isoformat(),
                    "message": "Limited status info (psutil not available)",
                }
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Status endpoint error: {e}")
                return {"status": "error", "message": f"Failed to get status: {str(e)}"}

        @self.agent.get("/info")
        async def node_info():
            return {
                "node_id": self.agent.node_id,
                "version": self.agent.version,
                "base_url": self.agent.base_url,
                "reasoners": self.agent.reasoners,
                "skills": self.agent.skills,
                "registered_at": datetime.now().isoformat(),
            }

        @self.agent.get("/mcp/status")
        async def mcp_status():
            """Get status of all MCP servers"""
            if not self.agent.mcp_manager:
                return {
                    "error": "MCP Manager not available",
                    "servers": {},
                    "total": 0,
                    "running": 0,
                    "failed": 0,
                }

            # MCP functionality disabled
            return {
                "error": "MCP functionality disabled - old modules removed",
                "servers": {},
                "total": 0,
                "running": 0,
                "failed": 0,
            }

        @self.agent.post("/mcp/{alias}/start")
        async def start_mcp_server(alias: str):
            """Start a specific MCP server"""
            if not self.agent.mcp_manager:
                return {
                    "success": False,
                    "error": "MCP Process Manager not available",
                    "alias": alias,
                }

            try:
                success = await self.agent.mcp_manager.start_server_by_alias(alias)
                if success:
                    # Get updated status
                    status = self.agent.mcp_manager.get_server_status(alias)
                    return {
                        "success": True,
                        "message": f"MCP server '{alias}' started successfully",
                        "alias": alias,
                        "status": status,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to start MCP server '{alias}'",
                        "alias": alias,
                        "timestamp": datetime.now().isoformat(),
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error starting MCP server '{alias}': {str(e)}",
                    "alias": alias,
                    "timestamp": datetime.now().isoformat(),
                }

        @self.agent.post("/mcp/{alias}/stop")
        async def stop_mcp_server(alias: str):
            """Stop a specific MCP server"""
            if not self.agent.mcp_manager:
                return {
                    "success": False,
                    "error": "MCP Process Manager not available",
                    "alias": alias,
                }

            try:
                success = self.agent.mcp_manager.stop_server(alias)
                if success:
                    return {
                        "success": True,
                        "message": f"MCP server '{alias}' stopped successfully",
                        "alias": alias,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to stop MCP server '{alias}' (may not be running)",
                        "alias": alias,
                        "timestamp": datetime.now().isoformat(),
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error stopping MCP server '{alias}': {str(e)}",
                    "alias": alias,
                    "timestamp": datetime.now().isoformat(),
                }

        @self.agent.post("/mcp/{alias}/restart")
        async def restart_mcp_server(alias: str):
            """Restart a specific MCP server"""
            if not self.agent.mcp_manager:
                return {
                    "success": False,
                    "error": "MCP Process Manager not available",
                    "alias": alias,
                }

            try:
                success = await self.agent.mcp_manager.restart_server(alias)
                if success:
                    # Get updated status
                    status = self.agent.mcp_manager.get_server_status(alias)
                    return {
                        "success": True,
                        "message": f"MCP server '{alias}' restarted successfully",
                        "alias": alias,
                        "status": status,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to restart MCP server '{alias}'",
                        "alias": alias,
                        "timestamp": datetime.now().isoformat(),
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error restarting MCP server '{alias}': {str(e)}",
                    "alias": alias,
                    "timestamp": datetime.now().isoformat(),
                }

        @self.agent.get("/health/mcp")
        async def mcp_health():
            """Get MCP health information in the format expected by AgentField server"""
            if not self.agent.mcp_manager:
                # Return empty response when MCP manager is not available
                return {
                    "servers": [],
                    "summary": {
                        "total_servers": 0,
                        "running_servers": 0,
                        "total_tools": 0,
                        "overall_health": 0.0,
                    },
                }

            try:
                # Get all server status from MCP manager
                all_status = self.agent.mcp_manager.get_all_status()
                servers = []
                total_tools = 0
                running_servers = 0

                # Process each server to get detailed health information
                for alias, server_info in all_status.items():
                    server_health = {
                        "alias": alias,
                        "status": server_info.get("status", "unknown"),
                        "tool_count": 0,
                        "started_at": None,
                        "last_health_check": datetime.now().isoformat(),
                        "port": server_info.get("port"),
                        "process_id": None,
                    }

                    # Get process ID if available
                    if alias in self.agent.mcp_manager.servers:
                        server_process = self.agent.mcp_manager.servers[alias]
                        if server_process.process:
                            server_health["process_id"] = server_process.process.pid

                    # Count running servers
                    if server_health["status"] == "running":
                        running_servers += 1

                        # Try to get tool count from MCP client
                        try:
                            if self.agent.mcp_client_registry:
                                client = self.agent.mcp_client_registry.get_client(
                                    alias
                                )
                                if client:
                                    tools = await client.list_tools()
                                    server_health["tool_count"] = len(tools)
                                    total_tools += len(tools)

                                    # Set started_at time (approximate)
                                    server_health["started_at"] = (
                                        datetime.now().isoformat()
                                    )

                        except Exception as e:
                            if self.agent.dev_mode:
                                log_warn(f"Failed to get tools for {alias}: {e}")

                    servers.append(server_health)

                # Calculate overall health score
                total_servers = len(servers)
                if total_servers == 0:
                    overall_health = 0.0
                else:
                    # Health score based on running servers ratio
                    health_ratio = running_servers / total_servers
                    # Adjust for any servers with errors
                    error_servers = sum(1 for s in servers if s["status"] == "error")
                    if error_servers > 0:
                        health_ratio *= 1 - (
                            error_servers * 0.2
                        )  # Reduce health for errors
                    overall_health = max(0.0, min(1.0, health_ratio))

                # Build summary
                summary = {
                    "total_servers": total_servers,
                    "running_servers": running_servers,
                    "total_tools": total_tools,
                    "overall_health": overall_health,
                }

                return {"servers": servers, "summary": summary}

            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Error getting MCP health: {e}")

                # Return error response in expected format
                return {
                    "servers": [],
                    "summary": {
                        "total_servers": 0,
                        "running_servers": 0,
                        "total_tools": 0,
                        "overall_health": 0.0,
                    },
                }

        @self.agent.post("/mcp/servers/{alias}/restart")
        async def restart_mcp_server_alt(alias: str):
            """Alternative restart endpoint for AgentField server compatibility"""
            return await restart_mcp_server(alias)

        @self.agent.get("/mcp/servers/{alias}/tools")
        async def get_mcp_server_tools(alias: str):
            """Get tools from a specific MCP server"""
            if not self.agent.mcp_client_registry:
                return {"error": "MCP Client Registry not available", "tools": []}

            try:
                client = self.agent.mcp_client_registry.get_client(alias)
                if not client:
                    return {
                        "error": f"MCP server '{alias}' not found or not running",
                        "tools": [],
                    }

                tools = await client.list_tools()

                # Transform tools to match expected format
                formatted_tools = []
                for tool in tools:
                    formatted_tool = {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("inputSchema", {}),
                    }
                    formatted_tools.append(formatted_tool)

                return {"tools": formatted_tools}

            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Error getting tools for {alias}: {e}")

                return {
                    "error": f"Failed to get tools from MCP server '{alias}': {str(e)}",
                    "tools": [],
                }

    async def _graceful_shutdown(self, timeout_seconds: int = 30):
        """
        Perform graceful shutdown with cleanup.

        Args:
            timeout_seconds: Maximum time to wait for graceful shutdown
        """
        try:
            if self.agent.dev_mode:
                log_info(f"Starting graceful shutdown (timeout: {timeout_seconds}s)")

            # Stop MCP servers first
            try:
                if hasattr(self.agent, "mcp_handler") and self.agent.mcp_handler:
                    self.agent.mcp_handler._cleanup_mcp_servers()
                    if self.agent.dev_mode:
                        log_info("MCP servers stopped")
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"MCP shutdown error: {e}")

            # Stop heartbeat
            try:
                if (
                    hasattr(self.agent, "agentfield_handler")
                    and self.agent.agentfield_handler
                ):
                    self.agent.agentfield_handler.stop_heartbeat()
                    if self.agent.dev_mode:
                        log_debug("Heartbeat stopped")
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Heartbeat stop error: {e}")

            # Clear agent registry
            try:
                from agentfield.agent_registry import clear_current_agent

                clear_current_agent()
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Registry clear error: {e}")

            # Wait a moment for cleanup to complete
            await asyncio.sleep(1)

            if self.agent.dev_mode:
                log_success("Graceful shutdown completed")

            # Exit the process
            os._exit(0)

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Graceful shutdown error: {e}")
            # Fallback to immediate shutdown
            await self._immediate_shutdown()

    async def _immediate_shutdown(self):
        """
        Perform immediate shutdown without cleanup.
        """
        try:
            if self.agent.dev_mode:
                log_warn("Immediate shutdown initiated")

            # Quick cleanup attempt
            try:
                if hasattr(self.agent, "mcp_handler") and self.agent.mcp_handler:
                    self.agent.mcp_handler._cleanup_mcp_servers()
            except Exception:
                pass  # Ignore errors in immediate shutdown

            # Exit immediately
            os._exit(0)

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Immediate shutdown error: {e}")
            os._exit(1)

    def _format_uptime(self, uptime_seconds: float) -> str:
        """
        Format uptime seconds into a human-readable string.

        Args:
            uptime_seconds: Uptime in seconds

        Returns:
            Formatted uptime string (e.g., "2h 30m 15s")
        """
        try:
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)

            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if seconds > 0 or not parts:  # Always show seconds if no other parts
                parts.append(f"{seconds}s")

            return " ".join(parts)
        except Exception:
            return f"{int(uptime_seconds)}s"

    def _validate_ssl_config(
        self, ssl_keyfile: Optional[str], ssl_certfile: Optional[str]
    ) -> bool:
        """
        Validate SSL configuration files exist and are readable.

        Args:
            ssl_keyfile: Path to SSL key file
            ssl_certfile: Path to SSL certificate file

        Returns:
            True if SSL configuration is valid, False otherwise
        """
        if not ssl_keyfile or not ssl_certfile:
            return False

        try:
            # Check if files exist and are readable
            if not os.path.isfile(ssl_keyfile):
                if self.agent.dev_mode:
                    log_error(f"SSL key file not found: {ssl_keyfile}")
                return False

            if not os.path.isfile(ssl_certfile):
                if self.agent.dev_mode:
                    log_error(f"SSL certificate file not found: {ssl_certfile}")
                return False

            # Check file permissions
            if not os.access(ssl_keyfile, os.R_OK):
                if self.agent.dev_mode:
                    log_error(f"SSL key file not readable: {ssl_keyfile}")
                return False

            if not os.access(ssl_certfile, os.R_OK):
                if self.agent.dev_mode:
                    log_error(f"SSL certificate file not readable: {ssl_certfile}")
                return False

            return True

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"SSL validation error: {e}")
            return False

    def _get_optimal_workers(self, workers: Optional[int] = None) -> Optional[int]:
        """
        Determine optimal number of workers based on system resources.

        Args:
            workers: Explicitly requested number of workers

        Returns:
            Optimal number of workers or None for single process
        """
        if workers is not None:
            return workers

        # Check environment variable
        env_workers = os.getenv("UVICORN_WORKERS")
        if env_workers and env_workers.isdigit():
            return int(env_workers)

        # Auto-detect based on CPU cores (only in production)
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()

            # Use 2 * CPU cores for I/O bound workloads, but cap at 8
            optimal_workers = min(cpu_count * 2, 8)

            if self.agent.dev_mode:
                log_debug(
                    f"Detected {cpu_count} CPU cores, optimal workers: {optimal_workers}"
                )

            return optimal_workers

        except Exception:
            return None

    def _check_performance_dependencies(self) -> dict:
        """
        Check availability of performance-enhancing dependencies.

        Returns:
            Dictionary with availability status of optional dependencies
        """
        deps = {
            "uvloop": False,
            "psutil": False,
            "orjson": False,
        }

        if importlib.util.find_spec("uvloop") is not None:
            deps["uvloop"] = True

        if importlib.util.find_spec("psutil") is not None:
            deps["psutil"] = True

        if importlib.util.find_spec("orjson") is not None:
            deps["orjson"] = True

        return deps

    def setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for graceful shutdown.

        This method registers signal handlers for SIGTERM and SIGINT
        to ensure MCP servers are properly stopped when the agent shuts down.
        """
        try:
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)

            if self.agent.dev_mode:
                log_debug("Signal handlers registered for graceful shutdown")

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Failed to setup signal handlers: {e}")
            # Continue without signal handlers - not critical

    def signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"

        if self.agent.dev_mode:
            log_warn(f"{signal_name} received, shutting down gracefully...")

        # Perform cleanup
        self.agent.mcp_handler._cleanup_mcp_servers()

        # Exit gracefully
        os._exit(0)

    def serve(
        self,
        port: Optional[int] = None,
        host: str = "0.0.0.0",
        dev: bool = False,
        heartbeat_interval: int = 2,  # Fast heartbeat for real-time detection
        auto_port: bool = False,
        workers: Optional[int] = None,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        log_level: str = "info",
        access_log: bool = True,
        **kwargs,
    ):
        """
        Start the agent node server with intelligent port management and production-ready configuration.

        This method implements smart port resolution that seamlessly works with AgentField CLI
        or standalone execution. The port selection priority is:
        1. Explicit port parameter (highest priority)
        2. PORT environment variable (AgentField CLI integration)
        3. auto_port=True: find free port automatically
        4. Default fallback with availability check

        Args:
            port (int, optional): The port on which the agent server will listen.
                                If specified, this takes highest priority.
            host (str): The host address for the agent server. Defaults to "0.0.0.0".
            dev (bool): If True, enables development mode features (e.g., hot reload, debug UI).
            heartbeat_interval (int): The interval in seconds for sending heartbeats to the AgentField server.
                                      Defaults to 2 seconds (fast detection architecture).
            auto_port (bool): If True, automatically find an available port. Defaults to False.
            workers (int, optional): Number of worker processes for production. If None, uses single process.
            ssl_keyfile (str, optional): Path to SSL key file for HTTPS.
            ssl_certfile (str, optional): Path to SSL certificate file for HTTPS.
            log_level (str): Log level for uvicorn. Defaults to "info".
            access_log (bool): Enable/disable access logging. Defaults to True.
            **kwargs: Additional keyword arguments to pass to `uvicorn.run`.
        """
        # Smart port resolution with priority order
        if port is None:
            # Check for AgentField CLI integration via environment variable
            env_port = os.getenv("PORT")
            if env_port and env_port.isdigit():
                suggested_port = int(env_port)
                if AgentUtils.is_port_available(suggested_port):
                    port = suggested_port
                    if self.agent.dev_mode:
                        log_debug(f"Using port from AgentField CLI: {port}")
                else:
                    # AgentField CLI suggested port is taken, find next available
                    try:
                        port = get_free_port(start_port=suggested_port)
                        if self.agent.dev_mode:
                            log_debug(
                                f"AgentField CLI port {suggested_port} taken, using {port}"
                            )
                    except RuntimeError:
                        port = get_free_port()  # Fallback to default range
                        if self.agent.dev_mode:
                            log_debug(f"Using fallback port: {port}")
            elif auto_port or os.getenv("AGENTFIELD_AUTO_PORT") == "true":
                # Auto-port mode: find any available port
                try:
                    port = get_free_port()
                    if self.agent.dev_mode:
                        log_debug(f"Auto-assigned port: {port}")
                except RuntimeError as e:
                    log_error(f"Failed to find free port: {e}")
                    port = 8001  # Fallback to default
            else:
                # Default behavior: try 8001, find alternative if taken
                if AgentUtils.is_port_available(8001):
                    port = 8001
                else:
                    try:
                        port = get_free_port()
                        if self.agent.dev_mode:
                            log_debug(f"Default port 8001 taken, using {port}")
                    except RuntimeError:
                        port = 8001  # Force use even if taken (will fail gracefully)
        else:
            # Explicit port provided - validate it's available
            if not AgentUtils.is_port_available(port):
                if self.agent.dev_mode:
                    log_warn(f"Requested port {port} is not available")
                # Try to find an alternative near the requested port
                try:
                    alternative_port = get_free_port(start_port=port)
                    if self.agent.dev_mode:
                        log_debug(f"Using alternative port: {alternative_port}")
                    port = alternative_port
                except RuntimeError:
                    if self.agent.dev_mode:
                        log_warn(
                            f"No alternative ports found, attempting to use {port}"
                        )
                    # Continue with original port (will fail if truly unavailable)

        log_info(f"Starting agent node '{self.agent.node_id}' on port {port}")

        # Set base_url for registration - preserve explicit callback URL if set
        if not self.agent.base_url:
            # Check AGENT_CALLBACK_URL environment variable before defaulting to localhost
            env_callback_url = os.getenv("AGENT_CALLBACK_URL")
            if env_callback_url:
                # Parse the environment variable URL to extract the hostname
                try:
                    parsed = urllib.parse.urlparse(env_callback_url)
                    if parsed.hostname:
                        self.agent.base_url = (
                            f"{parsed.scheme or 'http'}://{parsed.hostname}:{port}"
                        )
                        if self.agent.dev_mode:
                            log_debug(
                                f"Using AGENT_CALLBACK_URL from environment: {self.agent.base_url}"
                            )
                    else:
                        # Invalid URL in env var, fall back to localhost
                        self.agent.base_url = f"http://localhost:{port}"
                except Exception:
                    # Failed to parse env var, fall back to localhost
                    self.agent.base_url = f"http://localhost:{port}"
            else:
                # No env var set, use localhost
                self.agent.base_url = f"http://localhost:{port}"
        else:
            # Update port in existing base_url if needed
            parsed = urllib.parse.urlparse(self.agent.base_url)
            if parsed.port != port:
                # Update the port in the existing URL, but preserve the hostname
                self.agent.base_url = f"{parsed.scheme}://{parsed.hostname}:{port}"
                if self.agent.dev_mode:
                    log_debug(f"Updated port in callback URL: {self.agent.base_url}")
            elif self.agent.dev_mode:
                log_debug(f"Using explicit callback URL: {self.agent.base_url}")

        # Start heartbeat worker
        self.agent.agentfield_handler.start_heartbeat(heartbeat_interval)

        log_info(f"Agent server running at http://{host}:{port}")
        log_info("Available endpoints:")
        for route in self.agent.routes:
            # Check if the route is an APIRoute (has .path and .methods)
            if isinstance(route, APIRoute):
                for method in route.methods:
                    if method != "HEAD":  # Skip HEAD methods
                        log_debug(f"Endpoint registered: {method} {route.path}")

        # Setup fast lifecycle signal handlers
        self.agent.agentfield_handler.setup_fast_lifecycle_signal_handlers()

        # Add startup event handler for resilient lifecycle
        @self.agent.on_event("startup")
        async def startup_resilient_lifecycle():
            """Resilient lifecycle startup: connection manager handles AgentField server connectivity"""

            # Initialize connection manager
            from agentfield.connection_manager import (
                ConnectionConfig,
                ConnectionManager,
            )

            # Configure connection manager with reasonable retry interval
            config = ConnectionConfig(
                retry_interval=10.0,  # Check every 10 seconds for AgentField server
                health_check_interval=30.0,
                connection_timeout=10.0,
            )

            self.agent.connection_manager = ConnectionManager(self.agent, config)

            # Set up callbacks for connection state changes
            def on_connected():
                if self.agent.dev_mode:
                    log_info(
                        "Connected to AgentField server - full functionality available"
                    )
                # Kick a heartbeat immediately so the control plane renews the lease
                try:
                    asyncio.create_task(
                        self.agent.agentfield_handler.send_enhanced_heartbeat()
                    )
                except RuntimeError:
                    # Event loop not running; the heartbeat worker will recover shortly
                    pass
                # Start enhanced heartbeat when connected
                if (
                    not hasattr(self.agent, "_heartbeat_task")
                    or self.agent._heartbeat_task.done()
                ):
                    self.agent._heartbeat_task = asyncio.create_task(
                        self.agent.agentfield_handler.enhanced_heartbeat_loop(
                            heartbeat_interval
                        )
                    )

            def on_disconnected():
                if self.agent.dev_mode:
                    log_warn("AgentField server disconnected - running in local mode")
                # Cancel heartbeat task when disconnected
                if (
                    hasattr(self.agent, "_heartbeat_task")
                    and not self.agent._heartbeat_task.done()
                ):
                    self.agent._heartbeat_task.cancel()

            self.agent.connection_manager.on_connected = on_connected
            self.agent.connection_manager.on_disconnected = on_disconnected

            # Start connection manager (non-blocking)
            connected = await self.agent.connection_manager.start()

            # Always connect memory event client and start MCP initialization
            # These work independently of AgentField server connection
            if self.agent.memory_event_client:
                try:
                    await self.agent.memory_event_client.connect()
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"Memory event client connection failed: {e}")

            # Start background MCP initialization (non-blocking)
            asyncio.create_task(self.agent.mcp_handler._background_mcp_initialization())

            if connected:
                if self.agent.dev_mode:
                    log_info("Agent started with AgentField server connection")
            else:
                if self.agent.dev_mode:
                    log_info(
                        "Agent started in local mode - will connect to AgentField server when available"
                    )

        # Add shutdown event handler for cleanup
        @self.agent.on_event("shutdown")
        async def shutdown_cleanup():
            """Cleanup all resources when FastAPI shuts down"""

            # Stop connection manager
            if self.agent.connection_manager:
                await self.agent.connection_manager.stop()

            # Close memory event client
            if self.agent.memory_event_client:
                await self.agent.memory_event_client.close()

            # Stop MCP servers
            if self.agent.mcp_manager:
                try:
                    await self.agent.mcp_manager.shutdown_all()
                    if self.agent.dev_mode:
                        log_info("MCP servers stopped")
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"MCP shutdown error: {e}")

            if self.agent.mcp_client_registry:
                try:
                    await self.agent.mcp_client_registry.close_all()
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"MCP client shutdown error: {e}")

            if getattr(self.agent, "client", None):
                try:
                    await self.agent.client.aclose()
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"AgentField client shutdown error: {e}")

            # Clear agent from thread-local storage during shutdown
            from agentfield.agent_registry import clear_current_agent

            clear_current_agent()

        # Configure uvicorn parameters based on environment and requirements
        uvicorn_config = {
            "host": host,
            "port": port,
            "reload": dev
            and workers is None,  # Only enable reload in dev mode with single worker
            "access_log": access_log,
            "log_level": log_level,
            "timeout_graceful_shutdown": 30,  # Allow 30 seconds for graceful shutdown
            **kwargs,
        }

        # Add SSL configuration if provided and valid
        if ssl_keyfile and ssl_certfile:
            if self._validate_ssl_config(ssl_keyfile, ssl_certfile):
                uvicorn_config.update(
                    {
                        "ssl_keyfile": ssl_keyfile,
                        "ssl_certfile": ssl_certfile,
                    }
                )
                if self.agent.dev_mode:
                    log_info("HTTPS enabled with SSL certificates")
            else:
                log_error("Invalid SSL configuration, falling back to HTTP")
                ssl_keyfile = ssl_certfile = None

        # Configure workers for production
        if workers and workers > 1:
            uvicorn_config["workers"] = workers
            if self.agent.dev_mode:
                log_debug(f"Multi-process mode: {workers} workers")
        elif self.agent.dev_mode:
            log_debug("Single-process mode")

        # Performance optimizations for production
        if not dev:
            # Add production-specific configurations
            production_config = {
                "limit_concurrency": 1000,  # Limit concurrent connections
                "backlog": 2048,  # Connection queue size
            }

            # Only apply request limit for multi-worker deployments
            # Single-process apps don't benefit from this and it causes unwanted shutdowns
            if workers and workers > 1:
                production_config["limit_max_requests"] = (
                    100000  # Restart workers after N requests
                )

            uvicorn_config.update(production_config)

            # Try to use uvloop for better performance
            if importlib.util.find_spec("uvloop") is not None:
                uvicorn_config["loop"] = "uvloop"
                if self.agent.dev_mode:
                    log_info("Using uvloop for enhanced performance")
            elif self.agent.dev_mode:
                log_warn("uvloop not available, using default asyncio loop")

        # Environment-based log level adjustment
        env_log_level = os.getenv("UVICORN_LOG_LEVEL", log_level).lower()
        if env_log_level in ["critical", "error", "warning", "info", "debug", "trace"]:
            uvicorn_config["log_level"] = env_log_level

        # Disable access log in production if not explicitly enabled
        if not dev and "access_log" not in kwargs:
            uvicorn_config["access_log"] = False

        if self.agent.dev_mode:
            log_debug("Uvicorn configuration:")
            config_display = {
                k: v
                for k, v in uvicorn_config.items()
                if k not in ["ssl_keyfile", "ssl_certfile"]
            }
            for key, value in config_display.items():
                log_debug(f"  {key}: {value}")

        try:
            # Start FastAPI server with production-ready configuration
            uvicorn.run(self.agent, **uvicorn_config)
        except OSError as e:
            if "Address already in use" in str(e):
                log_error(
                    f"Port {port} is already in use. Choose a different port or stop the conflicting service."
                )
                if self.agent.dev_mode:
                    log_info(
                        "Try using auto_port=True or set a different port explicitly"
                    )
            else:
                log_error(f"Failed to start server: {e}")
            raise
        except KeyboardInterrupt:
            if self.agent.dev_mode:
                log_info("Server stopped by user (Ctrl+C)")
        except Exception as e:
            log_error(f"Unexpected server error: {e}")
            raise
        finally:
            # Phase 5: Graceful shutdown - stop heartbeat and MCP servers
            if self.agent.dev_mode:
                log_info("Agent shutdown initiated...")

            # Stop heartbeat worker
            self.agent.agentfield_handler.stop_heartbeat()

            # Stop all MCP servers
            self.agent.mcp_handler._cleanup_mcp_servers()

            if self.agent.dev_mode:
                log_success("Agent shutdown complete")
