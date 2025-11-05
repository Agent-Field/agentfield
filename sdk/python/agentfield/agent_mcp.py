import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentfield.agent_utils import AgentUtils
from agentfield.dynamic_skills import DynamicMCPSkillManager
from agentfield.execution_context import ExecutionContext
from agentfield.logger import log_debug, log_error, log_info, log_warn
from agentfield.mcp_client import MCPClientRegistry
from agentfield.mcp_manager import MCPManager
from agentfield.types import AgentStatus, MCPServerHealth
from fastapi import Request


class AgentMCP:
    """
    MCP Management handler for Agent class.

    This class encapsulates all MCP-related functionality including:
    - Agent directory detection
    - MCP server lifecycle management
    - MCP skill registration
    - Health monitoring
    """

    def __init__(self, agent_instance):
        """
        Initialize the MCP handler with a reference to the agent instance.

        Args:
            agent_instance: The Agent instance this handler belongs to
        """
        self.agent = agent_instance

    def _detect_agent_directory(self) -> str:
        """Detect the correct agent directory for MCP config discovery"""
        import os
        from pathlib import Path

        current_dir = Path(os.getcwd())

        # Check if packages/mcp exists in current directory
        if (current_dir / "packages" / "mcp").exists():
            return str(current_dir)

        # Look for agent directories in current directory
        for item in current_dir.iterdir():
            if item.is_dir() and (item / "packages" / "mcp").exists():
                if self.agent.dev_mode:
                    log_debug(f"Found agent directory: {item}")
                return str(item)

        # Look in parent directories (up to 3 levels)
        for i in range(3):
            parent = current_dir.parents[i] if i < len(current_dir.parents) else None
            if parent and (parent / "packages" / "mcp").exists():
                if self.agent.dev_mode:
                    log_debug(f"Found agent directory in parent: {parent}")
                return str(parent)

        # Fallback to current directory
        if self.agent.dev_mode:
            log_warn(
                f"No packages/mcp directory found, using current directory: {current_dir}"
            )
        return str(current_dir)

    async def initialize_mcp(self):
        """
        Initialize MCP management components.

        This method combines the MCP initialization logic that was previously
        scattered in the Agent.__init__ method.
        """
        try:
            agent_dir = self._detect_agent_directory()
            self.agent.mcp_manager = MCPManager(agent_dir, self.agent.dev_mode)
            self.agent.mcp_client_registry = MCPClientRegistry(self.agent.dev_mode)

            if self.agent.dev_mode:
                log_info(f"Initialized MCP Manager in {agent_dir}")

            # Initialize Dynamic Skill Manager when both MCP components are available
            if self.agent.mcp_manager and self.agent.mcp_client_registry:
                self.agent.dynamic_skill_manager = DynamicMCPSkillManager(
                    self.agent, self.agent.dev_mode
                )
                if self.agent.dev_mode:
                    log_info("Dynamic MCP skill manager initialized")

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Failed to initialize MCP Manager: {e}")
            self.agent.mcp_manager = None
            self.agent.mcp_client_registry = None
            self.agent.dynamic_skill_manager = None

    async def _start_mcp_servers(self) -> None:
        """Start all configured MCP servers using SimpleMCPManager."""
        if not self.agent.mcp_manager:
            if self.agent.dev_mode:
                log_info("No MCP Manager available - skipping server startup")
            return

        try:
            if self.agent.dev_mode:
                log_info("Starting MCP servers...")

            # Start all servers
            started_servers = await self.agent.mcp_manager.start_all_servers()

            if started_servers:
                successful = sum(1 for success in started_servers.values() if success)
                if self.agent.dev_mode:
                    log_info(f"Started {successful}/{len(started_servers)} MCP servers")
            elif self.agent.dev_mode:
                log_info("No MCP servers configured to start")

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"MCP server startup error: {e}")

    def _cleanup_mcp_servers(self) -> None:
        """
        Stop all MCP servers during agent shutdown.

        This method is called during graceful shutdown to ensure all
        MCP server processes are properly terminated.
        """
        if not self.agent.mcp_manager:
            if self.agent.dev_mode:
                log_info("No MCP Manager available - skipping cleanup")
            return

        async def async_cleanup():
            try:
                if self.agent.dev_mode:
                    log_info("Stopping MCP servers...")

                # Check if mcp_manager is still available
                if not self.agent.mcp_manager:
                    if self.agent.dev_mode:
                        log_info("MCP Manager not available during cleanup")
                    return

                # Get current server status before stopping
                all_status = self.agent.mcp_manager.get_all_status()

                if all_status:
                    running_servers = [
                        alias
                        for alias, health in all_status.items()
                        if health.get("status") == "running"
                    ]

                    if running_servers:
                        # Stop all running servers
                        for alias in running_servers:
                            try:
                                if (
                                    self.agent.mcp_manager
                                ):  # Double-check before each call
                                    await self.agent.mcp_manager.stop_server(alias)
                                    if self.agent.dev_mode:
                                        health = all_status.get(alias, {})
                                        pid = health.get("pid") or "N/A"
                                        log_info(
                                            f"Stopped MCP server: {alias} (PID: {pid})"
                                        )
                            except Exception as e:
                                if self.agent.dev_mode:
                                    log_error(f"Failed to stop MCP server {alias}: {e}")

                        if self.agent.dev_mode:
                            log_info(f"Stopped {len(running_servers)} MCP servers")
                    elif self.agent.dev_mode:
                        log_info("No running MCP servers to stop")
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Error during MCP server cleanup: {e}")
                # Continue with shutdown even if cleanup fails

        # Run the async cleanup properly
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a loop, create a task and store reference to prevent warning
                task = loop.create_task(async_cleanup())

                # Add a done callback to handle any exceptions and suppress warnings
                def handle_task_completion(t):
                    try:
                        if t.exception() is not None and self.agent.dev_mode:
                            log_error(f"MCP cleanup task failed: {t.exception()}")
                    except Exception:
                        # Suppress any callback exceptions to prevent warnings
                        pass

                task.add_done_callback(handle_task_completion)
                # Store task reference to prevent garbage collection warning
                if not hasattr(self, "_cleanup_tasks"):
                    self._cleanup_tasks = []
                self._cleanup_tasks.append(task)
            except RuntimeError:
                # No event loop running, we can use asyncio.run()
                try:
                    asyncio.run(async_cleanup())
                except Exception as cleanup_error:
                    if self.agent.dev_mode:
                        log_error(f"MCP cleanup failed: {cleanup_error}")
        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Failed to run MCP cleanup: {e}")

    def _register_mcp_server_skills(self) -> None:
        """
        DEPRECATED: This method is replaced by DynamicMCPSkillManager.
        The static file-based approach is broken after SimpleMCPManager refactor.
        """
        if self.agent.dev_mode:
            log_warn("DEPRECATED: _register_mcp_server_skills() is no longer used")
        return

    def _register_mcp_tool_as_skill(
        self, server_alias: str, tool: Dict[str, Any]
    ) -> None:
        """
        Register an MCP tool as a proper FastAPI skill endpoint.

        Args:
            server_alias: The alias of the MCP server
            tool: Tool definition from mcp.json
        """
        tool_name = tool.get("name", "")
        if not tool_name:
            if self.agent.dev_mode:
                log_warn(f"Skipping tool with missing name: {tool}")
            return

        skill_name = f"{server_alias}_{tool_name}"
        endpoint_path = f"/skills/{skill_name}"

        # Create a simple input schema - use dict for flexibility
        from pydantic import BaseModel

        class InputSchema(BaseModel):
            """Dynamic input schema for MCP tool"""

            args: dict = {}

            class Config:
                extra = "allow"  # Allow additional fields

        # Create the MCP skill function
        async def mcp_skill_function(**kwargs):
            """Dynamically created MCP skill function"""
            if self.agent.dev_mode:
                log_debug(
                    f"MCP skill called: {server_alias}.{tool_name} with args: {kwargs}"
                )

            try:
                # Get process-aware MCP client (reuses existing running processes)
                if not self.agent.mcp_client_registry:
                    raise Exception("MCPClientRegistry not initialized")
                mcp_client = self.agent.mcp_client_registry.get_client(server_alias)
                if not mcp_client:
                    raise Exception(f"MCP client for {server_alias} not found")

                # Call the MCP tool using existing process
                result = await mcp_client.call_tool(tool_name, kwargs)

                return {
                    "status": "success",
                    "result": result,
                    "server": server_alias,
                    "tool": tool_name,
                }

            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"MCP skill error: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "server": server_alias,
                    "tool": tool_name,
                    "args": kwargs,
                }

        # Create FastAPI endpoint
        @self.agent.post(endpoint_path, response_model=dict)
        async def mcp_skill_endpoint(input_data: InputSchema, request: Request):
            from agentfield.execution_context import ExecutionContext

            # Extract execution context from request headers
            execution_context = ExecutionContext.from_request(
                request, self.agent.node_id
            )

            # Store current context for use in app.call()
            self.agent._current_execution_context = execution_context

            # Convert input to function arguments
            kwargs = input_data.args

            # Call the MCP skill function
            result = await mcp_skill_function(**kwargs)

            return result

        # Register skill metadata
        self.agent.skills.append(
            {
                "id": skill_name,
                "input_schema": InputSchema.model_json_schema(),
                "tags": ["mcp", server_alias],
                "description": tool.get("description", f"MCP tool: {tool_name}"),
            }
        )

    def _create_and_register_mcp_skill(
        self, server_alias: str, tool: Dict[str, Any]
    ) -> None:
        """
        Create and register a single MCP tool as a AgentField skill.

        Args:
            server_alias: The alias of the MCP server
            tool: Tool definition from mcp.json
        """
        tool_name = tool.get("name", "")
        if not tool_name:
            raise ValueError("Tool missing 'name' field")

        # Generate skill function name: server_alias + tool_name
        skill_name = AgentUtils.generate_skill_name(server_alias, tool_name)

        # Create the skill function dynamically
        async def mcp_skill_function(
            execution_context: Optional[ExecutionContext] = None, **kwargs
        ) -> Any:
            """
            Auto-generated MCP skill function.

            This function calls the corresponding MCP tool and returns the result.
            """
            try:
                # Get MCP client
                if not self.agent.mcp_client_registry:
                    raise Exception("MCPClientRegistry not initialized")
                client = self.agent.mcp_client_registry.get_client(server_alias)
                if not client:
                    raise Exception(f"MCP client for {server_alias} not found")

                # Call the MCP tool
                result = await client.call_tool(tool_name, kwargs)
                return result

            except Exception as e:
                # Re-raise with helpful context
                raise Exception(
                    f"MCP tool '{server_alias}.{tool_name}' failed: {str(e)}"
                ) from e

        # Set function metadata
        mcp_skill_function.__name__ = skill_name
        mcp_skill_function.__doc__ = f"""
        {tool.get("description", f"MCP tool: {tool_name}")}
        
        This is an auto-generated skill that wraps the '{tool_name}' tool from the '{server_alias}' MCP server.
        
        Args:
            execution_context (ExecutionContext, optional): AgentField execution context for workflow tracking
            **kwargs: Arguments to pass to the MCP tool
            
        Returns:
            Any: The result from the MCP tool execution
            
        Raises:
            Exception: If the MCP server is unavailable or the tool execution fails
        """

        # Create input schema from tool's input schema
        input_schema = AgentUtils.create_input_schema_from_mcp_tool(skill_name, tool)

        # Create FastAPI endpoint
        endpoint_path = f"/skills/{skill_name}"

        @self.agent.post(endpoint_path, response_model=dict)
        async def mcp_skill_endpoint(input_data: Dict[str, Any], request: Request):
            # Extract execution context from request headers
            execution_context = ExecutionContext.from_request(
                request, self.agent.node_id
            )

            # Store current context for use in app.call()
            self.agent._current_execution_context = execution_context

            # Convert input to function arguments
            kwargs = input_data

            # Call the MCP skill function
            result = await mcp_skill_function(
                execution_context=execution_context, **kwargs
            )
            return result

        # Register skill metadata
        self.agent.skills.append(
            {
                "id": skill_name,
                "input_schema": input_schema.model_json_schema(),
                "tags": ["mcp", server_alias],
            }
        )

    def _get_mcp_server_health(self) -> List[MCPServerHealth]:
        """
        Get health information for all MCP servers.

        Returns:
            List of MCPServerHealth objects
        """
        mcp_servers = []

        if self.agent.mcp_manager:
            try:
                all_status = self.agent.mcp_manager.get_all_status()

                for alias, server_info in all_status.items():
                    server_health = MCPServerHealth(
                        alias=alias,
                        status=server_info.get("status", "unknown"),
                        tool_count=0,
                        port=server_info.get("port"),
                        process_id=(
                            server_info.get("process", {}).get("pid")
                            if server_info.get("process")
                            else None
                        ),
                        started_at=datetime.now().isoformat(),
                        last_health_check=datetime.now().isoformat(),
                    )

                    # Try to get tool count if server is running
                    if (
                        server_health.status == "running"
                        and self.agent.mcp_client_registry
                    ):
                        try:
                            client = self.agent.mcp_client_registry.get_client(alias)
                            if client:
                                # This would need to be implemented properly
                                server_health.tool_count = 0  # Placeholder
                        except Exception:
                            pass

                    mcp_servers.append(server_health)

            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Error getting MCP server health: {e}")

        return mcp_servers

    async def _background_mcp_initialization(self) -> None:
        """
        Initialize MCP servers in the background after registration.
        """
        try:
            if self.agent.dev_mode:
                log_info("Background MCP initialization started")

            # Start MCP servers
            if self.agent.mcp_manager:
                results = await self.agent.mcp_manager.start_all_servers()

                # Register clients for successfully started servers
                for alias, success in results.items():
                    if success and self.agent.mcp_client_registry:
                        server_status = self.agent.mcp_manager.get_server_status(alias)
                        if server_status and server_status.get("port"):
                            self.agent.mcp_client_registry.register_client(
                                alias, server_status["port"]
                            )

                successful = sum(1 for success in results.values() if success)
                total = len(results)

                if self.agent.dev_mode:
                    log_info(
                        f"MCP initialization: {successful}/{total} servers started"
                    )

                # Update status based on MCP results
                if successful == total and total > 0:
                    self.agent._current_status = AgentStatus.READY
                elif successful > 0:
                    self.agent._current_status = AgentStatus.DEGRADED
                else:
                    self.agent._current_status = (
                        AgentStatus.READY
                    )  # Still ready even without MCP
            else:
                # No MCP manager, agent is ready
                self.agent._current_status = AgentStatus.READY
                if self.agent.dev_mode:
                    log_info("No MCP servers to initialize - agent ready")

            # Register dynamic skills if available
            if self.agent.dynamic_skill_manager:
                if self.agent.dev_mode:
                    log_info("Registering MCP tools as skills...")
                await (
                    self.agent.dynamic_skill_manager.discover_and_register_all_skills()
                )

            self.agent._mcp_initialization_complete = True

            # Send status update heartbeat
            await self.agent.agentfield_handler.send_enhanced_heartbeat()

            if self.agent.dev_mode:
                log_info(
                    f"Background initialization complete - Status: {self.agent._current_status.value}"
                )

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Background MCP initialization error: {e}")
            self.agent._current_status = AgentStatus.DEGRADED
            await self.agent.agentfield_handler.send_enhanced_heartbeat()
