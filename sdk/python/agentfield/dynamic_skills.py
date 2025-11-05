import asyncio
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, create_model
from fastapi import Request

from agentfield.agent_utils import AgentUtils
from agentfield.execution_context import ExecutionContext
from agentfield.logger import log_debug, log_error, log_info, log_warn


class DynamicMCPSkillManager:
    """
    Dynamic MCP Skill Generator that converts MCP tools into AgentField skills.

    This class discovers MCP servers, lists their tools, and dynamically
    registers each tool as a AgentField skill with proper schema generation
    and execution context handling.
    """

    def __init__(self, agent, dev_mode: bool = False):
        """
        Initialize the Dynamic MCP Skill Manager.

        Args:
            agent: The AgentField agent instance
            dev_mode: Enable development mode logging
        """
        self.agent = agent
        self.dev_mode = dev_mode
        self.registered_skills: Dict[str, Dict] = {}

    async def discover_and_register_all_skills(self) -> None:
        """
        Discover and register all MCP tools as AgentField skills.

        This method:
        1. Checks for MCP client registry availability
        2. Iterates through all connected MCP servers
        3. Waits for server readiness
        4. Performs health checks on each server
        5. Lists tools from healthy servers
        6. Registers each tool as a AgentField skill
        """
        if not self.agent.mcp_client_registry:
            if self.dev_mode:
                log_warn("MCP client registry not available")
            return

        if self.dev_mode:
            log_info("Starting MCP skill discovery...")

        # Get all registered MCP clients
        clients = self.agent.mcp_client_registry.clients

        if not clients:
            if self.dev_mode:
                log_info("No MCP servers found in registry")
            return

        # Wait for server readiness
        await asyncio.sleep(1)

        for server_alias, client in clients.items():
            try:
                if self.dev_mode:
                    log_debug(f"Processing MCP server: {server_alias}")

                # Perform health check
                is_healthy = await client.health_check()
                if not is_healthy:
                    if self.dev_mode:
                        log_warn(
                            f"MCP server {server_alias} failed health check, skipping"
                        )
                    continue

                # List tools from the server
                tools = await client.list_tools()
                if not tools:
                    if self.dev_mode:
                        log_info(f"No tools found in MCP server {server_alias}")
                    continue

                if self.dev_mode:
                    log_debug(f"Found {len(tools)} tools in {server_alias}")

                # Register each tool as a skill
                for tool in tools:
                    try:
                        skill_name = AgentUtils.generate_skill_name(
                            server_alias, tool.get("name", "")
                        )
                        await self._register_mcp_tool_as_skill(
                            server_alias, tool, skill_name
                        )

                        if self.dev_mode:
                            log_info(f"Registered skill: {skill_name}")

                    except Exception as e:
                        if self.dev_mode:
                            log_error(
                                f"Failed to register tool {tool.get('name', 'unknown')} from {server_alias}: {e}"
                            )
                        continue

            except Exception as e:
                if self.dev_mode:
                    log_error(f"Error processing MCP server {server_alias}: {e}")
                continue

        if self.dev_mode:
            log_info(
                f"MCP skill discovery complete. Registered {len(self.registered_skills)} skills"
            )

    async def _register_mcp_tool_as_skill(
        self, server_alias: str, tool: Dict[str, Any], skill_name: str
    ) -> None:
        """
        Register an MCP tool as a AgentField skill.

        This method:
        1. Extracts tool metadata (name, description)
        2. Generates Pydantic input schema from tool definition
        3. Creates async wrapper function for MCP tool calls
        4. Sets function metadata
        5. Creates FastAPI endpoint
        6. Handles execution context from request headers
        7. Stores and clears execution context appropriately
        8. Registers skill metadata with agent
        9. Adds to internal skill registry

        Args:
            server_alias: MCP server alias
            tool: Tool definition from MCP server
            skill_name: Generated skill name
        """
        tool_name = tool.get("name", "")
        description = tool.get(
            "description", f"MCP tool {tool_name} from {server_alias}"
        )

        # Generate Pydantic input schema
        input_schema = self._create_input_schema_from_tool(skill_name, tool)

        # Create async wrapper function for MCP tool calls
        async def mcp_skill_wrapper(**kwargs):
            """Dynamically created MCP skill function"""
            try:
                # Get MCP client for this server
                client = self.agent.mcp_client_registry.get_client(server_alias)
                if not client:
                    return {
                        "status": "error",
                        "error": f"MCP client for server '{server_alias}' not available",
                        "server": server_alias,
                        "tool": tool_name,
                        "args": kwargs,
                    }

                # Call the MCP tool
                result = await client.call_tool(tool_name, kwargs)

                return {
                    "status": "success",
                    "result": result,
                    "server": server_alias,
                    "tool": tool_name,
                }

            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "server": server_alias,
                    "tool": tool_name,
                    "args": kwargs,
                }

        # Set function metadata
        mcp_skill_wrapper.__name__ = skill_name
        mcp_skill_wrapper.__doc__ = description

        # Create FastAPI endpoint
        endpoint_path = f"/skills/{skill_name}"

        # Create the endpoint function dynamically
        async def mcp_skill_endpoint(input_data: Any, request: Request):
            """Dynamically created MCP skill endpoint"""
            # Validate input data against the schema
            validated_data = (
                input_schema(**input_data)
                if isinstance(input_data, dict)
                else input_data
            )

            # Handle execution context from request headers
            execution_context = ExecutionContext.from_request(
                request, self.agent.node_id
            )

            # Store execution context in agent
            self.agent._current_execution_context = execution_context

            try:
                # Convert input to function arguments
                if hasattr(validated_data, "dict"):
                    kwargs = validated_data.model_dump()
                elif isinstance(validated_data, dict):
                    kwargs = validated_data
                else:
                    kwargs = {}

                # Call the MCP skill wrapper
                result = await mcp_skill_wrapper(**kwargs)

                return result

            finally:
                # Clear execution context after completion
                self.agent._current_execution_context = None

        # Set the correct parameter annotation for FastAPI
        mcp_skill_endpoint.__annotations__ = {
            "input_data": input_schema,
            "request": Request,
            "return": dict,
        }

        # Register the endpoint
        self.agent.post(endpoint_path, response_model=dict)(mcp_skill_endpoint)

        # Register skill metadata with agent
        skill_metadata = {
            "id": skill_name,
            "input_schema": input_schema.model_json_schema(),
            "tags": ["mcp", server_alias],
            "description": description,
            "server_alias": server_alias,
            "tool_name": tool_name,
        }

        self.agent.skills.append(skill_metadata)

        # Add to internal skill registry
        self.registered_skills[skill_name] = skill_metadata

    def _create_input_schema_from_tool(
        self, skill_name: str, tool: Dict[str, Any]
    ) -> Type[BaseModel]:
        """
        Create Pydantic input schema from MCP tool definition.

        Schema Generation Rules:
        - Extract inputSchema.properties and required fields
        - Map JSON Schema types to Python types
        - Handle required vs optional fields appropriately
        - Set default values when specified
        - Use Optional[Type] for non-required fields without defaults
        - Fallback to generic {"data": Optional[Dict[str, Any]]} if no properties
        - Create model with name pattern: {skill_name}Input

        Args:
            skill_name: Name of the skill
            tool: Tool definition from MCP server

        Returns:
            Pydantic BaseModel class for input validation
        """
        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_fields = set(input_schema.get("required", []))

        # If no properties defined, use generic schema
        if not properties:
            return create_model(
                f"{skill_name}Input", data=(Optional[Dict[str, Any]], None)
            )

        # Build field definitions for Pydantic model
        field_definitions = {}

        for field_name, field_def in properties.items():
            field_type = AgentUtils.map_json_type_to_python(
                field_def.get("type", "string")
            )
            default_value = field_def.get("default")
            is_required = field_name in required_fields

            if is_required and default_value is None:
                # Required field without default
                field_definitions[field_name] = (field_type, ...)
            elif default_value is not None:
                # Field with default value
                field_definitions[field_name] = (field_type, default_value)
            else:
                # Optional field without default
                field_definitions[field_name] = (Optional[field_type], None)

        # Create and return the Pydantic model
        model_name = f"{skill_name}Input"
        return create_model(model_name, **field_definitions)
