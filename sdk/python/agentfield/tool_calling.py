"""
Tool calling support for AgentField agents.

Converts discovered capabilities into LLM-native tool schemas and provides
an automatic tool-call execution loop that dispatches calls via app.call().
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from agentfield.logger import log_debug, log_error, log_warn
from agentfield.types import (
    AgentCapability,
    DiscoveryResponse,
    ReasonerCapability,
    SkillCapability,
)

if TYPE_CHECKING:
    from agentfield.agent import Agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ToolCallConfig:
    """Configuration for the tool-call loop."""

    max_turns: int = 10
    max_tool_calls: int = 25
    max_candidate_tools: Optional[int] = None
    max_hydrated_tools: Optional[int] = None
    schema_hydration: Literal["eager", "lazy"] = "eager"
    fallback_broadening: bool = False
    tags: Optional[List[str]] = None
    agent_ids: Optional[List[str]] = None
    health_status: Optional[str] = None


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """Record of a single tool call for observability."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    turn: int = 0


@dataclass
class ToolCallTrace:
    """Full trace of a tool-call loop execution."""

    calls: List[ToolCallRecord] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    final_response: Optional[str] = None
    hydration_retries: int = 0


class ToolCallResponse:
    """Typed wrapper for AI responses that went through the tool-calling loop.

    Provides direct access to the trace and delegates attribute access
    to the underlying LLM response for backward compatibility.

    Example:
        result = await app.ai("Help the user", tools="discover")
        print(result.text)              # final response text
        print(result.trace)             # ToolCallTrace with full observability
        print(result.trace.total_turns) # how many LLM round-trips
        print(result.trace.calls)       # list of ToolCallRecord
        # All original response attributes still accessible:
        print(result.choices)           # delegates to underlying response
    """

    def __init__(self, response: Any, trace: ToolCallTrace):
        self._response = response
        self.trace = trace

    @property
    def text(self) -> Optional[str]:
        """Final text response from the LLM."""
        return self.trace.final_response

    @property
    def response(self) -> Any:
        """The underlying LLM response object."""
        return self._response

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying response for backward compat."""
        return getattr(self._response, name)

    def __repr__(self) -> str:
        return (
            f"ToolCallResponse(turns={self.trace.total_turns}, "
            f"tool_calls={self.trace.total_tool_calls}, "
            f"text={self.text!r:.80})"
        )


# ---------------------------------------------------------------------------
# Target format conversion
# ---------------------------------------------------------------------------


def _invocation_target_to_call_target(invocation_target: str) -> str:
    """Convert discovery invocation_target format to agent.call() target format.

    Discovery returns colon-separated targets:
      - "node_id:skill:function_name" for skills
      - "node_id:function_name" for reasoners

    agent.call() expects dot-separated:
      - "node_id.function_name" for both
    """
    # Handle skill format: "node_id:skill:function_name" -> "node_id.function_name"
    if ":skill:" in invocation_target:
        parts = invocation_target.split(":skill:")
        return f"{parts[0]}.{parts[1]}"
    # Handle reasoner format: "node_id:function_name" -> "node_id.function_name"
    if ":" in invocation_target:
        parts = invocation_target.split(":", 1)
        return f"{parts[0]}.{parts[1]}"
    return invocation_target


def _sanitize_tool_name(invocation_target: str) -> str:
    """Convert an invocation_target to an LLM-safe function name.

    Many LLM providers (e.g., Google) only allow alphanumeric, underscores,
    dashes, and dots in function names. Colons are not allowed.

    We replace colons with double-underscores for a reversible mapping:
      "utility-worker:skill:get_weather" -> "utility-worker__skill__get_weather"
      "utility-worker:summarize"         -> "utility-worker__summarize"
    """
    return invocation_target.replace(":", "__")


def _unsanitize_tool_name(sanitized_name: str) -> str:
    """Reverse _sanitize_tool_name: double-underscores back to colons."""
    return sanitized_name.replace("__", ":")


# ---------------------------------------------------------------------------
# Capability -> Tool Schema Conversion
# ---------------------------------------------------------------------------


def capability_to_tool_schema(
    cap: Union[ReasonerCapability, SkillCapability],
) -> Dict[str, Any]:
    """Convert a ReasonerCapability or SkillCapability to an OpenAI-format tool schema.

    LiteLLM normalizes this format across providers.
    """
    parameters = cap.input_schema or {"type": "object", "properties": {}}

    # Ensure parameters has required top-level fields
    if "type" not in parameters:
        parameters = {"type": "object", "properties": parameters}

    return {
        "type": "function",
        "function": {
            "name": _sanitize_tool_name(cap.invocation_target),
            "description": cap.description or f"Call {cap.invocation_target}",
            "parameters": parameters,
        },
    }


def capabilities_to_tool_schemas(
    capabilities: Sequence[Union[ReasonerCapability, SkillCapability, AgentCapability]],
) -> List[Dict[str, Any]]:
    """Convert a list of capabilities into LLM-native tool schemas.

    Accepts individual ReasonerCapability/SkillCapability objects, or
    AgentCapability objects (which will have their reasoners and skills extracted).
    """
    tools: List[Dict[str, Any]] = []
    for cap in capabilities:
        if isinstance(cap, AgentCapability):
            for r in cap.reasoners:
                tools.append(capability_to_tool_schema(r))
            for s in cap.skills:
                tools.append(capability_to_tool_schema(s))
        elif isinstance(cap, (ReasonerCapability, SkillCapability)):
            tools.append(capability_to_tool_schema(cap))
    return tools


def capabilities_to_metadata_only(
    capabilities: Sequence[Union[ReasonerCapability, SkillCapability, AgentCapability]],
) -> List[Dict[str, Any]]:
    """Convert capabilities to metadata-only tool schemas (no full input_schema).

    Used for progressive discovery: first pass sends just name/description/tags
    so the LLM can select which tools it needs before hydrating full schemas.
    """
    tools: List[Dict[str, Any]] = []

    def _metadata(cap: Union[ReasonerCapability, SkillCapability]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": _sanitize_tool_name(cap.invocation_target),
                "description": cap.description or f"Call {cap.invocation_target}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    for cap in capabilities:
        if isinstance(cap, AgentCapability):
            for r in cap.reasoners:
                tools.append(_metadata(r))
            for s in cap.skills:
                tools.append(_metadata(s))
        elif isinstance(cap, (ReasonerCapability, SkillCapability)):
            tools.append(_metadata(cap))
    return tools


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _discover_tools(
    agent: "Agent",
    config: ToolCallConfig,
    hydrate_schemas: bool = True,
) -> List[Dict[str, Any]]:
    """Discover available tools from the control plane.

    Args:
        agent: The Agent instance to discover from.
        config: Tool call configuration with filtering options.
        hydrate_schemas: If True, include full input schemas. If False, metadata only.
    """
    discovery_result = agent.discover(
        tags=config.tags,
        agent_ids=config.agent_ids,
        include_input_schema=hydrate_schemas,
        include_output_schema=False,
        include_descriptions=True,
        health_status=config.health_status,
    )

    if discovery_result.json is None:
        return []

    all_caps = discovery_result.json.capabilities

    if hydrate_schemas:
        tools = capabilities_to_tool_schemas(all_caps)
    else:
        tools = capabilities_to_metadata_only(all_caps)

    if config.max_candidate_tools and len(tools) > config.max_candidate_tools:
        tools = tools[: config.max_candidate_tools]

    return tools


def _hydrate_selected_tools(
    agent: "Agent",
    config: ToolCallConfig,
    selected_names: List[str],
) -> List[Dict[str, Any]]:
    """Re-discover with full schemas for only the selected tool names."""
    discovery_result = agent.discover(
        tags=config.tags,
        agent_ids=config.agent_ids,
        include_input_schema=True,
        include_output_schema=False,
        include_descriptions=True,
        health_status=config.health_status,
    )

    if discovery_result.json is None:
        return []

    # selected_names are sanitized (from LLM), so unsanitize for matching
    selected_set = set(_unsanitize_tool_name(n) for n in selected_names)
    tools: List[Dict[str, Any]] = []
    for cap in discovery_result.json.capabilities:
        for r in cap.reasoners:
            if r.invocation_target in selected_set:
                tools.append(capability_to_tool_schema(r))
        for s in cap.skills:
            if s.invocation_target in selected_set:
                tools.append(capability_to_tool_schema(s))

    limit = config.max_hydrated_tools
    if limit and len(tools) > limit:
        tools = tools[:limit]

    return tools


# ---------------------------------------------------------------------------
# Tool-call execution loop
# ---------------------------------------------------------------------------


def _build_tool_config(
    tools_param: Any,
    agent: "Agent",
) -> tuple[List[Dict[str, Any]], ToolCallConfig, bool]:
    """Parse the `tools=` parameter into (tool_schemas, config, needs_lazy_hydration).

    Supported values for tools_param:
    - "discover": auto-discover all tools from control plane
    - DiscoveryResponse: use already-fetched discovery result
    - list of AgentCapability/ReasonerCapability/SkillCapability: convert directly
    - list of dicts: assumed to be raw OpenAI tool schemas
    - ToolCallConfig: discover with configuration
    - dict: treat as ToolCallConfig kwargs
    """
    config = ToolCallConfig()
    needs_lazy = False

    if isinstance(tools_param, str) and tools_param == "discover":
        tools = _discover_tools(agent, config)
        return tools, config, False

    if isinstance(tools_param, ToolCallConfig):
        config = tools_param
        if config.schema_hydration == "lazy":
            tools = _discover_tools(agent, config, hydrate_schemas=False)
            needs_lazy = True
        else:
            tools = _discover_tools(agent, config)
        return tools, config, needs_lazy

    if isinstance(tools_param, dict):
        config = ToolCallConfig(**tools_param)
        if config.schema_hydration == "lazy":
            tools = _discover_tools(agent, config, hydrate_schemas=False)
            needs_lazy = True
        else:
            tools = _discover_tools(agent, config)
        return tools, config, needs_lazy

    if isinstance(tools_param, DiscoveryResponse):
        tools = capabilities_to_tool_schemas(tools_param.capabilities)
        return tools, config, False

    if isinstance(tools_param, list):
        if not tools_param:
            return [], config, False
        first = tools_param[0]
        if isinstance(first, dict):
            # Already raw tool schemas
            return tools_param, config, False
        # List of capability objects
        tools = capabilities_to_tool_schemas(tools_param)
        return tools, config, False

    raise ValueError(
        f"Invalid tools parameter: expected 'discover', ToolCallConfig, dict, "
        f"DiscoveryResponse, or list of capabilities/schemas, got {type(tools_param)}"
    )


async def execute_tool_call_loop(
    agent: "Agent",
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    config: ToolCallConfig,
    needs_lazy_hydration: bool,
    litellm_params: Dict[str, Any],
    make_completion: Callable,
) -> tuple[Any, ToolCallTrace]:
    """Execute the LLM tool-call loop.

    Sends messages + tools to the LLM, dispatches any tool calls via app.call(),
    feeds results back, and repeats until the LLM produces a final text response
    or limits are reached.

    Args:
        agent: The Agent instance for dispatching calls.
        messages: The conversation messages.
        tools: LLM tool schemas.
        config: Tool call configuration.
        needs_lazy_hydration: Whether to hydrate schemas on first tool selection.
        litellm_params: Base LiteLLM parameters.
        make_completion: Async callable that takes (params) and returns LLM response.

    Returns:
        Tuple of (final_response, trace).
    """
    trace = ToolCallTrace()
    total_calls = 0
    hydrated = not needs_lazy_hydration
    _timeout_break = False

    for turn in range(config.max_turns):
        trace.total_turns = turn + 1

        # Build params for this turn
        params = {**litellm_params}
        params["messages"] = messages
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        resp = await make_completion(params)

        response_message = resp.choices[0].message

        # Check if the LLM wants to call tools
        tool_calls = getattr(response_message, "tool_calls", None)

        if not tool_calls:
            # No tool calls - LLM has produced a final response
            trace.final_response = getattr(response_message, "content", None)
            return resp, trace

        # If lazy hydration and this is the first tool selection, hydrate and retry
        if not hydrated and tool_calls:
            selected_names = [tc.function.name for tc in tool_calls]
            log_debug(
                f"Lazy hydration: LLM selected {len(selected_names)} tools, "
                f"hydrating schemas..."
            )
            tools = _hydrate_selected_tools(agent, config, selected_names)
            hydrated = True
            trace.hydration_retries += 1
            # Re-run this turn with hydrated schemas (don't count as a tool call
            # but DO consume a turn to prevent infinite loops)
            continue

        # Append assistant message with tool calls
        messages.append(response_message.model_dump())

        # Execute each tool call
        _timeout_break = False
        for tc in tool_calls:
            if total_calls >= config.max_tool_calls:
                log_warn(
                    f"Tool call limit reached ({config.max_tool_calls}), "
                    f"stopping tool execution"
                )
                # Add a message telling the LLM about the limit
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "error": "Tool call limit reached. Please provide a final response."
                            }
                        ),
                    }
                )
                continue

            total_calls += 1
            trace.total_tool_calls = total_calls

            func_name = tc.function.name
            # Unsanitize the LLM-safe name back to the original invocation_target
            invocation_target = _unsanitize_tool_name(func_name)
            raw_args = getattr(tc.function, "arguments", None)
            if raw_args is None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "error": f"Tool call to '{func_name}' is missing the 'arguments' field. Please retry with valid JSON arguments."
                            }
                        ),
                    }
                )
                continue
            try:
                func_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                func_args = {}

            record = ToolCallRecord(
                tool_name=func_name,
                arguments=func_args,
                turn=turn,
            )

            # Convert invocation_target format to agent.call() format
            call_target = _invocation_target_to_call_target(invocation_target)
            log_debug(f"Tool call [{total_calls}]: {func_name} -> {call_target}({json.dumps(func_args)})")

            start_time = time.monotonic()
            try:
                result = await agent.call(call_target, **func_args)
                record.result = result
                record.latency_ms = (time.monotonic() - start_time) * 1000

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

                log_debug(
                    f"Tool result [{total_calls}]: {func_name} "
                    f"completed in {record.latency_ms:.0f}ms"
                )

            except asyncio.TimeoutError as e:
                record.error = f"TimeoutError: {e}"
                record.latency_ms = (time.monotonic() - start_time) * 1000

                log_error(f"Tool call timed out: {func_name} - {e}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "error": f"Tool execution timed out: {e}",
                                "tool": func_name,
                            }
                        ),
                    }
                )

                trace.calls.append(record)
                _timeout_break = True
                break

            except Exception as e:
                record.error = str(e)
                record.latency_ms = (time.monotonic() - start_time) * 1000

                log_error(f"Tool call failed: {func_name} - {e}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": str(e), "tool": func_name}),
                    }
                )

            trace.calls.append(record)

        if _timeout_break:
            break

        # Check if we've hit the tool call limit
        if total_calls >= config.max_tool_calls:
            # Make one final call without tools to get a response
            final_params = {**litellm_params}
            final_params["messages"] = messages
            # Don't pass tools - force the LLM to respond with text
            resp = await make_completion(final_params)
            trace.final_response = getattr(resp.choices[0].message, "content", None)
            return resp, trace

    if _timeout_break:
        return resp, trace

    # Max turns reached - make a final call without tools
    log_warn(f"Max turns reached ({config.max_turns}), requesting final response")
    final_params = {**litellm_params}
    final_params["messages"] = messages
    resp = await make_completion(final_params)
    trace.final_response = getattr(resp.choices[0].message, "content", None)
    trace.total_turns = config.max_turns
    return resp, trace
