"""Agent dedicated to validating scoped memory behavior and header overrides."""

from __future__ import annotations

import os
from typing import Optional

from agentfield import Agent
from agentfield.execution_context import ExecutionContext

from agents import AgentSpec

AGENT_SPEC = AgentSpec(
    key="scoping_validation",
    display_name="Memory Scoping Validation Agent",
    default_node_id="scoping-agent",
    description="Provides reasoners to verify workflow/session/actor scope helpers.",
    reasoners=("scoped_memory",),
    skills=(),
)


def create_agent(
    *,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    resolved_node_id = node_id or AGENT_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )

    agent = Agent(node_id=resolved_node_id, **agent_kwargs)

    def _scope_client(scope: str, scope_id: Optional[str]):
        normalized = scope.lower()
        if normalized == "global":
            return agent.memory.global_scope
        if not scope_id:
            raise ValueError(f"scope_id is required for scope '{scope}'")
        if normalized == "session":
            return agent.memory.session(scope_id)
        if normalized == "actor":
            return agent.memory.actor(scope_id)
        if normalized == "workflow":
            return agent.memory.workflow(scope_id)
        raise ValueError(f"Unsupported scope '{scope}'")

    @agent.reasoner(name="scoped_memory")
    async def scoped_memory(
        scope: str,
        scope_id: Optional[str],
        key: str,
        action: str = "write",
        value: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
    ) -> dict:
        """Set/read/delete values in the requested scope and report what was observed."""

        client = _scope_client(scope, scope_id)

        normalized_action = (action or "").lower()
        if normalized_action == "write":
            if value is None:
                raise ValueError("value is required when action='write'")
            await client.set(key, value)
        elif normalized_action == "delete":
            await client.delete(key)
        elif normalized_action != "read":
            raise ValueError(f"Unsupported action '{action}'")

        scoped_value = await client.get(key, default=None)
        exists = await client.exists(key)

        try:
            listed_keys = await client.list_keys()
        except AttributeError:
            listed_keys = None

        hierarchy_value = await agent.memory.get(key, default=None)

        context_snapshot = {
            "workflow_id": execution_context.workflow_id if execution_context else None,
            "session_id": execution_context.session_id if execution_context else None,
            "actor_id": execution_context.actor_id if execution_context else None,
        }

        return {
            "scope": scope,
            "scope_id": scope_id,
            "action": normalized_action,
            "scoped_value": scoped_value,
            "exists": exists,
            "keys": listed_keys,
            "hierarchy_value": hierarchy_value,
            "execution_context": context_snapshot,
        }

    return agent


__all__ = ["AGENT_SPEC", "create_agent"]
