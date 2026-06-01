"""
Agent used for validating app.memory behaviors across scopes.
"""

from __future__ import annotations

import os
from typing import Optional

from agentfield import Agent
from agentfield.execution_context import ExecutionContext

from agents import AgentSpec

AGENT_SPEC = AgentSpec(
    key="memory_validation",
    display_name="Memory Validation Agent",
    default_node_id="memory-agent",
    description="Exercises session, actor, and global memory scopes.",
    reasoners=("remember_user",),
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

    agent = Agent(
        node_id=resolved_node_id,
        **agent_kwargs,
    )

    @agent.reasoner(name="remember_user")
    async def remember_user(
        user_id: str,
        message: str,
        actor_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
    ) -> dict:
        """
        Persist user message history using app.memory scoped helpers.
        """
        if execution_context and not execution_context.session_id:
            execution_context.session_id = f"session::{user_id}"

        session_scope = agent.memory.session(user_id)
        history = await session_scope.get("history", default=[])
        history.append(message)
        await session_scope.set("history", history)

        global_scope = agent.memory.global_scope
        global_key = f"user::{user_id}::count"
        global_count = int(await global_scope.get(global_key, default=0) or 0) + 1
        await global_scope.set(global_key, global_count)

        key_exists = await global_scope.exists(global_key)
        recent = history[-5:]
        return {
            "user_id": user_id,
            "messages_seen": global_count,
            "recent_history": recent,
            "global_key_exists": key_exists,
        }

    return agent


__all__ = ["AGENT_SPEC", "create_agent"]
