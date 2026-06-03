"""
Agent exposing reasoners via AgentRouter prefixes.
"""

from __future__ import annotations

import os
from typing import Optional

from agentfield import Agent, AgentRouter

from agents import AgentSpec

AGENT_SPEC = AgentSpec(
    key="router_prefix",
    display_name="Router Prefix Agent",
    default_node_id="router-prefix-agent",
    description="Validates router-prefixed reasoner registration and execution.",
    reasoners=("tools_echo", "tools_status"),
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

    tools_router = AgentRouter(prefix="tools")

    @tools_router.reasoner()
    async def echo(message: str) -> dict:
        return {"message": message, "length": len(message)}

    @tools_router.reasoner()
    async def status() -> dict:
        return {
            "node_id": agent.node_id,
            "router_prefix": "tools",
            "reasoners": sorted(r.get("id") for r in agent.reasoners),
        }

    agent.include_router(tools_router)
    return agent


__all__ = ["AGENT_SPEC", "create_agent"]
