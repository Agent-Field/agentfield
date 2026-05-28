"""
Agent definitions used to validate app.call behavior across nodes.
"""

from __future__ import annotations

import os
from typing import Optional

from agentfield import Agent

from agents import AgentSpec

WORKER_SPEC = AgentSpec(
    key="call_worker",
    display_name="Call Worker Agent",
    default_node_id="call-worker",
    description="Provides utility reasoners invoked via app.call.",
    reasoners=("uppercase_echo",),
    skills=(),
)

ORCHESTRATOR_SPEC = AgentSpec(
    key="call_orchestrator",
    display_name="Call Orchestrator Agent",
    default_node_id="call-orchestrator",
    description="Delegates to worker nodes using app.call.",
    reasoners=("delegate_pipeline",),
    skills=(),
)


def create_worker_agent(
    *,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    resolved_node_id = node_id or WORKER_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )

    agent = Agent(node_id=resolved_node_id, **agent_kwargs)

    @agent.reasoner(name="uppercase_echo")
    async def uppercase_echo(text: str) -> dict:
        normalized = text.strip()
        return {
            "text": normalized,
            "upper": normalized.upper(),
            "length": len(normalized),
        }

    return agent


def create_orchestrator_agent(
    *,
    target_node_id: str,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    resolved_node_id = node_id or ORCHESTRATOR_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )

    agent = Agent(node_id=resolved_node_id, **agent_kwargs)

    @agent.reasoner(name="delegate_pipeline")
    async def delegate_pipeline(text: str) -> dict:
        delegated = await agent.call(f"{target_node_id}.uppercase_echo", text=text)
        return {
            "original": text,
            "delegated": delegated,
            "tokens": len(text.split()),
        }

    return agent


__all__ = [
    "WORKER_SPEC",
    "ORCHESTRATOR_SPEC",
    "create_worker_agent",
    "create_orchestrator_agent",
]
