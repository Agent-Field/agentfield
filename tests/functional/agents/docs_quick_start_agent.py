"""
Agent definition that mirrors the documentation Quick Start (demo_echo router).

This matches the `af init my-agent --defaults` experience described in
`/docs/quick-start` by exposing the router-prefixed `demo_echo` reasoner that
works without any AI providers configured.
"""

from __future__ import annotations

import os
from typing import Optional

from agentfield import Agent, AgentRouter

from agents import AgentSpec

AGENT_SPEC = AgentSpec(
    key="docs_quick_start",
    display_name="Docs Quick Start Demo Agent",
    default_node_id="my-agent",
    description="Replicates the docs Quick Start flow with the demo_echo reasoner.",
    reasoners=("demo_echo",),
    skills=(),
)


def create_agent(
    *,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    """
    Build the Quick Start docs agent with the router-prefixed demo echo reasoner.
    """
    resolved_node_id = node_id or AGENT_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )
    agent_kwargs.setdefault("version", "1.0.0")

    agent = Agent(
        node_id=resolved_node_id,
        **agent_kwargs,
    )

    reasoners_router = AgentRouter(prefix="demo", tags=["example"])

    @reasoners_router.reasoner()
    async def echo(message: str) -> dict:
        """
        Simple echo reasoner that mirrors the docs Quick Start output.
        """
        response_text = message if isinstance(message, str) else str(message)
        return {
            "original": response_text,
            "echoed": response_text,
            "length": len(response_text),
        }

    agent.include_router(reasoners_router)
    return agent


def create_agent_from_env() -> Agent:
    """
    Convenience helper mirroring running the generated agent module directly.
    """
    node_id = os.environ.get("AGENT_NODE_ID")
    return create_agent(node_id=node_id)


__all__ = ["AGENT_SPEC", "create_agent", "create_agent_from_env"]


if __name__ == "__main__":
    # Allow `python -m agents.docs_quick_start_agent` for local debugging.
    agent = create_agent_from_env()
    agent.run()
