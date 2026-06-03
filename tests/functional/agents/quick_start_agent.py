"""
Agent definition that mirrors the README Quick Start example.

Tests can import `AGENT_SPEC` + `create_agent` to obtain a fully configured Agent
without replicating the agent definition inline. Each test can override the
node_id to ensure distinct AgentField registrations when multiple nodes run.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import requests
from agentfield import AIConfig, Agent

from agents import AgentSpec

AGENT_SPEC = AgentSpec(
    key="quick_start",
    display_name="Quick Start Reference Agent",
    default_node_id="quick-start-agent",
    description="Mirrors README Quick Start sample with fetch_url skill + summarize reasoner.",
    reasoners=("summarize",),
    skills=("fetch_url",),
)


def create_agent(
    ai_config: AIConfig,
    *,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs,
) -> Agent:
    """
    Build the Quick Start agent with the canonical fetch_url + summarize flow.
    """
    resolved_node_id = node_id or AGENT_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )

    agent = Agent(
        node_id=resolved_node_id,
        ai_config=ai_config,
        **agent_kwargs,
    )

    @agent.skill(name="fetch_url")
    def fetch_url(url: str) -> str:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    @agent.reasoner(name="summarize")
    async def summarize(url: str) -> Dict[str, str]:
        """
        Fetch a URL, summarize it via OpenRouter, and return metadata.
        """
        content = fetch_url(url)
        truncated = content[:2000]

        ai_response = await agent.ai(
            system=(
                "You summarize documentation for internal verification. "
                "Be concise and focus on the site's purpose."
            ),
            user=(
                "Summarize the following web page in no more than two sentences. "
                "Focus on what the site is intended for.\n"
                f"Content:\n{truncated}"
            ),
        )
        summary_text = getattr(ai_response, "text", str(ai_response)).strip()

        return {
            "url": url,
            "summary": summary_text,
            "content_snippet": truncated[:200],
        }

    return agent


def create_agent_from_env() -> Agent:
    """
    Convenience helper to instantiate the agent from environment variables.

    Useful if you want to run this module as a standalone script.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-2.5-flash-lite")
    node_id = os.environ.get("AGENT_NODE_ID")

    ai_config = AIConfig(
        model=model,
        api_key=api_key,
        temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.7")),
        max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "500")),
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "60.0")),
        retry_attempts=int(os.environ.get("OPENROUTER_RETRIES", "2")),
    )
    return create_agent(ai_config, node_id=node_id)


__all__ = ["AGENT_SPEC", "create_agent", "create_agent_from_env"]


if __name__ == "__main__":
    # Allow developers to run: `python -m agents.quick_start_agent`
    agent = create_agent_from_env()
    agent.run()
