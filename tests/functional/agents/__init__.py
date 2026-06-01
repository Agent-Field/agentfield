"""
Reusable agent definitions for functional tests.

Each module in this package should expose:
    - `AGENT_SPEC`: metadata describing the agent node
    - `create_agent(openrouter_config, **kwargs)`: factory returning an Agent
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AgentSpec:
    """
    Metadata describing a functional-test agent node.

    Attributes:
        key: Unique identifier for the agent definition (module-level)
        display_name: Human-friendly label for docs/logs
        default_node_id: Canonical node ID; tests may override this per instance
        description: Summary of what this agent does
        reasoners: Collection of reasoner IDs exposed by the agent
        skills: Collection of skill IDs exposed by the agent (optional)
    """

    key: str
    display_name: str
    default_node_id: str
    description: str
    reasoners: Sequence[str]
    skills: Sequence[str] = ()


__all__ = ["AgentSpec"]
