"""
Agent registry for tracking the current agent instance.

Uses contextvars.ContextVar so the agent instance is correctly inherited
by asyncio tasks (unlike threading.local which is thread-bound and can
return None when coroutines resume on a different thread).
"""

import contextvars
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

# Context variable for agent instances — works correctly with asyncio
_current_agent: contextvars.ContextVar[Optional["Agent"]] = contextvars.ContextVar(
    "current_agent", default=None
)


def set_current_agent(agent_instance: "Agent"):
    """Register the current agent instance for this context."""
    _current_agent.set(agent_instance)


def get_current_agent_instance() -> Optional["Agent"]:
    """Get the current agent instance for this context."""
    return _current_agent.get()


def clear_current_agent():
    """Clear the current agent instance."""
    _current_agent.set(None)
