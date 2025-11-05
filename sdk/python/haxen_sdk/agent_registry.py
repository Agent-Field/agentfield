"""
Agent registry for tracking the current agent instance in thread-local storage.
This allows reasoners to automatically find their parent agent for workflow tracking.
"""

import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

# Thread-local storage for agent instances
_thread_local = threading.local()


def set_current_agent(agent_instance: "Agent"):
    """Register the current agent instance for this thread."""
    _thread_local.current_agent = agent_instance


def get_current_agent_instance() -> Optional["Agent"]:
    """Get the current agent instance for this thread."""
    return getattr(_thread_local, "current_agent", None)


def clear_current_agent():
    """Clear the current agent instance."""
    if hasattr(_thread_local, "current_agent"):
        delattr(_thread_local, "current_agent")
