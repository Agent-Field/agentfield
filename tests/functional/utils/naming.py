"""
Helpers for naming agent nodes in functional tests.
"""

import re
import uuid


SAFE_NODE_ID_CHARS = re.compile(r"[^a-zA-Z0-9-_\.]")


def sanitize_node_id(base: str) -> str:
    """
    Sanitize a node_id base string to include only characters allowed by AgentField.
    """
    return SAFE_NODE_ID_CHARS.sub("-", base).strip("-") or "agent"


def unique_node_id(base: str, *, suffix: str | None = None) -> str:
    """
    Generate a unique node ID for concurrent functional tests.

    Args:
        base: Readable prefix (e.g., "quick-start-agent")
        suffix: Optional deterministic suffix. If omitted, a short UUID is appended.
    """
    token = suffix or uuid.uuid4().hex[:8]
    safe_base = sanitize_node_id(base)
    return f"{safe_base}-{token}"


__all__ = ["sanitize_node_id", "unique_node_id"]
