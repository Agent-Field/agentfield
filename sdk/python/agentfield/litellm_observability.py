"""Opt-in LiteLLM callbacks and AgentField execution metadata."""

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, FrozenSet, List, Optional

from agentfield.logger import log_debug, log_warn

CALLBACKS_ENV = "AGENTFIELD_LITELLM_CALLBACKS"
METADATA_ENV = "AGENTFIELD_LITELLM_METADATA"

_FALSE_VALUES = {"0", "false", "no", "off"}
_AGENTFIELD_REGISTERED: set[str] = set()


def resolve_callbacks(
    explicit: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Resolve callback names from explicit values and the environment."""
    source = os.environ if env is None else env
    values = list(explicit or [])
    configured = source.get(CALLBACKS_ENV)
    if configured:
        values.extend(configured.split(","))

    resolved: List[str] = []
    for value in values:
        name = str(value).strip().lower()
        if name and name not in resolved:
            resolved.append(name)
    return resolved


def register_callbacks(
    explicit: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    litellm_module: Any = None,
) -> List[str]:
    """Register configured LiteLLM callbacks without allowing failures to escape."""
    names = resolve_callbacks(explicit, env)
    if not names:
        return []

    registered: List[str] = []
    try:
        if litellm_module is None:
            import litellm as litellm_module

        known = (
            getattr(
                litellm_module,
                "_known_custom_logger_compatible_callbacks",
                [],
            )
            or []
        )
        manager = getattr(litellm_module, "logging_callback_manager", None)
        for name in names:
            if name not in known:
                log_debug(
                    f"LiteLLM callback '{name}' is not in LiteLLM's known callback list; registering it anyway"
                )
            add_callback = getattr(manager, "add_litellm_callback", None)
            if callable(add_callback):
                add_callback(name)
            else:
                callbacks = getattr(litellm_module, "callbacks", None)
                if isinstance(callbacks, list) and name not in callbacks:
                    callbacks.append(name)
            _AGENTFIELD_REGISTERED.add(name)
            registered.append(name)
    except Exception as exc:
        log_warn(f"Could not register LiteLLM observability callback: {exc}")
    return registered


def agentfield_registered_callbacks() -> FrozenSet[str]:
    """Return callbacks successfully registered by AgentField in this process."""
    return frozenset(_AGENTFIELD_REGISTERED)


def metadata_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether AgentField execution metadata injection is enabled."""
    source = os.environ if env is None else env
    return source.get(METADATA_ENV, "").strip().lower() not in _FALSE_VALUES


def build_execution_metadata(
    context: Any = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Build LiteLLM-only correlation metadata for an execution context."""
    if not metadata_enabled(env):
        return {}
    if context is None:
        from agentfield.execution_context import get_current_context

        context = get_current_context()
    if context is None:
        return {}

    node_id = getattr(context, "agent_node_id", None) or getattr(
        getattr(context, "agent_instance", None), "node_id", None
    )
    reasoner = getattr(context, "reasoner_name", None)
    run_id = getattr(context, "run_id", None)
    values = {
        "agentfield_execution_id": getattr(context, "execution_id", None),
        "agentfield_run_id": run_id,
        "agentfield_agent_node_id": node_id,
        "agentfield_reasoner": reasoner,
        "agentfield_session_id": getattr(context, "session_id", None),
        "agentfield_parent_execution_id": getattr(context, "parent_execution_id", None),
    }
    metadata: dict[str, Any] = {
        key: str(value) for key, value in values.items() if value not in (None, "")
    }

    if agentfield_registered_callbacks():
        aliases = {
            "trace_id": run_id,
            "session_id": getattr(context, "session_id", None),
            "trace_name": node_id,
            "generation_name": (
                f"{node_id}.{reasoner}" if node_id and reasoner else reasoner
            ),
        }
        metadata.update(
            {
                key: str(value)
                for key, value in aliases.items()
                if value not in (None, "")
            }
        )
        tags = ["agentfield"]
        if node_id not in (None, ""):
            tags.append(f"agentfield-node:{node_id}")
        if reasoner not in (None, ""):
            tags.append(f"agentfield-reasoner:{reasoner}")
        metadata["tags"] = tags

    metadata.pop("user_id", None)
    metadata.pop("requester_metadata", None)
    return metadata


def apply_execution_metadata(
    params: MutableMapping[str, Any],
    *,
    context: Any = None,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Merge execution metadata into completion parameters without overriding callers."""
    try:
        if not metadata_enabled(env):
            return
        stamp = build_execution_metadata(context, env=env)
        if not stamp:
            return
        existing = params.get("metadata")
        if existing is not None and not isinstance(existing, dict):
            return
        merged = dict(existing or {})
        for key, value in stamp.items():
            merged.setdefault(key, value)
        params["metadata"] = merged
    except Exception as exc:
        log_debug(f"Could not apply AgentField LiteLLM metadata: {exc}")
