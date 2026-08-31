"""Opt-in LiteLLM callbacks and AgentField execution metadata."""

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, FrozenSet, List, Optional

from agentfield.logger import log_debug, log_warn

CALLBACKS_ENV = "AGENTFIELD_LITELLM_CALLBACKS"
METADATA_ENV = "AGENTFIELD_LITELLM_METADATA"

_FALSE_VALUES = {"0", "false", "no", "off"}
_AGENTFIELD_REGISTERED: set[str] = set()
# LiteLLM callback names whose logger reads the LangFuse-native metadata
# aliases (trace_id, session_id, trace_name, generation_name, tags).
_LANGFUSE_CALLBACKS = frozenset({"langfuse", "langfuse_otel"})


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
            # Only record a name that one of the branches below actually
            # installed: reporting success for a callback that was never
            # registered would also flip the vendor-alias gate in
            # build_execution_metadata() on for it.
            added = False
            add_callback = getattr(manager, "add_litellm_callback", None)
            if callable(add_callback):
                add_callback(name)
                added = True
            else:
                callbacks = getattr(litellm_module, "callbacks", None)
                if isinstance(callbacks, list):
                    if name not in callbacks:
                        callbacks.append(name)
                    added = True
            if added:
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

    # The aliases below are LangFuse-native keys, so they are stamped only when
    # AgentField itself registered a LangFuse-family callback. Gating on "any
    # AgentField-registered callback" would still re-key, rename and re-tag the
    # generations of an application that registered `langfuse` itself while the
    # operator set AGENTFIELD_LITELLM_CALLBACKS to some other vendor.
    if agentfield_registered_callbacks() & _LANGFUSE_CALLBACKS:
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

    # NOTE: never add a "user_id" or "requester_metadata" key to this dict.
    # LiteLLM's anthropic transform copies metadata["user_id"] into the
    # provider request body, and its vertex transform turns
    # metadata["requester_metadata"] into request labels — either would leak
    # AgentField identifiers past the LiteLLM boundary.
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
            if key in merged:
                # A caller-supplied value — or one an earlier stamp left on the
                # params this call was shallow-copied from — always wins.
                value = merged[key]
            # Give a stamped list value its own object. The tool-calling loop
            # passes `{**litellm_params}` per turn, so without this every turn
            # would share one metadata["tags"] list, and LangFuse-style
            # integrations append to it.
            if isinstance(value, list):
                value = list(value)
            merged[key] = value
        params["metadata"] = merged
    except Exception as exc:
        log_debug(f"Could not apply AgentField LiteLLM metadata: {exc}")
