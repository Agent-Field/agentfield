"""Opt-in LiteLLM callbacks and AgentField execution metadata."""

import hashlib
import importlib.metadata
import os
import re
import threading
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, FrozenSet, List, Optional

from agentfield.logger import log_debug, log_warn

CALLBACKS_ENV = "AGENTFIELD_LITELLM_CALLBACKS"
METADATA_ENV = "AGENTFIELD_LITELLM_METADATA"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_AGENTFIELD_REGISTERED: set[str] = set()
_AGENTFIELD_REGISTRATION_MODULES: dict[str, Any] = {}
_REGISTRATION_LOCK = threading.RLock()
# LiteLLM callback names whose logger reads the LangFuse-native metadata
# aliases (trace_id, trace_metadata, session_id, trace_name, generation_name,
# tags).
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

    try:
        if litellm_module is None:
            import litellm as litellm_module
    except Exception as exc:
        log_warn(f"Could not register LiteLLM observability callback: {exc}")
        return []

    known = (
        getattr(
            litellm_module,
            "_known_custom_logger_compatible_callbacks",
            [],
        )
        or []
    )
    manager = getattr(litellm_module, "logging_callback_manager", None)
    registered: List[str] = []
    with _REGISTRATION_LOCK:
        _reconcile_registered_callbacks()
        for name in names:
            if (
                name == "langfuse"
                and getattr(litellm_module, "__name__", None) == "litellm"
                and not _standard_langfuse_runtime_compatible()
            ):
                continue
            if name not in known:
                log_debug(
                    f"LiteLLM callback '{name}' is not in LiteLLM's known callback list; registering it anyway"
                )

            # A callback already installed by application code is deliberately
            # not claimed by AgentField. This ownership distinction is what
            # prevents the native LangFuse aliases from re-keying an existing
            # application-managed integration.
            if _callback_is_active(litellm_module, name):
                if (
                    name in _AGENTFIELD_REGISTERED
                    and _AGENTFIELD_REGISTRATION_MODULES.get(name) is litellm_module
                ):
                    registered.append(name)
                continue

            try:
                add_callback = getattr(manager, "add_litellm_callback", None)
                if callable(add_callback):
                    add_callback(name)
                else:
                    callbacks = getattr(litellm_module, "callbacks", None)
                    if isinstance(callbacks, list) and name not in callbacks:
                        callbacks.append(name)
            except Exception as exc:
                log_warn(
                    f"Could not register LiteLLM observability callback '{name}': {exc}"
                )
                continue

            # LiteLLM's manager returns None even when it refuses a callback
            # (for example, at MAX_CALLBACKS). Only claim ownership after the
            # public callback list proves that the name is actually active.
            if _callback_is_active(litellm_module, name):
                _AGENTFIELD_REGISTERED.add(name)
                _AGENTFIELD_REGISTRATION_MODULES[name] = litellm_module
                registered.append(name)
            else:
                log_debug(f"LiteLLM did not install observability callback '{name}'")
    return registered


def _standard_langfuse_runtime_compatible() -> bool:
    """Validate the v2 client required by LiteLLM's standard callback path."""
    try:
        installed = importlib.metadata.version("langfuse")
    except importlib.metadata.PackageNotFoundError:
        log_warn(
            "LiteLLM callback 'langfuse' was not registered: install "
            "agentfield[langfuse] first"
        )
        return False

    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed)
    release = tuple(int(part) for part in match.groups()) if match else None
    if release is None or not ((2, 59, 7) <= release < (3, 0, 0)):
        log_warn(
            "LiteLLM callback 'langfuse' was not registered: installed "
            f"langfuse {installed} is incompatible; install "
            "agentfield[langfuse] for the supported v2 client"
        )
        return False
    return True


def _callback_is_active(litellm_module: Any, name: str) -> bool:
    """Return whether ``name`` is present in LiteLLM's process-global callbacks."""
    callbacks = getattr(litellm_module, "callbacks", None)
    if not isinstance(callbacks, Sequence) or isinstance(callbacks, (str, bytes)):
        return False
    try:
        return any(callback == name for callback in callbacks)
    except Exception:
        return False


def _reconcile_registered_callbacks() -> None:
    """Drop AgentField ownership records whose LiteLLM callback was removed."""
    for name in tuple(_AGENTFIELD_REGISTERED):
        module = _AGENTFIELD_REGISTRATION_MODULES.get(name)
        if module is None or not _callback_is_active(module, name):
            _AGENTFIELD_REGISTERED.discard(name)
            _AGENTFIELD_REGISTRATION_MODULES.pop(name, None)


def agentfield_registered_callbacks() -> FrozenSet[str]:
    """Return callbacks successfully registered by AgentField in this process."""
    with _REGISTRATION_LOCK:
        _reconcile_registered_callbacks()
        return frozenset(_AGENTFIELD_REGISTERED)


def metadata_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether AgentField execution metadata injection is enabled."""
    source = os.environ if env is None else env
    configured = source.get(METADATA_ENV)
    if configured is not None:
        return configured.strip().lower() in _TRUE_VALUES

    # Callback configuration is itself an explicit observability opt-in. The
    # active-registration fallback covers programmatic register_callbacks()
    # calls and keeps metadata enabled for the lifetime of a callback that
    # AgentField owns, even if the environment is later cleared.
    return bool(resolve_callbacks(env=source)) or bool(
        agentfield_registered_callbacks()
    )


def _langfuse_trace_id(run_id: Any) -> str:
    """Derive LangFuse's stable 32-hex W3C trace ID from an AgentField run ID."""
    return hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:32]


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
        trace_metadata = {
            key: value
            for key, value in metadata.items()
            if key.startswith("agentfield_")
        }
        aliases = {
            "trace_id": _langfuse_trace_id(run_id) if run_id else None,
            # LiteLLM intentionally filters arbitrary completion metadata from
            # LangFuse. Its trace_* steering convention promotes this mapping
            # to trace.metadata so the original AgentField IDs remain usable
            # for joins instead of being lost behind the derived trace ID.
            "trace_metadata": trace_metadata or None,
            "session_id": getattr(context, "session_id", None),
            "trace_name": node_id,
            "generation_name": (
                f"{node_id}.{reasoner}" if node_id and reasoner else reasoner
            ),
        }
        metadata.update(
            {
                key: value if key == "trace_metadata" else str(value)
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
