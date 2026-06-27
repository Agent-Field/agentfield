"""Shared metadata helpers for AgentField decorators."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, List, Optional, Union


PENDING_TRIGGERS_ATTR = "_pending_triggers"


def code_origin(fn: Callable[..., Any]) -> Optional[str]:
    """Capture the source file and declaration line for a function."""
    try:
        src = inspect.getsourcefile(fn)
        line = fn.__code__.co_firstlineno
        if src and line:
            return f"{src}:{line}"
    except Exception:
        pass
    return None


def stage_trigger(func: Callable[..., Any], trigger: Any) -> None:
    """Stage a trigger for the next outer reasoner decorator."""
    existing = getattr(func, PENDING_TRIGGERS_ATTR, None)
    if existing is None:
        existing = []
        setattr(func, PENDING_TRIGGERS_ATTR, existing)
    existing.append(trigger)


def split_direct_registration_arg(value: Any) -> tuple[Optional[Callable[..., Any]], Any]:
    """Separate ``@decorator`` usage from ``@decorator(...)`` options."""
    if value and (inspect.isfunction(value) or inspect.ismethod(value)):
        return value, None
    return None, value


def resolve_reasoner_metadata(
    func: Callable[..., Any],
    *,
    triggers: Optional[Iterable[Any]] = None,
    accepts_webhook: Optional[Union[bool, str]] = None,
) -> tuple[List[Any], Union[bool, str]]:
    """Merge staged and explicit trigger metadata for a reasoner function."""
    merged = list(triggers or [])
    pending = getattr(func, PENDING_TRIGGERS_ATTR, None)
    if pending:
        merged.extend(pending)
        try:
            delattr(func, PENDING_TRIGGERS_ATTR)
        except AttributeError:
            pass

    origin = code_origin(func)
    if origin:
        for trigger in merged:
            if not getattr(trigger, "code_origin", None):
                trigger.code_origin = origin

    if accepts_webhook is not None:
        resolved_accepts_webhook: Union[bool, str] = accepts_webhook
    elif merged:
        resolved_accepts_webhook = True
    else:
        resolved_accepts_webhook = "warn"

    return merged, resolved_accepts_webhook
