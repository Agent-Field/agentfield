"""Trigger binding types for AgentField reasoners.

A reasoner declares external event sources via the ``triggers`` kwarg on
``@reasoner``. The canonical form passes typed ``EventTrigger`` /
``ScheduleTrigger`` instances; the ``@on_event`` and ``@on_schedule`` sugar
in :mod:`agentfield.decorators` desugars to the same shape.

The control plane registers a code-managed Trigger row per binding when the
agent registers, so the agent never has to provision webhooks itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass(frozen=True)
class TriggerContext:
    """Webhook-trigger metadata exposed to reasoners.

    Available as ``ctx.trigger`` (None when the reasoner was invoked directly
    via app.call(...) instead of by an inbound event). Can also be injected
    as a typed parameter named ``trigger`` or ``webhook`` via signature inspection.

    Attributes:
        trigger_id: AgentField trigger row ID; stable, == public URL slug.
        source: Provider source ("stripe", "github", "slack", "cron", "generic_hmac", "generic_bearer").
        event_type: Provider's event type (or "" for cron tick).
        event_id: AgentField inbound_event ID (replay key).
        idempotency_key: Provider's idempotency key (e.g. evt_xxx).
        received_at: When control plane received the inbound event (ISO string parsed to datetime).
        vc_id: Trigger event VC ID, if DID enabled.
    """

    trigger_id: str
    source: str
    event_type: str
    event_id: str
    idempotency_key: str
    received_at: datetime
    vc_id: Optional[str] = None


@dataclass
class EventTrigger:
    """Bind a reasoner to events emitted by an HTTP-driven Source.

    Attributes:
        source: Registered Source name (``"stripe"``, ``"github"``, ``"slack"``,
            ``"generic_hmac"``, ``"generic_bearer"``).
        types: Event types the reasoner cares about. Empty list means "all".
            Supports prefix-match: ``"pull_request"`` matches
            ``"pull_request.opened"`` and friends.
        secret_env: Name of the env var on the **control plane** that holds
            the provider's webhook secret. Required for Sources whose
            ``secret_required`` is true.
        config: Source-specific JSON config (timestamp tolerance, custom
            header names, etc). The Source's ``Validate`` runs server-side.
        code_origin: Optional source code location (``path/to/file.py:42``)
            where this trigger is declared. Used for observability and drift
            detection. Automatically captured by decorators.
        transform: Optional sync callable to transform raw provider event
            to reasoner input. When set, SDK runs transform(event_dict) before
            invoking the reasoner; the reasoner's ``input`` parameter receives
            the return value rather than the raw event. None means pass raw event.
    """

    source: str
    types: List[str] = field(default_factory=list)
    secret_env: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    code_origin: Optional[str] = None
    transform: Optional[Callable[[dict], Any]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate that transform is not async."""
        if self.transform is not None:
            import inspect

            if inspect.iscoroutinefunction(self.transform):
                raise TypeError(
                    f"EventTrigger transform must be sync, not async. "
                    f"Got: {self.transform.__name__}"
                )


@dataclass
class ScheduleTrigger:
    """Bind a reasoner to a cron schedule.

    Attributes:
        cron: 5-field cron expression (``minute hour dom month dow``).
        timezone: IANA timezone name. Defaults to UTC.
        code_origin: Optional source code location (``path/to/file.py:42``)
            where this trigger is declared. Used for observability and drift
            detection. Automatically captured by decorators.
    """

    cron: str
    timezone: str = "UTC"
    code_origin: Optional[str] = None


Trigger = Union[EventTrigger, ScheduleTrigger]


def trigger_to_payload(trigger: Trigger) -> Dict[str, Any]:
    """Convert a typed trigger into the wire payload sent at registration.

    The control plane expects ``{source, event_types, config, secret_env_var}``;
    schedule triggers normalize to the ``cron`` source with their expression
    embedded in ``config``.

    Note: ``transform`` is not serialized (it's a runtime Python callable).
    """
    if isinstance(trigger, EventTrigger):
        payload: Dict[str, Any] = {
            "source": trigger.source,
            "event_types": list(trigger.types or []),
        }
        if trigger.config:
            payload["config"] = dict(trigger.config)
        if trigger.secret_env:
            payload["secret_env_var"] = trigger.secret_env
        if trigger.code_origin:
            payload["code_origin"] = trigger.code_origin
        return payload
    if isinstance(trigger, ScheduleTrigger):
        payload = {
            "source": "cron",
            "event_types": [],
            "config": {
                "expression": trigger.cron,
                "timezone": trigger.timezone or "UTC",
            },
        }
        if trigger.code_origin:
            payload["code_origin"] = trigger.code_origin
        return payload
    raise TypeError(f"unknown trigger type: {type(trigger).__name__}")
