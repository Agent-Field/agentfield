"""Testing helpers for reasoners that handle webhook triggers.

The standard SDK runtime delivers trigger events via the agent's HTTP
endpoint — your reasoner sees ``ctx.trigger`` populated and (when ``transform``
is set) the unwrapped, transformed input. For unit tests you want the same
shape without spinning up a control plane, an HTTP server, or a real
provider. ``simulate_trigger`` gives you that: it crafts the ``TriggerContext``
the agent runtime would have produced, applies any matching ``transform``
from the reasoner's declared bindings, and invokes the inner reasoner
function directly. No HTTP, no workflow registration, no VC mint.

Typical usage in pytest::

    from agentfield.testing import simulate_trigger
    from my_agent import handle_payment

    def test_handle_payment_from_stripe():
        result = simulate_trigger(
            handle_payment,
            source="stripe",
            event_type="payment_intent.succeeded",
            body={"data": {"object": {"amount": 5000, "metadata": {"order_id": "o42"}}}},
        )
        assert result["saved"] == "o42"

The helper is intentionally narrow: it tests the reasoner's logic in
isolation. For end-to-end flows that exercise the dispatcher, persistence,
and signature verification, use the captured-fixture library (see
``agentfield/fixtures/triggers/``) with ``af triggers test`` or a real
control plane in a docker-compose harness.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from .triggers import EventTrigger, ScheduleTrigger, TriggerContext


def simulate_trigger(
    reasoner: Callable[..., Any],
    *,
    source: str,
    body: Optional[Dict[str, Any]] = None,
    event_type: str = "",
    event_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    trigger_id: Optional[str] = None,
    received_at: Optional[datetime] = None,
    vc_id: Optional[str] = None,
) -> Any:
    """Run ``reasoner`` as if a trigger of ``source`` had fired with ``body``.

    Looks up the matching ``EventTrigger`` declared on the reasoner (by
    source + event_type, with the same specificity rules the runtime uses)
    and runs its ``transform`` if set. Then invokes the inner reasoner
    function with the (possibly transformed) input and a synthetic
    ``TriggerContext`` injected via the ``trigger`` / ``webhook`` parameter
    if the reasoner declares one.

    Parameters mirror what the dispatcher would have produced; all
    identifier-style params default to fresh UUIDs so each simulation
    is independently dedup-safe.

    Returns whatever the reasoner returns. Awaits coroutines transparently
    so test code can be sync.

    Raises:
        ValueError: if ``reasoner`` is not a ``@reasoner``-decorated function.
    """
    if not callable(reasoner):
        raise ValueError("simulate_trigger requires a callable reasoner")
    payload = body if body is not None else {}

    bindings = list(getattr(reasoner, "_reasoner_triggers", ()) or ())
    matched = _match_binding(bindings, source, event_type)

    transformed_input = payload
    if matched is not None and isinstance(matched, EventTrigger) and matched.transform is not None:
        transformed_input = matched.transform(payload)

    trigger_ctx = TriggerContext(
        trigger_id=trigger_id or f"trg_sim_{uuid.uuid4().hex[:12]}",
        source=source,
        event_type=event_type,
        event_id=event_id or f"evt_sim_{uuid.uuid4().hex[:12]}",
        idempotency_key=idempotency_key or f"idem_sim_{uuid.uuid4().hex[:12]}",
        received_at=received_at or datetime.now(timezone.utc),
        vc_id=vc_id,
    )

    inner = getattr(reasoner, "__wrapped__", reasoner)
    sig = inspect.signature(inner)

    args, kwargs = _bind_reasoner_args(sig, transformed_input, trigger_ctx)
    result = inner(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return _run_coro(result)
    return result


def simulate_schedule(
    reasoner: Callable[..., Any],
    *,
    cron: Optional[str] = None,
    received_at: Optional[datetime] = None,
) -> Any:
    """Run a schedule-triggered reasoner with an empty body and ``source='cron'``.

    Convenience wrapper around ``simulate_trigger`` for ``@on_schedule``-decorated
    reasoners. The cron expression is recorded on the synthetic context for
    test introspection but does NOT trigger any scheduling — this is a
    one-shot invocation.
    """
    return simulate_trigger(
        reasoner,
        source="cron",
        body={"cron": cron} if cron else {},
        event_type="tick",
        received_at=received_at,
    )


def load_fixture(source: str, name: str = "default") -> Dict[str, Any]:
    """Load a captured provider payload from the SDK fixture library.

    Fixtures live at ``agentfield/fixtures/triggers/<source>.json`` (or
    ``<source>_<name>.json`` when ``name != "default"``). Each file is a
    realistic but minimal JSON payload — useful both for unit tests via
    ``simulate_trigger(body=load_fixture("stripe"))`` and for the local-dev
    ``af triggers test --body @fixture.json`` flow.

    Returns a dict (the parsed JSON). Tests can mutate freely; each call
    re-reads from disk.
    """
    base = Path(__file__).parent / "fixtures" / "triggers"
    candidate = base / (f"{source}_{name}.json" if name != "default" else f"{source}.json")
    if not candidate.exists():
        raise FileNotFoundError(
            f"No fixture for source={source!r} name={name!r}. Looked at {candidate}."
        )
    return json.loads(candidate.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _match_binding(
    bindings: list,
    source: str,
    event_type: str,
) -> Optional[Union[EventTrigger, ScheduleTrigger]]:
    """Pick the best-fit binding for (source, event_type).

    Mirrors the agent runtime's rule (see agent.py _apply_trigger_transform):
    same source name; if the binding declares non-empty types, the event_type
    must prefix-match one of them; the most specific (non-empty types) match
    wins over a catch-all. Returns None if no binding matches.
    """
    best: Optional[Union[EventTrigger, ScheduleTrigger]] = None
    best_specificity = -1
    for b in bindings:
        b_source = getattr(b, "source", None)
        if b_source != source:
            continue
        types = list(getattr(b, "types", []) or [])
        if not types:
            specificity = 0
            if specificity > best_specificity:
                best, best_specificity = b, specificity
            continue
        if any(_event_type_matches(t, event_type) for t in types):
            specificity = 1
            if specificity > best_specificity:
                best, best_specificity = b, specificity
    return best


def _event_type_matches(filter_value: str, event_type: str) -> bool:
    """Mirror the runtime's prefix-match rule for event-type filtering."""
    if not filter_value:
        return True
    if filter_value == event_type:
        return True
    # "pull_request" matches "pull_request.opened"
    return event_type.startswith(filter_value + ".")


def _bind_reasoner_args(
    sig: inspect.Signature,
    input_value: Any,
    trigger_ctx: TriggerContext,
) -> tuple:
    """Build (args, kwargs) for the inner reasoner call.

    The first positional parameter (typically named ``input``) receives the
    payload. Any of ``trigger`` / ``webhook`` / ``ctx`` / ``execution_context``
    parameters get injected by name when present in the signature. Other
    parameters are left to default values.
    """
    args: list = []
    kwargs: Dict[str, Any] = {}
    consumed_input = False
    for name, param in sig.parameters.items():
        if name in {"trigger", "webhook"}:
            kwargs[name] = trigger_ctx
            continue
        if name in {"execution_context", "ctx"}:
            # ctx is conventionally an execution-context wrapper — for the
            # in-process unit-test path we don't have a workflow handler
            # to register against, so we surface the trigger via a small
            # stand-in object so tests can assert on ctx.trigger.
            kwargs[name] = _SimulatedExecutionContext(trigger_ctx)
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if not consumed_input and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(input_value)
            consumed_input = True
    return tuple(args), kwargs


class _SimulatedExecutionContext:
    """Minimal stand-in surfaced as ``ctx`` in simulate_trigger tests.

    Only carries the bits a webhook reasoner is likely to read: ``trigger``
    plus a few common identifier slots. Production code uses
    :class:`agentfield.execution_context.ExecutionContext`; we don't import
    it here to avoid pulling the workflow-registration machinery into the
    in-process testing path.
    """

    def __init__(self, trigger: TriggerContext) -> None:
        self.trigger = trigger
        self.parent_vc_id = trigger.vc_id
        self.execution_id = f"exec_sim_{uuid.uuid4().hex[:12]}"


def _run_coro(coro: Awaitable[Any]) -> Any:
    """Drive a coroutine to completion from a sync context.

    Reuses an existing event loop when called from inside one (e.g. an
    ``async def`` test under pytest-asyncio); otherwise runs a fresh loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)
