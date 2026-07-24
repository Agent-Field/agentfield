"""Trigger/webhook parameter binding — runtime invocation contract.

Regression coverage for the production failure where a scheduled reasoner
declared with the documented ``trigger`` parameter crashed every time it fired:

    cla_reminder_sweep() got multiple values for argument 'trigger'

Root cause: when a reasoner is invoked by a trigger, the runtime passed the
event payload as the FIRST POSITIONAL argument *and* separately injected
``trigger=`` as a keyword. For a reasoner whose only parameter is ``trigger``
(``def r(trigger=None)``) both filled the same slot → TypeError. The unit-test
harness (``simulate_trigger`` / ``_bind_reasoner_args``) bound correctly, so the
bug was invisible to existing tests — the runtime and the harness disagreed.

Validation Contract (behaviours, not implementation):
  C1. Firing a trigger at a reasoner whose only parameter is ``trigger``
      succeeds (HTTP 200) and ``trigger`` receives the TriggerContext.
  C2. ``def r(event, trigger=None)`` binds the event payload to ``event`` and
      the TriggerContext to ``trigger``.
  C3. The ``webhook`` alias behaves exactly like ``trigger``.
  C4. ``def r(payload)`` (no trigger param) still receives the event payload.
  C5. The runtime binds a payload to the first parameter that is NOT a
      framework-injected slot (trigger / webhook / execution_context); a
      reasoner declaring only injected slots drops the payload rather than
      colliding.
  C6. Non-trigger ``app.call`` invocations (plain kwargs dict) are unchanged.

The HTTP tests below exercise the REAL runtime path (route → envelope unwrap →
``_execute_reasoner_endpoint`` → reasoner call), which is the path that was
broken; the unit tests pin the binding helper directly.
"""

import inspect

import httpx
import pytest

from agentfield import EventTrigger, ScheduleTrigger, TriggerContext
from agentfield.agent import Agent, _bind_trigger_payload
from agentfield.agent_registry import (
    clear_current_agent,
    get_current_agent_instance,
    set_current_agent,
)
from agentfield.logger import get_logger, set_cp_client

from tests.helpers import create_test_agent


@pytest.fixture(autouse=True)
def _restore_global_agent_state():
    """Constructing an Agent has two process-wide side effects that these tests
    must not leak into later tests:

      1. ``Agent.__init__`` calls ``set_cp_client(self.client)``, repointing the
         GLOBAL logger's control-plane client at this test's fake client (which
         has no ``post_execution_logs``). test_workflow_parent_child assumes the
         global client is ``None`` and dispatches execution logs to it — a leaked
         fake makes it raise ``AttributeError`` mid-reasoner.
      2. Driving a reasoner over HTTP marks its Agent as the process-wide
         "current agent" (``Agent._current_agent`` + the current-agent contextvar).

    Snapshot and restore both so this module is hermetic regardless of order."""
    prev_cp_client = get_logger()._cp_client
    prev_class_attr = getattr(Agent, "_current_agent", None)
    prev_contextvar = get_current_agent_instance()
    try:
        yield
    finally:
        set_cp_client(prev_cp_client)
        Agent._current_agent = prev_class_attr
        if prev_contextvar is None:
            clear_current_agent()
        else:
            set_current_agent(prev_contextvar)


def _envelope(event: dict) -> dict:
    """A dispatcher trigger envelope as the control plane forwards it."""
    return {
        "event": event,
        "_meta": {
            "trigger_id": "trg_sched_1",
            "source": "schedule",
            "event_type": "schedule.fired",
            "event_id": "evt_1",
            "idempotency_key": "evt_1_key",
            "received_at": "2026-06-08T09:00:00+00:00",
        },
    }


async def _fire(agent, path: str, event: dict) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        return await client.post(
            path,
            json=_envelope(event),
            headers={"x-workflow-id": "wf-1", "x-execution-id": "exec-1"},
        )


def _inline(agent):
    """Run reasoners synchronously in-process (no async dispatch / network)."""
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None


# --------------------------------------------------------------------------- #
# End-to-end runtime path (the path that crashed in production)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_schedule_reasoner_with_only_trigger_param_succeeds(monkeypatch):
    """C1: the exact cla_reminder_sweep shape — only a ``trigger`` parameter."""
    agent, _ = create_test_agent(monkeypatch)
    _inline(agent)
    seen = {}

    @agent.reasoner(tags=["scheduled"], triggers=[ScheduleTrigger(cron="0 9 * * *")])
    async def cla_reminder_sweep(trigger: TriggerContext | None = None) -> dict:
        seen["trigger"] = trigger
        return {"ran": True, "source": getattr(trigger, "source", None)}

    resp = await _fire(
        agent,
        "/reasoners/cla_reminder_sweep",
        {
            "expression": "0 9 * * *",
            "fired_at": "2026-06-08T09:00:00Z",
            "timezone": "UTC",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ran": True, "source": "schedule"}
    assert isinstance(seen["trigger"], TriggerContext)
    assert seen["trigger"].source == "schedule"


@pytest.mark.asyncio
async def test_trigger_reasoner_with_event_and_trigger_binds_both(monkeypatch):
    """C2: payload → first non-injected param; context → ``trigger``."""
    agent, _ = create_test_agent(monkeypatch)
    _inline(agent)
    seen = {}

    @agent.reasoner(triggers=[ScheduleTrigger(cron="0 9 * * *")])
    async def handler(
        event: dict | None = None, trigger: TriggerContext | None = None
    ) -> dict:
        seen["event"] = event
        seen["trigger"] = trigger
        return {"ok": True}

    resp = await _fire(agent, "/reasoners/handler", {"k": "v"})

    assert resp.status_code == 200, resp.text
    assert seen["event"] == {"k": "v"}
    assert isinstance(seen["trigger"], TriggerContext)


@pytest.mark.asyncio
async def test_webhook_alias_only_param_succeeds(monkeypatch):
    """C3: ``webhook`` behaves like ``trigger``."""
    agent, _ = create_test_agent(monkeypatch)
    _inline(agent)
    seen = {}

    @agent.reasoner(triggers=[ScheduleTrigger(cron="0 9 * * *")])
    async def hook(webhook: TriggerContext | None = None) -> dict:
        seen["webhook"] = webhook
        return {"ran": True}

    resp = await _fire(agent, "/reasoners/hook", {"any": "thing"})

    assert resp.status_code == 200, resp.text
    assert isinstance(seen["webhook"], TriggerContext)


@pytest.mark.asyncio
async def test_trigger_reasoner_with_plain_payload_param(monkeypatch):
    """C4: a reasoner with no trigger param still receives the event payload."""
    agent, _ = create_test_agent(monkeypatch)
    _inline(agent)
    seen = {}

    @agent.reasoner(triggers=[ScheduleTrigger(cron="0 9 * * *")])
    async def consume(payload: dict | None = None) -> dict:
        seen["payload"] = payload
        return {"ran": True}

    resp = await _fire(agent, "/reasoners/consume", {"n": 1})

    assert resp.status_code == 200, resp.text
    assert seen["payload"] == {"n": 1}


@pytest.mark.asyncio
async def test_canonical_event_trigger_transform_is_applied(monkeypatch):
    """C7 (regression, #693): an ``EventTrigger`` declared with a ``transform``
    on a PLAIN handler via the canonical ``@app.reasoner(triggers=[...])`` form
    must have that transform applied on the real runtime path — the handler
    receives the TRANSFORMED object, not the raw event payload.

    The decorator-dedup refactor moved trigger merging into
    ``resolve_reasoner_metadata`` but stopped stamping ``_reasoner_triggers`` on
    the stored handler, so ``_execute_reasoner_endpoint`` saw no bindings and
    silently skipped the transform. This drives the same route → envelope
    unwrap → ``_execute_reasoner_endpoint`` path production uses."""
    agent, _ = create_test_agent(monkeypatch)
    _inline(agent)
    seen = {}

    def to_domain(event: dict) -> dict:
        return {"transformed": True, "amount_cents": event.get("amount", 0) * 100}

    @agent.reasoner(
        triggers=[EventTrigger(source="stripe", types=["payment"], transform=to_domain)]
    )
    async def on_payment(payload: dict | None = None) -> dict:
        seen["payload"] = payload
        return {"ran": True}

    envelope = {
        "event": {"amount": 5},
        "_meta": {
            "trigger_id": "trg_evt_1",
            "source": "stripe",
            "event_type": "payment.succeeded",
            "event_id": "evt_1",
            "idempotency_key": "evt_1_key",
            "received_at": "2026-06-08T09:00:00+00:00",
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/on_payment",
            json=envelope,
            headers={"x-workflow-id": "wf-1", "x-execution-id": "exec-1"},
        )

    assert resp.status_code == 200, resp.text
    # The handler must receive the transformed object, not the raw event.
    assert seen["payload"] == {"transformed": True, "amount_cents": 500}


# --------------------------------------------------------------------------- #
# Binding helper — exhaustive signature shapes (C5)
# --------------------------------------------------------------------------- #

PAYLOAD = {"hello": "world"}


def _sig(fn) -> inspect.Signature:
    return inspect.signature(fn)


def test_bind_only_trigger_drops_payload():
    def r(trigger=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {})


def test_bind_only_webhook_drops_payload():
    def r(webhook=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {})


def test_bind_event_then_trigger_binds_event_by_keyword():
    def r(event, trigger=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {"event": PAYLOAD})


def test_bind_payload_only():
    def r(payload): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {"payload": PAYLOAD})


def test_bind_skips_execution_context():
    def r(event, execution_context=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {"event": PAYLOAD})


def test_bind_trigger_before_event_does_not_collide():
    """Pathological ordering: injected param first. Payload must still land on
    ``event`` (by keyword) so it never fills the injected slot positionally."""

    def r(trigger=None, event=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {"event": PAYLOAD})


def test_bind_positional_only_uses_positional():
    def r(event, /, trigger=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((PAYLOAD,), {})


def test_bind_var_positional_uses_positional():
    def r(*events, trigger=None): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((PAYLOAD,), {})


def test_bind_no_params_drops_payload():
    def r(): ...

    assert _bind_trigger_payload(_sig(r), PAYLOAD) == ((), {})
