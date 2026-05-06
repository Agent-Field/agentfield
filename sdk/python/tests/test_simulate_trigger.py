"""Tests for the in-process trigger simulation helper.

Exercises the unit-test path that lets a webhook reasoner be tested without
a control plane, dispatcher, or HTTP server in the loop. The simulation
helper must:

- Build a TriggerContext that mirrors what the runtime would have produced
- Apply the matching binding's transform when one is set
- Pick the most-specific binding when several match the same source
- Inject ``trigger`` / ``webhook`` parameters by name (alias support)
- Run async reasoners transparently
- Load captured payloads from the fixture library

The shipped fixture library (sdk/python/agentfield/fixtures/triggers/*.json)
doubles as the backing data so test fixtures and the production
``af triggers test`` flow stay in lock-step.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentfield.testing import load_fixture, simulate_schedule, simulate_trigger
from agentfield.triggers import EventTrigger, ScheduleTrigger, TriggerContext


# ---------------------------------------------------------------------------
# Lightweight reasoner shims — we don't import the real @reasoner here
# because that decorator pulls in the workflow-registration machinery, which
# isn't relevant to the in-process unit-test path. The simulation helper
# only requires that the function carry a ``_reasoner_triggers`` attribute,
# matching what @reasoner stamps in production.
# ---------------------------------------------------------------------------


def _make_reasoner(fn, triggers):
    """Wrap ``fn`` with the trigger metadata @reasoner would have stamped."""
    fn._reasoner_triggers = list(triggers)
    return fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulateTriggerBasics:
    def test_passes_raw_body_when_no_transform(self):
        seen = {}

        def handler(input, trigger: TriggerContext):
            seen["input"] = input
            seen["source"] = trigger.source
            return "ok"

        bound = _make_reasoner(handler, [EventTrigger(source="stripe")])
        result = simulate_trigger(
            bound,
            source="stripe",
            event_type="payment_intent.succeeded",
            body={"foo": "bar"},
        )
        assert result == "ok"
        assert seen["input"] == {"foo": "bar"}
        assert seen["source"] == "stripe"

    def test_runs_transform_on_matching_binding(self):
        @dataclass
        class Order:
            order_id: str
            amount: int

        def stripe_to_order(event):
            obj = event["data"]["object"]
            return Order(order_id=obj["metadata"]["order_id"], amount=obj["amount"])

        seen = {}

        def handler(order, trigger: TriggerContext):
            seen["order"] = order
            return order.order_id

        bound = _make_reasoner(
            handler,
            [
                EventTrigger(
                    source="stripe",
                    types=["payment_intent.succeeded"],
                    transform=stripe_to_order,
                )
            ],
        )
        result = simulate_trigger(
            bound,
            source="stripe",
            event_type="payment_intent.succeeded",
            body=load_fixture("stripe"),
        )
        assert result == "ord_demo_42"
        assert isinstance(seen["order"], Order)
        assert seen["order"].amount == 5000

    def test_no_match_skips_transform(self):
        called_transform = {"hit": False}

        def stripe_transform(event):
            called_transform["hit"] = True
            return event

        def handler(input, trigger: TriggerContext):
            return input

        bound = _make_reasoner(
            handler,
            [EventTrigger(source="stripe", transform=stripe_transform)],
        )
        result = simulate_trigger(bound, source="github", body={"action": "opened"})
        assert result == {"action": "opened"}
        assert called_transform["hit"] is False

    def test_picks_most_specific_binding_when_multiple_match(self):
        called = []

        def t_specific(_):
            called.append("specific")
            return {"via": "specific"}

        def t_catchall(_):
            called.append("catchall")
            return {"via": "catchall"}

        def handler(input, trigger: TriggerContext):
            return input

        bound = _make_reasoner(
            handler,
            [
                EventTrigger(source="github", transform=t_catchall),
                EventTrigger(
                    source="github", types=["pull_request"], transform=t_specific
                ),
            ],
        )
        result = simulate_trigger(
            bound,
            source="github",
            event_type="pull_request.opened",
            body={"x": 1},
        )
        assert result == {"via": "specific"}
        assert called == ["specific"]


class TestSimulateTriggerInjection:
    def test_injects_trigger_param_by_name(self):
        def handler(input, trigger: TriggerContext):
            return trigger.source

        bound = _make_reasoner(handler, [EventTrigger(source="github")])
        out = simulate_trigger(bound, source="github", body={})
        assert out == "github"

    def test_injects_webhook_alias(self):
        def handler(input, webhook: TriggerContext):
            return webhook.event_type

        bound = _make_reasoner(handler, [EventTrigger(source="slack")])
        out = simulate_trigger(
            bound, source="slack", event_type="app_mention", body={}
        )
        assert out == "app_mention"

    def test_injects_ctx_with_trigger(self):
        def handler(input, ctx):
            return ctx.trigger.idempotency_key

        bound = _make_reasoner(handler, [EventTrigger(source="generic_hmac")])
        out = simulate_trigger(
            bound,
            source="generic_hmac",
            idempotency_key="idem_xyz_42",
            body={},
        )
        assert out == "idem_xyz_42"

    def test_handler_with_no_extra_params(self):
        def handler(input):
            return input["foo"]

        bound = _make_reasoner(handler, [EventTrigger(source="stripe")])
        assert simulate_trigger(bound, source="stripe", body={"foo": 99}) == 99


class TestSimulateTriggerAsync:
    def test_awaits_async_reasoner(self):
        async def handler(input, trigger: TriggerContext):
            return f"{trigger.source}:{input['n']}"

        bound = _make_reasoner(handler, [EventTrigger(source="github")])
        assert (
            simulate_trigger(bound, source="github", body={"n": 7})
            == "github:7"
        )


class TestSimulateSchedule:
    def test_cron_invocation_synthesizes_source(self):
        def handler(input, trigger: TriggerContext):
            assert trigger.source == "cron"
            return "tick"

        bound = _make_reasoner(handler, [ScheduleTrigger(cron="* * * * *")])
        assert simulate_schedule(bound, cron="* * * * *") == "tick"


class TestFixtureLoader:
    def test_loads_each_shipped_fixture(self):
        for source in [
            "stripe",
            "github",
            "slack",
            "generic_hmac",
            "generic_bearer",
            "cron",
        ]:
            data = load_fixture(source)
            assert isinstance(data, dict)
            assert data, f"fixture {source!r} should not be empty"

    def test_missing_fixture_raises(self):
        with pytest.raises(FileNotFoundError):
            load_fixture("nonexistent_source_xyz")

    def test_named_variant(self):
        # Default lookup should still work (reuses existing stripe.json).
        data = load_fixture("stripe", "default")
        assert data["type"] == "payment_intent.succeeded"


class TestSimulateTriggerErrors:
    def test_rejects_non_callable(self):
        with pytest.raises(ValueError):
            simulate_trigger(None, source="stripe", body={})  # type: ignore[arg-type]

    def test_reasoner_without_triggers_still_runs(self):
        # A handler that hasn't declared any bindings can still be simulated;
        # no transform is applied (none to match) and the trigger context is
        # synthesized from the call args. Useful when you wire a UI-managed
        # trigger to an existing reasoner without re-decorating it.
        def handler(input, trigger: TriggerContext):
            return trigger.source

        # No _reasoner_triggers attribute at all.
        out = simulate_trigger(handler, source="stripe", body={"x": 1})
        assert out == "stripe"
