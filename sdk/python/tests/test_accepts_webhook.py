"""Tests for accepts_webhook flag on reasoners."""

import pytest
from agentfield import Agent
from agentfield.decorators import reasoner, on_schedule
from agentfield.triggers import EventTrigger


@pytest.mark.unit
async def test_accepts_webhook_default_is_warn():
    """Without triggers or explicit accepts_webhook, default should be 'warn'."""
    
    @reasoner
    async def no_triggers(x: str) -> dict:
        return {"result": x}

    # Check the _accepts_webhook attribute
    assert hasattr(no_triggers, "_accepts_webhook")
    assert no_triggers._accepts_webhook == "warn"


@pytest.mark.unit
async def test_accepts_webhook_auto_true_with_event_trigger():
    """When triggers=[EventTrigger(...)] is passed, auto-set accepts_webhook=True."""

    @reasoner(
        triggers=[
            EventTrigger(
                source="stripe",
                types=["payment_intent.succeeded"],
                secret_env="STRIPE_SECRET",
            )
        ]
    )
    async def webhook_handler(x: str) -> dict:
        return {"result": x}

    assert hasattr(webhook_handler, "_accepts_webhook")
    assert webhook_handler._accepts_webhook is True


@pytest.mark.unit
async def test_accepts_webhook_auto_true_with_schedule_trigger():
    """When @on_schedule sugar is used, auto-set accepts_webhook=True."""

    @reasoner()
    @on_schedule("*/5 * * * *")
    async def scheduled_handler(x: str) -> dict:
        return {"result": x}

    assert hasattr(scheduled_handler, "_accepts_webhook")
    assert scheduled_handler._accepts_webhook is True


@pytest.mark.unit
async def test_accepts_webhook_explicit_false_overrides():
    """Explicit accepts_webhook=False should override any triggers."""

    @reasoner(
        accepts_webhook=False,
        triggers=[
            EventTrigger(
                source="stripe",
                types=["payment_intent.succeeded"],
                secret_env="STRIPE_SECRET",
            )
        ],
    )
    async def no_webhooks(x: str) -> dict:
        return {"result": x}

    assert hasattr(no_webhooks, "_accepts_webhook")
    assert no_webhooks._accepts_webhook is False


@pytest.mark.unit
async def test_accepts_webhook_explicit_true():
    """Explicit accepts_webhook=True should be honored."""

    @reasoner(accepts_webhook=True)
    async def webhook_ready(x: str) -> dict:
        return {"result": x}

    assert hasattr(webhook_ready, "_accepts_webhook")
    assert webhook_ready._accepts_webhook is True


@pytest.mark.unit
def test_accepts_webhook_in_agent_registration():
    """accepts_webhook should be present in ReasonerEntry when registered with Agent."""
    app = Agent(node_id="test_agent", auto_register=False)

    @app.reasoner()
    async def no_triggers(x: str) -> dict:
        return {"result": x}

    @app.reasoner(vc_enabled=None)
    async def webhook_enabled(y: str) -> dict:
        return {"result": y}

    @app.reasoner()
    async def webhook_disabled(z: str) -> dict:
        return {"result": z}

    # Get entries from the registry
    no_triggers_entry = app._reasoner_registry.get("no_triggers")
    webhook_enabled_entry = app._reasoner_registry.get("webhook_enabled")
    webhook_disabled_entry = app._reasoner_registry.get("webhook_disabled")

    assert no_triggers_entry is not None
    assert hasattr(no_triggers_entry, "accepts_webhook")
    assert no_triggers_entry.accepts_webhook == "warn"

    assert webhook_enabled_entry is not None
    assert hasattr(webhook_enabled_entry, "accepts_webhook")
    # Should default to "warn" since no triggers declared via @reasoner
    assert webhook_enabled_entry.accepts_webhook == "warn"

    assert webhook_disabled_entry is not None
    assert hasattr(webhook_disabled_entry, "accepts_webhook")
    assert webhook_disabled_entry.accepts_webhook == "warn"


def test_agent_reasoner_explicit_false_overrides_trigger_metadata():
    """@app.reasoner should preserve accepts_webhook=False with trigger bindings."""
    app = Agent(node_id="test_agent", agentfield_server="http://localhost:8080")

    @app.reasoner(
        accepts_webhook=False,
        triggers=[EventTrigger(source="github")],
    )
    async def no_webhook_trigger(x: str) -> dict:
        return {"x": x}

    metadata = next(r for r in app.reasoners if r["id"] == "no_webhook_trigger")

    assert metadata["accepts_webhook"] == "false"


def test_agent_reasoner_preserves_inner_reasoner_trigger_metadata():
    """@app.reasoner should preserve metadata from stacked module-level @reasoner."""
    app = Agent(node_id="test_agent", agentfield_server="http://localhost:8080")

    @app.reasoner()
    @reasoner(accepts_webhook=False, triggers=[EventTrigger(source="github")])
    async def stacked_no_webhook_trigger(x: str) -> dict:
        return {"x": x}

    metadata = next(
        r for r in app.reasoners if r["id"] == "stacked_no_webhook_trigger"
    )

    assert metadata["accepts_webhook"] == "false"
    assert metadata["triggers"][0]["source"] == "github"


def test_agent_reasoner_outer_triggers_override_inner_warn_default():
    """Outer @app.reasoner triggers should imply webhook opt-in when inner is default."""
    app = Agent(node_id="test_agent", agentfield_server="http://localhost:8080")

    @app.reasoner(triggers=[EventTrigger(source="github")])
    @reasoner()
    async def stacked_trigger_opt_in(x: str) -> dict:
        return {"x": x}

    metadata = next(r for r in app.reasoners if r["id"] == "stacked_trigger_opt_in")

    assert metadata["accepts_webhook"] == "true"
    assert metadata["triggers"][0]["source"] == "github"


def test_agent_reasoner_staged_triggers_override_inner_warn_default():
    """Staged trigger sugar should imply webhook opt-in when inner is default."""
    app = Agent(node_id="test_agent", agentfield_server="http://localhost:8080")

    @app.reasoner()
    @on_schedule("*/5 * * * *")
    @reasoner()
    async def stacked_schedule_opt_in(x: str) -> dict:
        return {"x": x}

    metadata = next(r for r in app.reasoners if r["id"] == "stacked_schedule_opt_in")

    assert metadata["accepts_webhook"] == "true"
    assert metadata["triggers"][0]["source"] == "cron"


def test_agent_reasoner_registers_bound_method_without_metadata_write():
    """Bound method registration should not require setting attrs on method objects."""
    app = Agent(node_id="test_agent", agentfield_server="http://localhost:8080")

    class Service:
        async def handle(self, x: str) -> dict:
            return {"x": x}

    app.reasoner()(Service().handle)

    metadata = next(r for r in app.reasoners if r["id"] == "handle")

    assert metadata["accepts_webhook"] == "warn"


def test_agent_reasoner_does_not_reuse_triggers_between_agent_registrations():
    """Agent-local trigger registrations should not leak through function attrs."""
    app1 = Agent(node_id="agent_one", agentfield_server="http://localhost:8080")
    app2 = Agent(node_id="agent_two", agentfield_server="http://localhost:8080")

    async def shared_handler(x: str) -> dict:
        return {"x": x}

    app1.reasoner(triggers=[EventTrigger(source="github")])(shared_handler)
    app2.reasoner(triggers=[EventTrigger(source="stripe")])(shared_handler)

    app1_metadata = next(r for r in app1.reasoners if r["id"] == "shared_handler")
    app2_metadata = next(r for r in app2.reasoners if r["id"] == "shared_handler")

    assert [trigger["source"] for trigger in app1_metadata["triggers"]] == ["github"]
    assert [trigger["source"] for trigger in app2_metadata["triggers"]] == ["stripe"]
