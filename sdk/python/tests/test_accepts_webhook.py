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
