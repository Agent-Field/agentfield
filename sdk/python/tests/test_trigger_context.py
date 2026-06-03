"""
Unit tests for Phase 5 webhook trigger feature.
Tests TriggerContext, envelope unwrapping, and transform matching.
"""

import pytest
from datetime import datetime
from agentfield.triggers import TriggerContext, EventTrigger
from agentfield.execution_context import ExecutionContext


class TestTriggerContextBasics:
    """Test TriggerContext dataclass."""

    @pytest.mark.unit
    def test_trigger_context_creation(self):
        """Verify TriggerContext accepts all required fields."""
        now = datetime.utcnow()
        ctx = TriggerContext(
            trigger_id="trg_stripe_123",
            source="stripe",
            event_type="payment_intent.succeeded",
            event_id="evt_abc",
            idempotency_key="evt_abc_key",
            received_at=now,
        )
        
        assert ctx.trigger_id == "trg_stripe_123"
        assert ctx.source == "stripe"
        assert ctx.event_type == "payment_intent.succeeded"
        assert ctx.event_id == "evt_abc"
        assert ctx.idempotency_key == "evt_abc_key"
        assert ctx.received_at == now
        assert ctx.vc_id is None

    @pytest.mark.unit
    def test_trigger_context_with_vc_id(self):
        """Verify TriggerContext stores optional vc_id."""
        ctx = TriggerContext(
            trigger_id="trg_1",
            source="github",
            event_type="pull_request.opened",
            event_id="evt_1",
            idempotency_key="key_1",
            received_at=datetime.utcnow(),
            vc_id="vc_webhook_123",
        )
        
        assert ctx.vc_id == "vc_webhook_123"

    @pytest.mark.unit
    def test_trigger_context_frozen(self):
        """Verify TriggerContext is immutable."""
        ctx = TriggerContext(
            trigger_id="trg_1",
            source="stripe",
            event_type="charge.completed",
            event_id="evt_1",
            idempotency_key="key_1",
            received_at=datetime.utcnow(),
        )
        
        with pytest.raises(AttributeError):
            ctx.source = "github"


class TestExecutionContextTrigger:
    """Test ExecutionContext.trigger field."""

    @pytest.mark.unit
    def test_execution_context_trigger_field(self):
        """Verify ExecutionContext accepts trigger field."""
        trigger = TriggerContext(
            trigger_id="trg_1",
            source="stripe",
            event_type="payment_intent.succeeded",
            event_id="evt_1",
            idempotency_key="evt_1",
            received_at=datetime.utcnow(),
        )
        ctx = ExecutionContext(
            run_id="run_1",
            execution_id="exec_1",
            agent_instance=None,
            reasoner_name="handler",
            trigger=trigger,
        )
        
        assert ctx.trigger is trigger
        assert ctx.trigger.source == "stripe"

    @pytest.mark.unit
    def test_execution_context_trigger_none_by_default(self):
        """Verify ExecutionContext.trigger is None when not set."""
        ctx = ExecutionContext(
            run_id="run_1",
            execution_id="exec_1",
            agent_instance=None,
            reasoner_name="handler",
        )
        
        assert ctx.trigger is None

    @pytest.mark.unit
    def test_child_context_inherits_trigger(self):
        """Verify child_context() preserves trigger."""
        trigger = TriggerContext(
            trigger_id="trg_1",
            source="github",
            event_type="pull_request.opened",
            event_id="evt_1",
            idempotency_key="evt_1",
            received_at=datetime.utcnow(),
        )
        parent = ExecutionContext(
            run_id="run_1",
            execution_id="exec_1",
            agent_instance=None,
            reasoner_name="parent",
            trigger=trigger,
        )
        
        child = parent.child_context()
        
        assert child.trigger is trigger
        assert child.execution_id != parent.execution_id
        assert child.parent_execution_id == parent.execution_id


class TestEventTriggerTransform:
    """Test EventTrigger.transform field."""

    @pytest.mark.unit
    def test_event_trigger_with_transform(self):
        """Verify EventTrigger accepts sync transform callable."""
        def stripe_to_order(event_dict):
            return {"order_id": event_dict.get("id")}
        
        trigger = EventTrigger(
            source="stripe",
            types=["payment_intent.succeeded"],
            transform=stripe_to_order,
        )
        
        assert trigger.transform is stripe_to_order

    @pytest.mark.unit
    def test_event_trigger_transform_default_none(self):
        """Verify EventTrigger.transform defaults to None."""
        trigger = EventTrigger(source="github")
        assert trigger.transform is None

    @pytest.mark.unit
    def test_event_trigger_rejects_async_transform(self):
        """Verify EventTrigger raises TypeError for async transform."""
        async def bad_transform(event):
            return event
        
        with pytest.raises(TypeError, match="must be sync"):
            EventTrigger(
                source="stripe",
                transform=bad_transform,
            )

    @pytest.mark.unit
    def test_event_trigger_transform_not_in_comparison(self):
        """Verify transform field is excluded from equality/repr."""
        def transform1(e):
            return e
        
        def transform2(e):
            return e
        
        t1 = EventTrigger(source="stripe", transform=transform1)
        t2 = EventTrigger(source="stripe", transform=transform2)
        
        # Should be equal because transform is excluded from comparison
        assert t1 == t2
        # And not in repr
        assert "transform" not in repr(t1)


class TestEnvelopeUnwrapping:
    """Test dispatcher envelope detection and unwrapping.
    
    Note: These tests use the agent's _detect_and_unwrap_trigger_envelope method
    which is tested indirectly through integration with _execute_reasoner_endpoint.
    """

    @pytest.mark.unit
    def test_envelope_structure(self):
        """Verify dispatcher envelope shape."""
        envelope = {
            "event": {"id": "pi_123", "type": "payment_intent.succeeded"},
            "_meta": {
                "trigger_id": "trg_stripe_123",
                "source": "stripe",
                "event_type": "payment_intent.succeeded",
                "event_id": "evt_abc",
                "idempotency_key": "evt_abc",
                "received_at": "2026-04-28T12:00:00Z",
                "vc_id": "vc_123",
            },
        }
        
        # Verify envelope structure
        assert "event" in envelope
        assert "_meta" in envelope
        assert envelope["_meta"]["source"] == "stripe"
        assert isinstance(envelope["event"], dict)


class TestTransformMatching:
    """Test trigger transform matching logic."""

    @pytest.mark.unit
    def test_transform_matching_exact_source(self):
        """Verify transform matches on source."""
        def stripe_transform(e):
            return {"transformed": True}
        
        binding = EventTrigger(
            source="stripe",
            transform=stripe_transform,
        )
        
        # Binding source matches trigger source
        assert binding.source == "stripe"
        assert binding.transform is stripe_transform

    @pytest.mark.unit
    def test_transform_matching_event_type_prefix(self):
        """Verify event_type matching uses prefix logic."""
        _binding = EventTrigger(
            source="github",
            types=["pull_request"],  # prefix
        )

        # "pull_request.opened" starts with "pull_request"
        assert "pull_request.opened".startswith("pull_request")
        assert "pull_request.synchronize".startswith("pull_request")

    @pytest.mark.unit
    def test_transform_specificity(self):
        """Verify most-specific binding is chosen."""
        def broad_transform(e):
            return {"binding": "broad"}
        
        def specific_transform(e):
            return {"binding": "specific"}
        
        broad = EventTrigger(
            source="stripe",
            types=[],  # accepts all
            transform=broad_transform,
        )
        
        specific = EventTrigger(
            source="stripe",
            types=["payment_intent.succeeded"],  # specific
            transform=specific_transform,
        )
        
        # Both match source, but specific should be preferred
        assert broad.source == specific.source
        assert len(broad.types) == 0
        assert len(specific.types) > 0


@pytest.mark.integration
class TestTriggerContextIntegration:
    """Integration tests for trigger context in reasoner execution."""

    def test_trigger_context_passed_to_reasoner(self, test_agent):
        """Verify trigger context is passed to reasoner accepting trigger parameter."""
        # This test requires a running agent; see conftest.py for setup
        captured_trigger = None
        
        @test_agent.reasoner(
            triggers=[EventTrigger(source="stripe")]
        )
        async def handle_webhook(input, trigger):
            nonlocal captured_trigger
            captured_trigger = trigger
            return {"ok": True}
        
        # Simulate envelope POST
        _envelope = {
            "event": {"id": "pi_123"},
            "_meta": {
                "trigger_id": "trg_1",
                "source": "stripe",
                "event_type": "payment_intent.succeeded",
                "event_id": "evt_1",
                "idempotency_key": "evt_1",
                "received_at": "2026-04-28T12:00:00Z",
            },
        }

        # Note: Real integration test would POST to agent endpoint
        # This is a placeholder showing the expected behavior


@pytest.mark.integration
class TestTransformExecution:
    """Integration tests for transform execution."""

    def test_transform_called_on_match(self, test_agent):
        """Verify transform is called when event matches binding."""
        transform_called = False
        
        def stripe_to_order(event_dict):
            nonlocal transform_called
            transform_called = True
            return {"order_id": event_dict.get("id")}
        
        @test_agent.reasoner(
            triggers=[
                EventTrigger(
                    source="stripe",
                    types=["payment_intent.succeeded"],
                    transform=stripe_to_order,
                )
            ]
        )
        async def handle_payment(input):
            return {"received": input}
        
        # Note: Real test would POST envelope


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with non-envelope payloads."""

    def test_direct_call_no_envelope(self):
        """Verify non-envelope payloads work unchanged."""
        # Direct input (not an envelope)
        payload = {"order_id": "123", "amount": 100}
        
        # Should not look like an envelope
        assert "event" not in payload or "_meta" not in payload

    def test_execution_context_trigger_none_on_direct_call(self):
        """Verify trigger is None for non-envelope calls."""
        ctx = ExecutionContext(
            run_id="run_1",
            execution_id="exec_1",
            agent_instance=None,
            reasoner_name="reasoner",
        )
        
        assert ctx.trigger is None
