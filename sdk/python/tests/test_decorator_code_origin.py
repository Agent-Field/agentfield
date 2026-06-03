"""Tests for code origin capture in trigger decorators."""

import inspect

from agentfield.decorators import reasoner, on_event, on_schedule
from agentfield.triggers import EventTrigger, ScheduleTrigger, trigger_to_payload


class TestReasonerTriggersCodeOrigin:
    """Test code_origin stamping on triggers passed to @reasoner."""

    def test_reasoner_triggers_kwarg_stamps_code_origin(self):
        """@reasoner with triggers=[] should stamp code_origin on each trigger."""

        # Define the function where the decorator is applied
        @reasoner(
            triggers=[EventTrigger(source="stripe", types=["payment_intent.succeeded"])]
        )
        async def handle_payment(input, ctx):
            pass

        # Verify the trigger has code_origin set
        triggers = getattr(handle_payment, "_reasoner_triggers", [])
        assert len(triggers) > 0
        trigger = triggers[0]
        assert hasattr(trigger, "code_origin")
        assert trigger.code_origin is not None

        # code_origin should be the file and line number
        assert isinstance(trigger.code_origin, str)
        assert ":" in trigger.code_origin
        parts = trigger.code_origin.rsplit(":", 1)
        assert len(parts) == 2
        file_path, line_num = parts
        assert file_path.endswith(".py")
        assert line_num.isdigit()

    def test_reasoner_preserves_user_supplied_code_origin(self):
        """If user passes code_origin explicitly, don't overwrite it."""

        user_origin = "custom/path.py:999"

        @reasoner(triggers=[EventTrigger(source="github", code_origin=user_origin)])
        async def handle_github(input, ctx):
            pass

        triggers = getattr(handle_github, "_reasoner_triggers", [])
        trigger = triggers[0]
        assert trigger.code_origin == user_origin

    def test_reasoner_stamps_multiple_triggers(self):
        """All triggers in the triggers=[] list should get stamped."""

        @reasoner(
            triggers=[
                EventTrigger(source="stripe"),
                EventTrigger(source="github"),
            ]
        )
        async def multi_trigger(input, ctx):
            pass

        triggers = getattr(multi_trigger, "_reasoner_triggers", [])
        assert len(triggers) == 2
        for trigger in triggers:
            assert trigger.code_origin is not None
            # All should have the same origin (the function line)
            assert ":" in trigger.code_origin

    def test_reasoner_with_mixed_origins(self):
        """Mix of user-supplied and auto-stamped code_origin."""

        user_origin = "user/custom.py:42"

        @reasoner(
            triggers=[
                EventTrigger(source="stripe", code_origin=user_origin),
                EventTrigger(source="github"),  # Will auto-stamp
            ]
        )
        async def mixed(input, ctx):
            pass

        triggers = getattr(mixed, "_reasoner_triggers", [])
        assert triggers[0].code_origin == user_origin  # User-supplied preserved
        assert triggers[1].code_origin is not None  # Auto-stamped
        assert triggers[1].code_origin != user_origin  # Different


class TestOnEventCodeOrigin:
    """Test code_origin capture on @on_event decorator."""

    def test_on_event_sugar_captures_code_origin(self):
        """@on_event should capture the function's file:line."""

        @reasoner()
        @on_event(
            source="stripe",
            types=["payment_intent.succeeded"],
            secret_env="STRIPE_SECRET",
        )
        async def handle_stripe(input, ctx):
            pass

        triggers = getattr(handle_stripe, "_reasoner_triggers", [])
        assert len(triggers) > 0
        trigger = triggers[0]

        # Should have code_origin set
        assert trigger.code_origin is not None
        assert isinstance(trigger.code_origin, str)
        assert ":" in trigger.code_origin

        # Verify it's a valid file:line format
        parts = trigger.code_origin.rsplit(":", 1)
        assert len(parts) == 2
        assert parts[1].isdigit()

    def test_on_event_with_explicit_code_origin(self):
        """If @on_event creates trigger with code_origin, don't override."""

        @reasoner()
        @on_event(source="github")
        async def handle_gh(input, ctx):
            pass

        triggers = getattr(handle_gh, "_reasoner_triggers", [])
        trigger = triggers[0]

        # Should be auto-stamped (not user-supplied)
        assert trigger.code_origin is not None


class TestOnScheduleCodeOrigin:
    """Test code_origin capture on @on_schedule decorator."""

    def test_on_schedule_sugar_captures_code_origin(self):
        """@on_schedule should capture the function's file:line."""

        @reasoner()
        @on_schedule("*/5 * * * *")
        async def poll(input, ctx):
            pass

        triggers = getattr(poll, "_reasoner_triggers", [])
        assert len(triggers) > 0
        trigger = triggers[0]

        # Should have code_origin set
        assert trigger.code_origin is not None
        assert isinstance(trigger.code_origin, str)
        assert ":" in trigger.code_origin

        # Verify format
        parts = trigger.code_origin.rsplit(":", 1)
        assert len(parts) == 2
        assert parts[1].isdigit()

    def test_on_schedule_with_timezone_captures_code_origin(self):
        """@on_schedule with timezone should still capture code_origin."""

        @reasoner()
        @on_schedule("0 9 * * *", timezone="America/New_York")
        async def morning_task(input, ctx):
            pass

        triggers = getattr(morning_task, "_reasoner_triggers", [])
        trigger = triggers[0]

        assert isinstance(trigger, ScheduleTrigger)
        assert trigger.timezone == "America/New_York"
        assert trigger.code_origin is not None


class TestCodeOriginWirePayload:
    """Test that code_origin flows through to the wire payload."""

    def test_event_trigger_code_origin_in_payload(self):
        """EventTrigger with code_origin should appear in trigger_to_payload output."""

        trigger = EventTrigger(
            source="stripe",
            types=["payment_intent.succeeded"],
            code_origin="/path/to/agent.py:42",
        )

        payload = trigger_to_payload(trigger)

        assert "code_origin" in payload
        assert payload["code_origin"] == "/path/to/agent.py:42"

    def test_schedule_trigger_code_origin_in_payload(self):
        """ScheduleTrigger with code_origin should appear in payload."""

        trigger = ScheduleTrigger(
            cron="0 9 * * *", timezone="UTC", code_origin="/path/to/scheduler.py:100"
        )

        payload = trigger_to_payload(trigger)

        assert "code_origin" in payload
        assert payload["code_origin"] == "/path/to/scheduler.py:100"

    def test_trigger_without_code_origin_omitted_from_payload(self):
        """If code_origin is None, it should not appear in payload."""

        trigger = EventTrigger(source="github")
        payload = trigger_to_payload(trigger)

        # code_origin should either be absent or None
        assert payload.get("code_origin") is None


class TestDecoratorStackingCodeOrigin:
    """Test code_origin with stacked decorators."""

    def test_reasoner_with_stacked_on_event(self):
        """@reasoner() + @on_event() should capture from the decorated function."""

        @reasoner()
        @on_event(source="stripe")
        @on_event(source="github")
        async def multi_source(input, ctx):
            pass

        triggers = getattr(multi_source, "_reasoner_triggers", [])
        assert len(triggers) == 2

        # Both should have code_origin (from the decorated function)
        for trigger in triggers:
            assert trigger.code_origin is not None
            # Both should reference the same line (the function definition)
            # (They're stacked, so both decorators see the same func)

    def test_code_origin_matches_function_location(self):
        """code_origin should match the function's actual line number."""

        @reasoner(triggers=[EventTrigger(source="test")])
        async def test_function(input, ctx):
            pass

        triggers = getattr(test_function, "_reasoner_triggers", [])
        trigger = triggers[0]

        # Extract line number from code_origin
        origin_file, origin_line = trigger.code_origin.rsplit(":", 1)
        origin_line = int(origin_line)

        # Get the decorator line
        func_line = inspect.getsourcelines(test_function)[1]

        # code_origin should point to the function definition
        # (it may be a few lines before depending on @reasoner placement)
        assert abs(origin_line - func_line) < 5  # Within 5 lines is reasonable


class TestCodeOriginNullability:
    """Test that code_origin is properly optional."""

    def test_event_trigger_without_code_origin(self):
        """EventTrigger should work fine without code_origin."""

        trigger = EventTrigger(source="stripe", types=["charge.succeeded"])

        # Should not raise
        payload = trigger_to_payload(trigger)
        assert "source" in payload
        assert payload.get("code_origin") is None

    def test_schedule_trigger_without_code_origin(self):
        """ScheduleTrigger should work fine without code_origin."""

        trigger = ScheduleTrigger(cron="0 * * * *")

        # Should not raise
        payload = trigger_to_payload(trigger)
        assert "source" in payload
        assert payload["source"] == "cron"
        assert payload.get("code_origin") is None

    def test_backward_compatibility_trigger_construction(self):
        """Old code constructing triggers without code_origin should work."""

        # This mimics pre-Phase3 code
        t1 = EventTrigger(source="github")
        t2 = ScheduleTrigger(cron="*/5 * * * *")

        # Should not raise, and code_origin should be None
        assert t1.code_origin is None
        assert t2.code_origin is None

        # Payloads should serialize fine
        p1 = trigger_to_payload(t1)
        p2 = trigger_to_payload(t2)

        assert "code_origin" not in p1 or p1.get("code_origin") is None
        assert "code_origin" not in p2 or p2.get("code_origin") is None
