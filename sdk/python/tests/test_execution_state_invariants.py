"""
Behavioral invariant tests for ExecutionState and related classes.

These tests verify structural properties that must always hold regardless
of implementation changes, protecting against AI regressions that break
the execution lifecycle contract.
"""
from __future__ import annotations

import time



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(execution_id: str = "exec-001", target: str = "node.reasoner",
                status=None):
    """Create a minimal ExecutionState with optional initial status."""
    from agentfield.execution_state import ExecutionState

    state = ExecutionState(
        execution_id=execution_id,
        target=target,
        input_data={"input": "data"},
    )
    if status is not None:
        state.status = status
    return state


# ---------------------------------------------------------------------------
# 1. Status normalization idempotency
# ---------------------------------------------------------------------------

class TestStatusNormalizationIdempotency:
    """normalize(normalize(x)) == normalize(x) for all canonical status values."""

    def _normalize(self, status: str) -> str:
        from agentfield.status import normalize_status
        return normalize_status(status)

    def test_invariant_normalize_is_idempotent_for_all_canonical_values(self):
        """normalize(normalize(x)) == normalize(x) for every canonical status."""
        canonical = ["pending", "queued", "waiting", "running",
                     "succeeded", "failed", "cancelled", "timeout", "unknown"]
        for status in canonical:
            once = self._normalize(status)
            twice = self._normalize(once)
            assert once == twice, (
                f"INVARIANT VIOLATION: normalize is not idempotent for '{status}'. "
                f"normalize('{status}')='{once}', normalize('{once}')='{twice}'"
            )

    def test_invariant_normalize_is_idempotent_for_aliases(self):
        """normalize(normalize(alias)) == normalize(alias) for all known aliases."""
        aliases = [
            "success", "error", "completed", "done", "ok",
            "canceled", "timed_out", "in_progress", "processing",
            "RUNNING", "FAILED", "Success", "Error",
        ]
        for alias in aliases:
            once = self._normalize(alias)
            twice = self._normalize(once)
            assert once == twice, (
                f"INVARIANT VIOLATION: normalize is not idempotent for alias '{alias}'. "
                f"normalize('{alias}')='{once}', normalize('{once}')='{twice}'"
            )


# ---------------------------------------------------------------------------
# 2. Terminal state detection consistency
# ---------------------------------------------------------------------------

class TestTerminalStateConsistency:
    """is_terminal must agree with the TERMINAL_STATUSES set."""

    TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "timeout"}
    ACTIVE_STATUSES = {"pending", "queued", "waiting", "running"}

    def test_invariant_is_terminal_true_for_all_terminal_statuses(self):
        """is_terminal must be True for every terminal status."""
        from agentfield.execution_state import ExecutionStatus

        terminal_enum = {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT,
        }
        for status in terminal_enum:
            state = _make_state()
            state.status = status
            assert state.is_terminal is True, (
                f"INVARIANT VIOLATION: is_terminal is False for terminal status {status.value}. "
                "is_terminal must agree with TERMINAL_STATUSES."
            )

    def test_invariant_is_terminal_false_for_all_active_statuses(self):
        """is_terminal must be False for every active status."""
        from agentfield.execution_state import ExecutionStatus

        active_enum = {
            ExecutionStatus.PENDING,
            ExecutionStatus.QUEUED,
            ExecutionStatus.WAITING,
            ExecutionStatus.RUNNING,
        }
        for status in active_enum:
            state = _make_state()
            state.status = status
            assert state.is_terminal is False, (
                f"INVARIANT VIOLATION: is_terminal is True for active status {status.value}. "
                "Active statuses must NOT be terminal."
            )

    def test_invariant_terminal_and_active_are_disjoint(self):
        """No status can be both is_terminal=True and is_active=True."""
        from agentfield.execution_state import ExecutionStatus

        for status in ExecutionStatus:
            state = _make_state()
            state.status = status
            assert not (state.is_terminal and state.is_active), (
                f"INVARIANT VIOLATION: Status {status.value} is both terminal AND active. "
                "These properties must be mutually exclusive."
            )


# ---------------------------------------------------------------------------
# 3. Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerializationRoundtrip:
    """ExecutionState.to_dict() must preserve all observable state."""

    def test_invariant_to_dict_contains_all_required_keys(self):
        """to_dict() must include every field the control plane contracts on."""
        state = _make_state()
        d = state.to_dict()

        required_keys = [
            "execution_id", "target", "status", "priority",
            "created_at", "updated_at",
            "result", "error_message", "error_details",
            "workflow_id", "parent_execution_id", "session_id", "actor_id",
            "timeout", "is_terminal", "is_active", "is_successful", "is_cancelled",
            "metrics",
        ]
        for key in required_keys:
            assert key in d, (
                f"INVARIANT VIOLATION: to_dict() missing required key '{key}'. "
                f"Present keys: {set(d.keys())}"
            )

    def test_invariant_to_dict_preserves_execution_id(self):
        """Roundtrip via to_dict() must preserve execution_id exactly."""
        state = _make_state(execution_id="exec-roundtrip-123")
        d = state.to_dict()
        assert d["execution_id"] == "exec-roundtrip-123", (
            f"INVARIANT VIOLATION: to_dict() changed execution_id. "
            f"Expected 'exec-roundtrip-123', got '{d['execution_id']}'"
        )

    def test_invariant_to_dict_status_is_string_value(self):
        """to_dict() must serialize status as its string value, not enum name."""
        from agentfield.execution_state import ExecutionStatus

        state = _make_state()
        state.status = ExecutionStatus.RUNNING
        d = state.to_dict()

        assert d["status"] == "running", (
            f"INVARIANT VIOLATION: to_dict() serialized status as '{d['status']}' "
            "instead of 'running'. Must use .value not .name."
        )

    def test_invariant_to_dict_is_terminal_agrees_with_status(self):
        """to_dict()['is_terminal'] must agree with state.is_terminal."""
        from agentfield.execution_state import ExecutionStatus

        for status in ExecutionStatus:
            state = _make_state()
            state.status = status
            d = state.to_dict()
            assert d["is_terminal"] == state.is_terminal, (
                f"INVARIANT VIOLATION: to_dict()['is_terminal']={d['is_terminal']} "
                f"disagrees with state.is_terminal={state.is_terminal} for status {status.value}."
            )

    def test_invariant_to_dict_metrics_contains_poll_count(self):
        """to_dict()['metrics'] must contain 'poll_count'."""
        state = _make_state()
        state.record_poll_attempt(success=True, duration=0.1)
        state.record_poll_attempt(success=True, duration=0.2)
        d = state.to_dict()

        assert "poll_count" in d["metrics"], (
            "INVARIANT VIOLATION: to_dict()['metrics'] missing 'poll_count'."
        )
        assert d["metrics"]["poll_count"] == 2, (
            f"INVARIANT VIOLATION: After 2 polls, to_dict()['metrics']['poll_count'] "
            f"== {d['metrics']['poll_count']} instead of 2."
        )


# ---------------------------------------------------------------------------
# 4. Metrics monotonicity
# ---------------------------------------------------------------------------

class TestMetricsMonotonicity:
    """poll_count must only increase, never decrease."""

    def test_invariant_poll_count_is_monotonically_increasing(self):
        """poll_count must strictly increase with each add_poll call."""
        from agentfield.execution_state import ExecutionMetrics

        metrics = ExecutionMetrics()
        prev_count = metrics.poll_count

        for i in range(20):
            metrics.add_poll(0.01 * i)
            assert metrics.poll_count > prev_count, (
                f"INVARIANT VIOLATION: poll_count did not increase after add_poll(). "
                f"Before: {prev_count}, After: {metrics.poll_count} (iteration {i})."
            )
            prev_count = metrics.poll_count

    def test_invariant_poll_count_starts_at_zero(self):
        """ExecutionMetrics must start with poll_count = 0."""
        from agentfield.execution_state import ExecutionMetrics

        metrics = ExecutionMetrics()
        assert metrics.poll_count == 0, (
            f"INVARIANT VIOLATION: ExecutionMetrics.poll_count must start at 0, "
            f"got {metrics.poll_count}."
        )

    def test_invariant_record_poll_attempt_increments_poll_count(self):
        """ExecutionState.record_poll_attempt() must increment metrics.poll_count."""
        state = _make_state()
        initial_count = state.metrics.poll_count

        state.record_poll_attempt(success=True, duration=0.05)
        assert state.metrics.poll_count == initial_count + 1, (
            f"INVARIANT VIOLATION: record_poll_attempt() did not increment poll_count. "
            f"Expected {initial_count + 1}, got {state.metrics.poll_count}."
        )

    def test_invariant_poll_count_never_decreases_on_failure(self):
        """Failed poll attempts must still increment poll_count (not decrement)."""
        state = _make_state()

        for _ in range(5):
            state.record_poll_attempt(success=True, duration=0.01)

        count_after_success = state.metrics.poll_count

        state.record_poll_attempt(success=False, duration=0.01)
        assert state.metrics.poll_count > count_after_success, (
            f"INVARIANT VIOLATION: Failed poll_attempt decreased poll_count. "
            f"Before: {count_after_success}, After: {state.metrics.poll_count}."
        )


# ---------------------------------------------------------------------------
# 5. Timestamp ordering
# ---------------------------------------------------------------------------

class TestTimestampOrdering:
    """When both are set, started_at <= completed_at must always hold."""

    def test_invariant_start_time_before_end_time_after_transitions(self):
        """
        After transitioning QUEUED → RUNNING → SUCCEEDED, start_time must be
        <= end_time in the metrics.
        """
        from agentfield.execution_state import ExecutionStatus

        state = _make_state()
        # Transition to running
        state.update_status(ExecutionStatus.RUNNING)
        assert state.metrics.start_time is not None, (
            "INVARIANT VIOLATION: metrics.start_time not set after RUNNING transition."
        )

        # Small sleep to ensure end_time > start_time
        time.sleep(0.001)

        # Transition to terminal
        state.update_status(ExecutionStatus.SUCCEEDED)
        assert state.metrics.end_time is not None, (
            "INVARIANT VIOLATION: metrics.end_time not set after SUCCEEDED transition."
        )

        assert state.metrics.start_time <= state.metrics.end_time, (
            f"INVARIANT VIOLATION: start_time ({state.metrics.start_time}) > "
            f"end_time ({state.metrics.end_time}). Temporal ordering must hold."
        )

    def test_invariant_created_at_before_updated_at_after_status_change(self):
        """created_at must be <= updated_at after any status update."""
        from agentfield.execution_state import ExecutionStatus

        state = _make_state()

        time.sleep(0.001)
        state.update_status(ExecutionStatus.RUNNING)

        assert state.created_at <= state.updated_at, (
            f"INVARIANT VIOLATION: created_at ({state.created_at}) > "
            f"updated_at ({state.updated_at}). Temporal ordering violated."
        )

    def test_invariant_set_result_sets_end_time(self):
        """set_result() must record an end_time in metrics."""
        state = _make_state()
        state.set_result({"output": "value"})

        assert state.metrics.end_time is not None, (
            "INVARIANT VIOLATION: set_result() did not set metrics.end_time."
        )

    def test_invariant_set_error_sets_end_time(self):
        """set_error() must record an end_time in metrics."""
        state = _make_state()
        state.set_error("something went wrong")

        assert state.metrics.end_time is not None, (
            "INVARIANT VIOLATION: set_error() did not set metrics.end_time."
        )
