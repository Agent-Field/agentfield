import time
from agentfield.execution_state import (
    ExecutionBatch,
    ExecutionState,
    ExecutionStatus,
)


def test_execution_state_lifecycle_and_properties():
    st = ExecutionState(execution_id="e1", target="node.skill", input_data={})
    assert st.status == ExecutionStatus.QUEUED
    assert st.is_active is True
    assert st.is_terminal is False

    st.update_status(ExecutionStatus.RUNNING)
    assert st.is_active is True

    st.set_result({"ok": True})
    assert st.status == ExecutionStatus.SUCCEEDED
    assert st.is_successful is True
    assert st.is_terminal is True


def test_execution_state_waiting_is_active_non_terminal():
    st = ExecutionState(execution_id="ew1", target="node.skill", input_data={})
    st.update_status(ExecutionStatus.WAITING)
    assert st.is_active is True
    assert st.is_terminal is False

    st.update_status(ExecutionStatus.RUNNING)
    assert st.metrics.start_time is not None


def test_execution_state_failure_and_cancel():
    st = ExecutionState(execution_id="e2", target="t", input_data={})
    st.set_error("boom")
    assert st.status == ExecutionStatus.FAILED
    assert st.error_message == "boom"
    assert st.is_terminal is True

    st = ExecutionState(execution_id="e3", target="t", input_data={})
    st.cancel("nope")
    assert st.status == ExecutionStatus.CANCELLED
    assert st.is_cancelled is True
    assert st.is_terminal is True


def test_polling_and_timeout():
    st = ExecutionState(execution_id="e4", target="t", input_data={}, timeout=0.1)
    assert st.should_poll in (True, False)  # property is computed
    st.record_poll_attempt(success=True, duration=0.01)
    st.update_poll_interval(0.05)
    assert st.current_poll_interval == 0.05
    time.sleep(0.12)
    assert st.is_overdue is True
    st.timeout_execution()
    assert st.status == ExecutionStatus.TIMEOUT


class _StubPauseClock:
    """Minimal stand-in for ``agent_pause.PauseClock`` for unit tests."""

    def __init__(self, paused: float) -> None:
        self._paused = paused

    def total_paused(self) -> float:
        return self._paused


def test_is_overdue_subtracts_attached_pause_clock_from_age():
    """Regression: the polling task's overdue check must respect a parent's
    pause clock so a long-paused child is not flipped TIMEOUT at the
    wallclock budget. Mirrors the production scenario where github-buddy's
    ``app.call`` to swe-af.build sat in ``waiting`` for hours and the
    polling task pre-empted the pause-aware wait_for_result."""

    st = ExecutionState(execution_id="e-pause", target="t", input_data={}, timeout=0.1)
    time.sleep(0.12)
    # Wallclock alone says overdue.
    assert st.is_overdue is True
    # Attach a clock claiming most of the wallclock was paused; subtracting
    # it pulls the active age back under the budget so the polling task
    # should NOT mark this execution TIMEOUT.
    st._pause_clock = _StubPauseClock(paused=0.20)
    assert st.is_overdue is False


def test_is_overdue_pause_clock_failure_falls_back_to_wallclock():
    """If the attached clock raises (defensive), we fall back to wallclock
    so executions still time out eventually rather than hanging forever."""

    class _BrokenClock:
        def total_paused(self):
            raise RuntimeError("clock broken")

    st = ExecutionState(execution_id="e-broken", target="t", input_data={}, timeout=0.1)
    st._pause_clock = _BrokenClock()
    time.sleep(0.12)
    assert st.is_overdue is True


def test_is_overdue_no_pause_clock_matches_legacy_behavior():
    """Without a pause clock the property must be unchanged from before."""

    st = ExecutionState(execution_id="e-legacy", target="t", input_data={}, timeout=0.1)
    assert st._pause_clock is None
    time.sleep(0.12)
    assert st.is_overdue is True


def test_execution_metrics_computations():
    metrics = ExecutionState(execution_id="e5", target="t", input_data={}).metrics
    metrics.submit_time = 10.0
    metrics.start_time = 12.0
    metrics.end_time = 20.0
    metrics.poll_count = 3
    metrics.total_poll_time = 0.6

    assert metrics.total_duration == 10.0
    assert metrics.execution_duration == 8.0
    assert metrics.queue_duration == 2.0
    assert metrics.average_poll_interval == 0.3


def test_execution_state_post_init_initializes_defaults():
    st = ExecutionState(
        execution_id="e6",
        target="t",
        input_data={},
        metrics=None,
        next_poll_time=0,
    )
    assert st.metrics is not None
    assert st.next_poll_time > 0


def test_execution_state_str_and_repr():
    st = ExecutionState(execution_id="abcdef12", target="t", input_data={})
    str_val = str(st)
    repr_val = repr(st)
    assert "ExecutionState" in repr_val
    assert "target=t" in str_val


def test_execution_batch_helpers():
    st1 = ExecutionState(execution_id="batch1", target="a", input_data={})
    st2 = ExecutionState(execution_id="batch2", target="b", input_data={})
    batch = ExecutionBatch([st1, st2])

    assert batch.size == 2
    assert set(batch.execution_ids) == {"batch1", "batch2"}
    assert len(batch.active_executions) == 2

    st1.set_result({"ok": True})
    st2.cancel("cancelled")
    assert len(batch.completed_executions) == 2

    batch.add_execution(st1)  # duplicate ignored
    assert batch.size == 2
