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
