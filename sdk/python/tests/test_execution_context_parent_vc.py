"""
Unit tests for parent_vc_id support in ExecutionContext.
Tests the round-trip of header reading, field setting, and header emission.
"""

import pytest
from agentfield.execution_context import ExecutionContext, _PARENT_VC_HEADER


@pytest.mark.unit
def test_execution_context_parent_vc_id_field():
    """Verify that ExecutionContext accepts and stores parent_vc_id field."""
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        parent_vc_id="vc_trigger_webhook_stripe_123",
        run_id="run-1",
    )

    assert ctx.parent_vc_id == "vc_trigger_webhook_stripe_123"


@pytest.mark.unit
def test_to_headers_includes_parent_vc_id():
    """Verify that to_headers() includes X-Parent-VC-ID when parent_vc_id is set."""
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        parent_vc_id="vc_trigger_webhook_123",
        run_id="run-1",
    )

    headers = ctx.to_headers()

    assert headers["X-Parent-VC-ID"] == "vc_trigger_webhook_123"
    assert _PARENT_VC_HEADER in headers


@pytest.mark.unit
def test_to_headers_omits_parent_vc_id_when_none():
    """Verify that to_headers() omits X-Parent-VC-ID when parent_vc_id is None."""
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        run_id="run-1",
    )

    headers = ctx.to_headers()

    assert "X-Parent-VC-ID" not in headers


@pytest.mark.unit
def test_from_request_reads_parent_vc_id():
    """Verify that from_request() reads X-Parent-VC-ID header and sets parent_vc_id field."""
    # Mock FastAPI request headers
    class MockHeaders:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key.lower()) or self._data.get(key) or default

    class MockRequest:
        def __init__(self, headers_dict):
            self.headers = MockHeaders(headers_dict)

    request = MockRequest({
        "X-Workflow-ID": "wf-1",
        "X-Execution-ID": "exec-1",
        "X-Parent-VC-ID": "vc_trigger_webhook_456",
        "X-Run-ID": "run-1",
    })

    ctx = ExecutionContext.from_request(request, agent_node_id="node-1")

    assert ctx.parent_vc_id == "vc_trigger_webhook_456"


@pytest.mark.unit
def test_from_request_handles_missing_parent_vc_id():
    """Verify that from_request() handles missing X-Parent-VC-ID gracefully."""
    class MockHeaders:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key.lower()) or self._data.get(key) or default

    class MockRequest:
        def __init__(self, headers_dict):
            self.headers = MockHeaders(headers_dict)

    request = MockRequest({
        "X-Workflow-ID": "wf-1",
        "X-Execution-ID": "exec-1",
        "X-Run-ID": "run-1",
    })

    ctx = ExecutionContext.from_request(request, agent_node_id="node-1")

    assert ctx.parent_vc_id is None


@pytest.mark.unit
def test_child_context_preserves_parent_vc_id():
    """Verify that child_context() preserves parent_vc_id from parent."""
    parent = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="parent",
        parent_vc_id="vc_trigger_webhook_789",
        run_id="run-1",
    )

    child = parent.child_context()

    assert child.parent_vc_id == "vc_trigger_webhook_789"
    assert child.parent_execution_id == parent.execution_id
    assert child.execution_id != parent.execution_id


@pytest.mark.unit
def test_to_log_attributes_includes_parent_vc_id():
    """Verify that to_log_attributes() includes parent_vc_id when set."""
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        parent_vc_id="vc_trigger_github_push_001",
        run_id="run-1",
    )

    attributes = ctx.to_log_attributes()

    assert attributes["parent_vc_id"] == "vc_trigger_github_push_001"


@pytest.mark.unit
def test_to_log_attributes_omits_parent_vc_id_when_none():
    """Verify that to_log_attributes() omits parent_vc_id when None."""
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        run_id="run-1",
    )

    attributes = ctx.to_log_attributes()

    assert "parent_vc_id" not in attributes


@pytest.mark.unit
def test_parent_vc_id_round_trip_headers():
    """Integration test: verify parent_vc_id survives round-trip through headers."""
    original_vc_id = "vc_trigger_cron_hourly_2024"

    # Create context with parent_vc_id
    ctx1 = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        parent_vc_id=original_vc_id,
        run_id="run-1",
    )

    # Get headers
    headers = ctx1.to_headers()
    assert headers.get("X-Parent-VC-ID") == original_vc_id

    # Simulate reading headers from a request
    class MockHeaders:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=None):
            return self._data.get(key.lower()) or self._data.get(key) or default

    class MockRequest:
        def __init__(self, headers_dict):
            self.headers = MockHeaders(headers_dict)

    request = MockRequest(headers)
    ctx2 = ExecutionContext.from_request(request, agent_node_id="node-1")

    # Verify parent_vc_id is preserved
    assert ctx2.parent_vc_id == original_vc_id
