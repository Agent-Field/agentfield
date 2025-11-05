from agentfield.types import (
    ExecutionHeaders,
    AgentStatus,
    HeartbeatData,
    MCPServerHealth,
)


def test_execution_headers_minimal():
    headers = ExecutionHeaders(run_id="run-123")
    result = headers.to_headers()
    assert result["X-Run-ID"] == "run-123"
    assert "X-Session-ID" not in result
    assert "X-Parent-Execution-ID" not in result


def test_execution_headers_with_optional_fields():
    headers = ExecutionHeaders(
        run_id="run-1",
        parent_execution_id="exec-parent",
        session_id="sess-9",
        actor_id="user-42",
    )
    result = headers.to_headers()
    assert result["X-Run-ID"] == "run-1"
    assert result["X-Parent-Execution-ID"] == "exec-parent"
    assert result["X-Session-ID"] == "sess-9"
    assert result["X-Actor-ID"] == "user-42"


def test_heartbeat_data_to_dict():
    mcp = MCPServerHealth(alias="test", status="healthy", tool_count=2)
    hb = HeartbeatData(status=AgentStatus.READY, mcp_servers=[mcp], timestamp="now")
    d = hb.to_dict()
    assert d["status"] == AgentStatus.READY.value
    assert isinstance(d["mcp_servers"], list)
    assert d["mcp_servers"][0]["alias"] == "test"
