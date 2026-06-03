from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.client import AgentFieldClient, _Submission
from agentfield.exceptions import AgentFieldClientError, ExecutionTimeoutError
from agentfield.execution_state import ExecuteError


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text="{}", content=b"{}"):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.content = content
        self.headers = {"Content-Length": str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def client():
    return AgentFieldClient(
        base_url="http://example.test",
        api_key="api-key",
        async_config=AsyncConfig(enable_event_stream=True),
    )


def test_get_headers_with_context_uses_fallback_when_context_raises(client):
    client._current_workflow_context = SimpleNamespace(
        to_headers=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    headers = client._get_headers_with_context({"X-Test": "1"})

    assert headers == {"X-API-Key": "api-key", "X-Test": "1"}


def test_maybe_update_event_stream_headers_uses_context_and_manager(client):
    calls = []
    client._current_workflow_context = SimpleNamespace(
        to_headers=lambda: {"Authorization": "Bearer token", "X-Trace": "abc", "Ignore": "nope"}
    )
    client._async_execution_manager = SimpleNamespace(
        set_event_stream_headers=lambda headers: calls.append(headers)
    )

    client._maybe_update_event_stream_headers(None)

    assert client._latest_event_stream_headers == {
        "Authorization": "Bearer token",
        "X-Trace": "abc",
    }
    assert calls == [client._latest_event_stream_headers]


def test_discover_capabilities_returns_xml_and_compact(monkeypatch, client):
    responses = [
        DummyResponse(text="<agents/>"),
        DummyResponse(
            payload={
                "reasoners": [
                    {
                        "id": "reasoner-1",
                        "agent_id": "agent-1",
                        "target": "agent-1.reasoner-1",
                        "tags": ["a"],
                    }
                ],
                "skills": [],
            },
            text='{"reasoners":[{"id":"reasoner-1"}],"skills":[]}',
        ),
    ]
    captured = []

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.append((params, headers))
        return responses.pop(0)

    monkeypatch.setattr("agentfield.client.requests.get", fake_get)

    xml_result = client.discover_capabilities(
        agent="node-1",
        node_id="node-1",
        agent_ids=["node-1", "node-2"],
        node_ids=["node-2", "node-3"],
        tags=["a", "a", "b"],
        format="xml",
        headers={"X-Number": 5},
    )
    compact_result = client.discover_capabilities(
        agent_ids=["agent-1"],
        include_examples=True,
        format="compact",
    )

    assert xml_result.xml == "<agents/>"
    assert captured[0][0]["agent_ids"] == "node-1,node-2,node-3"
    assert captured[0][0]["tags"] == "a,b"
    assert captured[0][1]["Accept"] == "application/xml"
    assert captured[0][1]["X-Number"] == "5"
    assert compact_result.compact.reasoners[0].id == "reasoner-1"
    assert compact_result.compact.reasoners[0].agent_id == "agent-1"


def test_parse_submission_requires_identifiers(client):
    with pytest.raises(AgentFieldClientError, match="missing identifiers"):
        client._parse_submission({}, {"X-Run-ID": "run-1"}, "node.reasoner")


@pytest.mark.asyncio
async def test_submit_execution_async_raises_execute_error_and_formats_body(monkeypatch, client):
    async def fake_request(method, url, **kwargs):
        assert kwargs["content"] == b'{"input":{"value":1}}'
        return DummyResponse(status_code=400, payload={"error": "bad input"})

    monkeypatch.setattr(client, "_async_request", fake_request)

    with pytest.raises(ExecuteError) as excinfo:
        await client._submit_execution_async(
            "node.reasoner",
            {"value": 1},
            {"X-Run-ID": "run-1"},
        )

    assert excinfo.value.status_code == 400
    assert "bad input" in str(excinfo.value)


def test_await_execution_sync_uses_cache_and_failure_payload(monkeypatch, client):
    submission = _Submission(
        execution_id="exec-1",
        run_id="run-1",
        target="node.reasoner",
        status="queued",
    )
    client._result_cache = SimpleNamespace(get_execution_result=lambda execution_id: {"cached": True})

    cached = client._await_execution_sync(submission, {"X-Run-ID": "run-1"})
    assert cached["status"] == "succeeded"
    assert cached["result"] == {"cached": True}

    client._result_cache = SimpleNamespace(get_execution_result=lambda execution_id: None)

    def fake_get(url, headers=None, timeout=None):
        return DummyResponse(status_code=200, payload={"status": "FAILED", "error": "nope"})

    monkeypatch.setattr("agentfield.client.requests.get", fake_get)

    failed = client._await_execution_sync(submission, {"X-Run-ID": "run-1"})
    assert failed["status"] == "failed"
    assert failed["error_message"] == "nope"


@pytest.mark.asyncio
async def test_await_execution_async_times_out(monkeypatch, client):
    submission = _Submission(
        execution_id="exec-1",
        run_id="run-1",
        target="node.reasoner",
        status="queued",
    )
    client._result_cache = SimpleNamespace(get_execution_result=lambda execution_id: None)
    client.async_config.initial_poll_interval = 0.01
    client.async_config.max_execution_timeout = 0.05

    async def fake_request(method, url, **kwargs):
        return DummyResponse(status_code=200, payload={"status": "running"})

    async def fake_sleep(delay):
        return None

    times = iter([0.0, 0.02, 0.08])
    monkeypatch.setattr(client, "_async_request", fake_request)
    monkeypatch.setattr("agentfield.client.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("agentfield.client.time.time", lambda: next(times))

    with pytest.raises(ExecutionTimeoutError, match="exceeded timeout"):
        await client._await_execution_async(submission, {"X-Run-ID": "run-1"})


def test_format_execution_result_and_build_response_for_failure(client):
    submission = _Submission(
        execution_id="exec-2",
        run_id="run-2",
        target="node.reasoner",
        status="queued",
        target_type="reasoner",
    )

    result, metadata = client._format_execution_result(
        submission,
        {
            "status": "failed",
            "error_message": "bad",
            "error_details": {"code": "E_BAD"},
            "completed_at": "2026-04-09T00:00:00Z",
        },
    )

    response = client._build_execute_response(
        submission,
        {"cost": {"usd": 1.2}},
        result,
        metadata,
    )

    assert result["status"] == "failed"
    assert response["result"] is None
    assert response["error_message"] == "bad"
    assert response["error_details"] == {"code": "E_BAD"}
    assert response["cost"] == {"usd": 1.2}


@pytest.mark.asyncio
async def test_execute_async_falls_back_to_sync_and_skips_auth_errors(monkeypatch, client):
    client.async_config.fallback_to_sync = True
    client._get_async_execution_manager = AsyncMock(side_effect=RuntimeError("down"))
    client.execute = AsyncMock(return_value={"status": "succeeded"})
    synthetic = "sync_20260409_deadbeef"
    monkeypatch.setattr(client, "_generate_id", lambda prefix: synthetic)

    execution_id = await client.execute_async("node.reasoner", {"value": 1})

    assert execution_id == synthetic
    client.execute.assert_awaited_once()

    client._get_async_execution_manager = AsyncMock(side_effect=ExecuteError(401, "unauthorized"))
    client.execute.reset_mock()

    with pytest.raises(ExecuteError):
        await client.execute_async("node.reasoner", {"value": 1})

    client.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_manager_metric_and_cleanup_helpers(client):
    client.async_config.enable_async_execution = True

    assert await client.get_async_execution_metrics() == {
        "manager_started": False,
        "message": "Async execution manager not yet initialized",
    }
    assert await client.cleanup_async_executions() == 0

    manager = SimpleNamespace(
        get_metrics=lambda: {"active": 2},
        cleanup_completed_executions=AsyncMock(return_value=3),
    )
    client._async_execution_manager = manager

    assert await client.get_async_execution_metrics() == {"active": 2}
    assert await client.cleanup_async_executions() == 3


@pytest.mark.asyncio
async def test_post_execution_logs_handles_single_list_and_transport_errors(client):
    posted = []

    class FakeHttpClient:
        async def post(self, url, json=None, headers=None, timeout=None):
            posted.append((url, json, headers, timeout))
            if len(posted) > 1:
                raise RuntimeError("boom")

    client.get_async_http_client = AsyncMock(return_value=FakeHttpClient())

    await client.post_execution_logs("exec-1", {"message": "one"})
    await client.post_execution_logs("exec-1", [{"message": "two"}])
    await client.post_execution_logs("", {"message": "ignored"})

    assert posted[0][1] == {"entries": [{"message": "one"}]}
    assert posted[1][1] == {"entries": [{"message": "two"}]}
