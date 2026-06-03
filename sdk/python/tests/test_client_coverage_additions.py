from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentfield.client import AgentFieldClient, _Submission
from agentfield.exceptions import AgentFieldClientError
from agentfield.execution_state import ExecuteError


class DummyAsyncClient:
    def __init__(self):
        self.headers = {}
        self.requests = []
        self.is_closed = False

    async def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return {"ok": True}


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


@pytest.fixture
def client():
    return AgentFieldClient(base_url="http://localhost:8080", api_key="secret")


@pytest.mark.asyncio
async def test_get_async_http_client_falls_back_for_simple_test_double(monkeypatch, client):
    class FlexibleAsyncClient(DummyAsyncClient):
        def __init__(self, **kwargs):
            if kwargs:
                raise TypeError("no kwargs")
            super().__init__()

    class HttpxStub:
        AsyncClient = FlexibleAsyncClient
        Limits = None
        Timeout = None

    monkeypatch.setattr("agentfield.client.httpx", None)
    monkeypatch.setattr("agentfield.client._ensure_httpx", lambda force_reload=False: HttpxStub)

    async_client = await client.get_async_http_client()

    assert isinstance(async_client, FlexibleAsyncClient)
    assert async_client.headers == {
        "User-Agent": "AgentFieldSDK/1.0",
        "Accept": "application/json",
    }


@pytest.mark.asyncio
async def test_async_request_injects_api_key_and_sync_fallback(monkeypatch, client):
    captured = {}

    async def fake_to_thread(func, method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = dict(kwargs["headers"])
        return "sync-response"

    async def fail_get_async_http_client():
        raise AgentFieldClientError("missing httpx")

    monkeypatch.setattr(client, "get_async_http_client", fail_get_async_http_client)
    monkeypatch.setattr("agentfield.client._to_thread", fake_to_thread)

    result = await client._async_request("GET", "http://example.test/resource")

    assert result == "sync-response"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://example.test/resource"
    assert captured["headers"]["X-API-Key"] == "secret"


def test_prepare_execution_headers_normalizes_casing_and_updates_sse(client):
    seen = {}
    client.caller_agent_id = "caller-1"
    client._maybe_update_event_stream_headers = lambda headers: seen.setdefault(
        "headers", dict(headers)
    )

    headers = client._prepare_execution_headers(
        {
            "x-session-id": "sess-1",
            "x-actor-id": "actor-1",
            "x-parent-execution-id": " parent-1 ",
            "X-Run-ID": "run-1",
            "X-Flag": 5,
        }
    )

    assert headers["X-Run-ID"] == "run-1"
    assert headers["X-Session-ID"] == "sess-1"
    assert headers["X-Actor-ID"] == "actor-1"
    assert headers["X-Parent-Execution-ID"] == "parent-1"
    assert headers["X-Caller-Agent-ID"] == "caller-1"
    assert headers["X-Flag"] == "5"
    assert seen["headers"]["X-Caller-Agent-ID"] == "caller-1"


def test_submit_execution_sync_raises_execute_error_with_message(monkeypatch, client):
    response = DummyResponse(
        status_code=422,
        payload={"message": "invalid input", "details": {"field": "value"}},
    )
    signer = Mock(return_value={"X-Caller-DID": "did:example"})
    client._did_authenticator = SimpleNamespace(
        is_configured=True,
        sign_headers=signer,
    )
    monkeypatch.setattr("agentfield.client.requests.post", lambda *args, **kwargs: response)

    with pytest.raises(ExecuteError) as excinfo:
        client._submit_execution_sync("agent.reasoner", {"value": 1}, {"X-Run-ID": "run-1"})

    assert excinfo.value.status_code == 422
    assert "invalid input" in str(excinfo.value)
    signer.assert_called_once()


def test_format_execution_result_caches_success_and_derives_node_id(monkeypatch, client):
    submission = _Submission(
        execution_id="exec-1",
        run_id="run-1",
        target="node-1.reasoner",
        status="pending",
        target_type="reasoner",
    )
    cached = {}
    client._result_cache = SimpleNamespace(
        set_execution_result=lambda execution_id, result: cached.setdefault(
            execution_id, result
        )
    )

    result, metadata = client._format_execution_result(
        submission,
        {
            "status": "SUCCEEDED",
            "result": {"ok": True},
            "duration": 25,
            "started_at": "2026-04-09T00:00:00Z",
        },
    )

    assert result == {"ok": True}
    assert metadata["node_id"] == "node-1"
    assert metadata["duration_ms"] == 25
    assert metadata["timestamp"] == "2026-04-09T00:00:00Z"
    assert cached == {"exec-1": {"ok": True}}


def test_sync_request_sets_defaults_and_logs_truncation(monkeypatch):
    class Session:
        def request(self, method, url, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                status_code=200,
                headers={"Content-Length": "10"},
                content=b"12345",
            )

    session = Session()
    debug = Mock()
    error = Mock()
    monkeypatch.setattr(AgentFieldClient, "_get_sync_session", classmethod(lambda cls: session))
    monkeypatch.setattr("agentfield.client.logger.debug", debug)
    monkeypatch.setattr("agentfield.client.logger.error", error)

    AgentFieldClient._sync_request("POST", "http://example.test", json={"a": 1})

    assert session.kwargs["headers"]["Content-Type"] == "application/json"
    assert session.kwargs["stream"] is False
    error.assert_called_once()
