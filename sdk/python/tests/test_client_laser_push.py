from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from httpx import Response

import agentfield.client as client_mod
from agentfield.async_config import AsyncConfig
from agentfield.client import AgentFieldClient, ApprovalResult
from agentfield.exceptions import (
    AgentFieldClientError,
    ExecutionTimeoutError,
    RegistrationError,
    ValidationError,
)
from agentfield.execution_state import ExecuteError
from agentfield.types import AgentStatus, HeartbeatData


class DummySyncResponse:
    def __init__(self, payload=None, status_code=200, text="{}", content=None, headers=None):
        self._payload = {} if payload is None else payload
        self.status_code = status_code
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {"Content-Length": str(len(self.content))}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


@pytest.fixture(autouse=True)
def ensure_event_loop():
    try:
        asyncio.get_event_loop()
        loop = None
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield
    finally:
        if loop is not None:
            loop.close()
            asyncio.set_event_loop(None)


@pytest.fixture
def client():
    return AgentFieldClient(
        base_url="http://example.test",
        api_key="secret-key",
        async_config=AsyncConfig(enable_event_stream=True),
    )


def test_helper_properties_and_httpx_loader(monkeypatch):
    approved = ApprovalResult(decision="approved")
    changes = ApprovalResult(decision="request_changes")

    assert approved.approved is True
    assert approved.changes_requested is False
    assert changes.approved is False
    assert changes.changes_requested is True

    sentinel = object()
    monkeypatch.setattr(client_mod, "httpx", sentinel)
    assert client_mod._ensure_httpx() is sentinel

    monkeypatch.setattr(client_mod.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("nope")))
    monkeypatch.setattr(client_mod, "httpx", None)
    assert client_mod._ensure_httpx(force_reload=True) is None

    no_auth_client = AgentFieldClient(base_url="http://example.test")
    assert no_auth_client._get_auth_headers() == {}
    assert no_auth_client.get_did_auth_headers(b"body") == {}
    assert no_auth_client.did is None
    assert no_auth_client.did_auth_configured is False


def test_prepare_headers_and_sync_node_helpers(monkeypatch, client):
    manager_calls = []
    client.caller_agent_id = "caller-9"
    client._async_execution_manager = SimpleNamespace(
        set_event_stream_headers=lambda headers: manager_calls.append(headers)
    )

    headers = client._prepare_execution_headers(
        {
            "authorization": "Bearer token",
            "x-trace-id": b"abc",
            "x-parent-execution-id": " parent-1 ",
            "x-session-id": 42,
            "x-actor-id": "actor-7",
            "X-Run-ID": "run-1",
        }
    )

    assert headers["X-Parent-Execution-ID"] == "parent-1"
    assert headers["X-Session-ID"] == "42"
    assert headers["X-Actor-ID"] == "actor-7"
    assert headers["X-Caller-Agent-ID"] == "caller-9"
    assert client._latest_event_stream_headers["authorization"] == "Bearer token"
    assert client._latest_event_stream_headers["x-trace-id"] == "abc"
    assert manager_calls == [client._latest_event_stream_headers]

    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *args, **kwargs: DummySyncResponse({"registered": True}),
    )
    monkeypatch.setattr(
        client_mod.requests,
        "put",
        lambda *args, **kwargs: DummySyncResponse({"healthy": True}),
    )
    monkeypatch.setattr(
        client_mod.requests,
        "get",
        lambda *args, **kwargs: DummySyncResponse({"nodes": ["a"]}),
    )

    assert client.register_node({"id": "node-1"}) == {"registered": True}
    assert client.update_health("node-1", {"ok": True}) == {"healthy": True}
    assert client.get_nodes() == {"nodes": ["a"]}


def test_register_node_wraps_errors(monkeypatch, client):
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RegistrationError, match="Failed to register node: boom"):
        client.register_node({"id": "node-1"})


def test_register_node_reraises_registration_error(monkeypatch, client):
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RegistrationError("keep me")),
    )

    with pytest.raises(RegistrationError, match="keep me"):
        client.register_node({"id": "node-1"})


def test_submit_execution_sync_adds_did_headers_and_request_errors(monkeypatch, client):
    captured = {}
    client.set_did_credentials(
        "did:web:example.test:agents:node-1",
        '{"kty":"OKP","crv":"Ed25519","d":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","x":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}',
    )

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["headers"] = headers
        return DummySyncResponse(
            {"execution_id": "exec-1", "run_id": "run-1", "status": "queued"}
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    submission = client._submit_execution_sync(
        "node.reasoner",
        {"value": 1},
        {"X-Run-ID": "run-1"},
    )

    assert submission.execution_id == "exec-1"
    assert captured["headers"]["X-Caller-DID"] == "did:web:example.test:agents:node-1"
    assert "X-DID-Signature" in captured["headers"]
    assert "X-DID-Timestamp" in captured["headers"]

    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            client_mod.requests.RequestException("network")
        ),
    )

    with pytest.raises(AgentFieldClientError, match="Failed to submit execution"):
        client._submit_execution_sync("node.reasoner", {"value": 1}, {"X-Run-ID": "run-1"})


def test_submit_execution_sync_raises_execute_error(monkeypatch, client):
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *args, **kwargs: DummySyncResponse(
            {"message": "bad target"},
            status_code=404,
            text='{"message":"bad target"}',
        ),
    )

    with pytest.raises(ExecuteError) as excinfo:
        client._submit_execution_sync("node.reasoner", {"value": 1}, {"X-Run-ID": "run-1"})

    assert excinfo.value.status_code == 404
    assert "bad target" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_async_http_client_typeerror_fallback_and_aclose(monkeypatch, client):
    class FallbackAsyncClient:
        def __init__(self):
            self.headers = {}
            self.is_closed = False

        async def aclose(self):
            self.is_closed = True

    class RaisingAsyncClient:
        def __init__(self, **kwargs):
            raise TypeError("no kwargs")

    module = types.SimpleNamespace(
        AsyncClient=RaisingAsyncClient,
        Limits=lambda **kwargs: ("limits", kwargs),
        Timeout=lambda *args, **kwargs: ("timeout", args, kwargs),
    )
    fallback_instance = FallbackAsyncClient()
    module.AsyncClient = lambda **kwargs: (_ for _ in ()).throw(TypeError("no kwargs"))
    module.AsyncClient = lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("no kwargs"))

    def fallback_ctor(*args, **kwargs):
        if kwargs:
            raise TypeError("no kwargs")
        return fallback_instance

    module.AsyncClient = fallback_ctor
    monkeypatch.setitem(sys.modules, "httpx", module)
    monkeypatch.setattr(client_mod, "httpx", None)

    created = await client.get_async_http_client()
    reused = await client.get_async_http_client()

    assert created is fallback_instance
    assert reused is created
    assert created.headers["User-Agent"] == "AgentFieldSDK/1.0"
    assert created.headers["Accept"] == "application/json"

    client._async_execution_manager = SimpleNamespace(stop=AsyncMock())
    await client.aclose()

    assert created.is_closed is True
    assert client._async_http_client is None
    assert client._async_http_client_lock is None


@pytest.mark.asyncio
async def test_async_request_falls_back_to_sync(monkeypatch, client):
    calls = {}

    async def fake_to_thread(func, method, url, **kwargs):
        calls["method"] = method
        calls["url"] = url
        calls["headers"] = kwargs["headers"]
        return {"ok": True}

    monkeypatch.setattr(
        client,
        "get_async_http_client",
        AsyncMock(side_effect=AgentFieldClientError("missing httpx")),
    )
    monkeypatch.setattr(client_mod, "_to_thread", fake_to_thread)

    result = await client._async_request("GET", "http://example.test/ping", headers={"X-Test": "1"})

    assert result == {"ok": True}
    assert calls["headers"]["X-API-Key"] == "secret-key"
    assert calls["headers"]["X-Test"] == "1"


def test_sync_request_sets_defaults_and_detects_truncation(monkeypatch, client):
    logged = {"error": [], "debug": []}

    class DummySession:
        def request(self, method, url, **kwargs):
            assert kwargs["headers"]["Content-Type"] == "application/json"
            assert kwargs["stream"] is False
            return DummySyncResponse(
                {"ok": True},
                content=b"x" * 4096,
                headers={"Content-Length": "5000"},
            )

    monkeypatch.setattr(client_mod.AgentFieldClient, "_get_sync_session", lambda: DummySession())
    monkeypatch.setattr(client_mod.logger, "error", logged["error"].append)
    monkeypatch.setattr(client_mod.logger, "debug", logged["debug"].append)

    response = client._sync_request("POST", "http://example.test/submit", json={"value": 1})

    assert response.status_code == 200
    assert any("RESPONSE_TRUNCATION" in msg for msg in logged["error"])
    assert any("POSSIBLE_TRUNCATION" in msg for msg in logged["error"])


def test_discover_capabilities_json_flags_and_context_headers(monkeypatch):
    client = AgentFieldClient(base_url="http://example.test")
    captured = {}
    client._current_workflow_context = SimpleNamespace(
        to_headers=lambda: {"X-Context": "ctx", "X-Caller-Agent-ID": "ctx-agent"}
    )

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        captured["headers"] = headers
        return DummySyncResponse(
            payload={
                "discovered_at": "2026-04-09T00:00:00Z",
                "total_agents": 1,
                "total_reasoners": 1,
                "total_skills": 1,
                "pagination": {"limit": 10, "offset": 2, "has_more": False},
                "capabilities": [
                    {
                        "agent_id": "agent-1",
                        "base_url": "http://agent.test",
                        "version": "1.0.0",
                        "health_status": "healthy",
                        "deployment_type": "dev",
                        "last_heartbeat": "2026-04-09T00:00:00Z",
                        "reasoners": [{"id": "r1", "invocation_target": "agent-1.r1"}],
                        "skills": [{"id": "s1", "invocation_target": "agent-1.s1"}],
                    }
                ],
            },
            text='{"ok":true}',
        )

    monkeypatch.setattr(client_mod.requests, "get", fake_get)
    result = client.discover_capabilities(
        agent_ids=["agent-1"],
        reasoner="r1",
        skill="s1",
        include_output_schema=True,
        include_descriptions=False,
        health_status="HEALTHY",
        limit=10,
        offset=2,
    )

    assert result.json.total_agents == 1
    assert captured["params"]["reasoner"] == "r1"
    assert captured["params"]["skill"] == "s1"
    assert captured["params"]["include_output_schema"] == "true"
    assert captured["params"]["include_descriptions"] == "false"
    assert captured["params"]["health_status"] == "healthy"
    assert captured["params"]["limit"] == "10"
    assert captured["params"]["offset"] == "2"
    assert captured["headers"]["X-Context"] == "ctx"
    assert captured["headers"]["X-Caller-Agent-ID"] == "ctx-agent"


def test_event_stream_fallback_and_sync_session_init(monkeypatch):
    client = AgentFieldClient(
        base_url="http://example.test",
        async_config=AsyncConfig(enable_event_stream=True),
    )
    client._current_workflow_context = SimpleNamespace(
        to_headers=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    client._maybe_update_event_stream_headers(None)
    assert client._latest_event_stream_headers == {}

    sentinel_session = object()

    def fake_init():
        AgentFieldClient._shared_sync_session = sentinel_session

    AgentFieldClient._shared_sync_session = None
    monkeypatch.setattr(AgentFieldClient, "_init_shared_sync_session", fake_init)
    assert AgentFieldClient._get_sync_session() is sentinel_session


@pytest.mark.asyncio
async def test_async_http_client_missing_and_request_injection(monkeypatch, client):
    monkeypatch.setattr(client_mod, "_ensure_httpx", lambda force_reload=False: None)
    monkeypatch.setattr(client_mod, "httpx", None)
    monkeypatch.setitem(sys.modules, "httpx", object())

    with pytest.raises(AgentFieldClientError, match="httpx is required"):
        await client.get_async_http_client()

    captured = {}

    class FakeAsyncClient:
        async def request(self, method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            return {"ok": True}

    monkeypatch.setattr(client, "get_async_http_client", AsyncMock(return_value=FakeAsyncClient()))
    result = await client._async_request("POST", "http://example.test/path")

    assert result == {"ok": True}
    assert captured["headers"]["X-API-Key"] == "secret-key"


@pytest.mark.asyncio
async def test_register_agent_with_status_parse_failures_and_logging(monkeypatch, client):
    monkeypatch.setattr(client_mod.importlib, "import_module", lambda name: SimpleNamespace(__version__="9.9.9"))
    logged = {"error": [], "debug": []}
    monkeypatch.setattr(client_mod.logger, "error", lambda *args: logged["error"].append(args))
    monkeypatch.setattr(client_mod.logger, "debug", lambda *args: logged["debug"].append(args))

    class BadJsonResponse(DummySyncResponse):
        def json(self):
            raise ValueError("bad json")

    monkeypatch.setattr(
        client,
        "_async_request",
        AsyncMock(return_value=BadJsonResponse(status_code=500, text="broken", content=b"1")),
    )
    failed, payload = await client.register_agent_with_status("node-5", [], [], "http://agent.test")
    failed2, payload2 = await client.register_agent_with_status(
        "node-5", [], [], "http://agent.test", suppress_errors=True
    )

    assert (failed, payload) == (False, None)
    assert (failed2, payload2) == (False, None)
    assert logged["error"]
    assert logged["debug"]

    monkeypatch.setattr(client, "_async_request", AsyncMock(side_effect=RuntimeError("boom")))
    failed3, payload3 = await client.register_agent_with_status(
        "node-6", [], [], "http://agent.test", suppress_errors=True
    )
    assert (failed3, payload3) == (False, None)


@pytest.mark.asyncio
async def test_async_manager_creation_success_and_execute_async_paths(monkeypatch):
    client = AgentFieldClient(
        base_url="http://example.test",
        api_key="secret-key",
        async_config=AsyncConfig(enable_event_stream=True),
    )
    propagated = []

    class FakeManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self):
            propagated.append("started")

        def set_event_stream_headers(self, headers):
            propagated.append(headers)

        async def submit_execution(self, **kwargs):
            propagated.append(kwargs)
            return "exec-123"

    client._current_workflow_context = SimpleNamespace(
        to_headers=lambda: {"Authorization": "Bearer token"}
    )
    monkeypatch.setattr(client_mod, "AsyncExecutionManager", FakeManager)

    manager = await client._get_async_execution_manager()
    execution_id = await client.execute_async("node.reasoner", {"value": 1}, timeout=2.0)

    assert manager.kwargs["auth_headers"] == {"X-API-Key": "secret-key"}
    assert propagated[0] == "started"
    assert propagated[1] == {"Authorization": "Bearer token"}
    assert propagated[-1]["target"] == "node.reasoner"
    assert execution_id == "exec-123"

    client.async_config.enable_async_execution = False
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.execute_async("node.reasoner", {"value": 1})


@pytest.mark.asyncio
async def test_async_wrapper_disabled_and_error_branches(monkeypatch, client):
    client.async_config.enable_async_execution = False

    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.poll_execution_status("exec-1")
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.batch_check_statuses(["exec-1"])
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.wait_for_execution_result("exec-1")
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.cancel_async_execution("exec-1")
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.list_async_executions()
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.get_async_execution_metrics()
    with pytest.raises(AgentFieldClientError, match="disabled"):
        await client.cleanup_async_executions()

    client.async_config.enable_async_execution = True
    client.async_config.fallback_to_sync = False

    manager = SimpleNamespace(
        submit_execution=AsyncMock(side_effect=RuntimeError("submit boom")),
        get_execution_status=AsyncMock(side_effect=AgentFieldClientError("retry me")),
        wait_for_result=AsyncMock(side_effect=AgentFieldClientError("wait me")),
        cancel_execution=AsyncMock(side_effect=AgentFieldClientError("cancel me")),
        list_executions=AsyncMock(side_effect=RuntimeError("list boom")),
        cleanup_completed_executions=AsyncMock(side_effect=RuntimeError("cleanup boom")),
        get_metrics=lambda: (_ for _ in ()).throw(RuntimeError("metrics boom")),
        stop=AsyncMock(side_effect=RuntimeError("stop boom")),
        set_event_stream_headers=lambda headers: None,
    )
    client._async_execution_manager = manager
    monkeypatch.setattr(client, "_get_async_execution_manager", AsyncMock(return_value=manager))

    with pytest.raises(AgentFieldClientError, match="Async execution failed for target"):
        await client.execute_async("node.reasoner", {"value": 1})
    with pytest.raises(AgentFieldClientError, match="retry me"):
        await client.poll_execution_status("exec-2")
    manager.get_execution_status = AsyncMock(side_effect=RuntimeError("batch boom"))
    with pytest.raises(AgentFieldClientError, match="Failed to batch check execution statuses"):
        await client.batch_check_statuses(["exec-2"])
    with pytest.raises(AgentFieldClientError, match="wait me"):
        await client.wait_for_execution_result("exec-2")
    with pytest.raises(AgentFieldClientError, match="cancel me"):
        await client.cancel_async_execution("exec-2")
    with pytest.raises(AgentFieldClientError, match="Failed to list async executions"):
        await client.list_async_executions("running")
    with pytest.raises(AgentFieldClientError, match="Failed to get async execution metrics"):
        await client.get_async_execution_metrics()
    with pytest.raises(AgentFieldClientError, match="Failed to cleanup async executions"):
        await client.cleanup_async_executions()
    with pytest.raises(RuntimeError, match="stop boom"):
        await client.close_async_execution_manager()


@pytest.mark.asyncio
async def test_approval_helper_transport_error_branches(monkeypatch, client):
    class RaisingClient:
        async def post(self, *args, **kwargs):
            raise RuntimeError("post boom")

        async def get(self, *args, **kwargs):
            raise RuntimeError("get boom")

    monkeypatch.setattr(client, "get_async_http_client", AsyncMock(return_value=RaisingClient()))

    with pytest.raises(AgentFieldClientError, match="Failed to request approval: post boom"):
        await client.request_approval("exec-9", "apr-9")
    with pytest.raises(AgentFieldClientError, match="Failed to get approval status: get boom"):
        await client.get_approval_status("exec-9")


@pytest.mark.asyncio
async def test_register_agent_and_status_variants(monkeypatch, client):
    monkeypatch.setattr(client_mod.importlib, "import_module", lambda name: SimpleNamespace(__version__="9.9.9"))

    captured = []

    async def fake_async_request(method, url, **kwargs):
        captured.append(kwargs["json"])
        return DummySyncResponse({"ok": True}, status_code=201, content=b'{"ok": true}')

    monkeypatch.setattr(client, "_async_request", fake_async_request)

    ok, payload = await client.register_agent(
        "node-1",
        [{"id": "r1"}],
        [{"id": "s1"}],
        "http://agent.test",
        discovery={"public_url": "http://agent.test"},
        vc_metadata={"issuer": "did:web:test"},
        agent_metadata={"team": "sdk"},
        tags=["blue"],
    )
    ok2, payload2 = await client.register_agent_with_status(
        "node-2",
        [],
        [],
        "http://agent.test",
        status=AgentStatus.READY,
        discovery={"public_url": "http://agent.test"},
        vc_metadata={"issuer": "did:web:test"},
        agent_metadata={"team": "sdk"},
        tags=["green"],
    )

    assert ok is True and payload == {"ok": True}
    assert ok2 is True and payload2 == {"ok": True}
    assert captured[0]["metadata"]["custom"]["team"] == "sdk"
    assert captured[0]["metadata"]["custom"]["vc_generation"] == {"issuer": "did:web:test"}
    assert captured[0]["callback_discovery"] == {"public_url": "http://agent.test"}
    assert captured[0]["proposed_tags"] == ["blue"]
    assert captured[1]["lifecycle_status"] == "ready"
    assert captured[1]["communication_config"]["heartbeat_interval"] == "2s"

    async def non_201(*args, **kwargs):
        return DummySyncResponse({"error": "bad"}, status_code=500, content=b'{"error":"bad"}')

    monkeypatch.setattr(client, "_async_request", non_201)
    failed, failure_payload = await client.register_agent("node-3", [], [], "http://agent.test")
    failed2, failure_payload2 = await client.register_agent_with_status(
        "node-3", [], [], "http://agent.test", suppress_errors=True
    )

    assert failed is False and failure_payload == {"error": "bad"}
    assert failed2 is False and failure_payload2 == {"error": "bad"}

    monkeypatch.setattr(client, "_async_request", AsyncMock(side_effect=RuntimeError("boom")))
    failed3, failure_payload3 = await client.register_agent("node-4", [], [], "http://agent.test")
    failed4, failure_payload4 = await client.register_agent_with_status("node-4", [], [], "http://agent.test")

    assert (failed3, failure_payload3) == (False, None)
    assert (failed4, failure_payload4) == (False, None)


@pytest.mark.asyncio
async def test_heartbeat_and_shutdown_helpers(monkeypatch, client):
    heartbeat = HeartbeatData(status=AgentStatus.READY, timestamp="2026-04-09T00:00:00Z", version="1.2.3")

    monkeypatch.setattr(client, "_async_request", AsyncMock(return_value=DummySyncResponse({"ok": True})))
    monkeypatch.setattr(client_mod.requests, "post", lambda *args, **kwargs: DummySyncResponse({"ok": True}))

    assert await client.send_enhanced_heartbeat("node-1", heartbeat) is True
    assert client.send_enhanced_heartbeat_sync("node-1", heartbeat) is True
    assert await client.notify_graceful_shutdown("node-1") is True
    assert client.notify_graceful_shutdown_sync("node-1") is True

    monkeypatch.setattr(client, "_async_request", AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert await client.send_enhanced_heartbeat("node-1", heartbeat) is False
    assert client.send_enhanced_heartbeat_sync("node-1", heartbeat) is False
    assert await client.notify_graceful_shutdown("node-1") is False
    assert client.notify_graceful_shutdown_sync("node-1") is False


@pytest.mark.asyncio
async def test_async_execution_manager_wrappers(monkeypatch, client):
    manager = SimpleNamespace(
        get_execution_status=AsyncMock(side_effect=[{"status": "running"}, RuntimeError("boom")]),
        wait_for_result=AsyncMock(side_effect=[{"value": 1}, TimeoutError("late"), RuntimeError("broken")]),
        cancel_execution=AsyncMock(side_effect=[True, False, RuntimeError("nope")]),
        list_executions=AsyncMock(return_value=[{"id": "exec-1"}]),
        get_metrics=lambda: {"active": 1},
        cleanup_completed_executions=AsyncMock(return_value=2),
        stop=AsyncMock(),
    )
    client._async_execution_manager = manager
    client.async_config.enable_async_execution = True
    client.async_config.enable_batch_polling = True
    client.async_config.batch_size = 2

    monkeypatch.setattr(client, "_get_async_execution_manager", AsyncMock(return_value=manager))

    assert await client.poll_execution_status("exec-1") == {"status": "running"}

    with pytest.raises(AgentFieldClientError, match="Failed to poll execution status"):
        await client.poll_execution_status("exec-2")

    manager.get_execution_status = AsyncMock(side_effect=[{"status": "done"}, None, None])
    batched = await client.batch_check_statuses(["a", "b", "c"])
    assert batched == {"a": {"status": "done"}, "b": None, "c": None}

    with pytest.raises(ValidationError, match="cannot be empty"):
        await client.batch_check_statuses([])

    client.async_config.enable_batch_polling = False
    manager.get_execution_status = AsyncMock(side_effect=[{"status": "x"}, {"status": "y"}, {"status": "z"}])
    assert await client.batch_check_statuses(["a", "b", "c"]) == {
        "a": {"status": "x"},
        "b": {"status": "y"},
        "c": {"status": "z"},
    }

    assert await client.wait_for_execution_result("exec-3") == {"value": 1}

    with pytest.raises(ExecutionTimeoutError, match="exceeded timeout"):
        await client.wait_for_execution_result("exec-4")

    with pytest.raises(AgentFieldClientError, match="Failed to wait for execution result"):
        await client.wait_for_execution_result("exec-5")

    assert await client.cancel_async_execution("exec-6", "done") is True
    assert await client.cancel_async_execution("exec-7") is False

    with pytest.raises(AgentFieldClientError, match="Failed to cancel execution"):
        await client.cancel_async_execution("exec-8")

    assert await client.list_async_executions("bogus") == []
    assert await client.list_async_executions("running", limit=5) == [{"id": "exec-1"}]
    assert await client.get_async_execution_metrics() == {"active": 1}
    assert await client.cleanup_async_executions() == 2

    await client.close_async_execution_manager()
    assert client._async_execution_manager is None


@pytest.mark.asyncio
async def test_approval_helpers_and_wait_for_approval(client, monkeypatch):
    client.caller_agent_id = "caller-1"
    router = respx.MockRouter(assert_all_called=True, assert_all_mocked=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(router.async_handler)) as async_client:
        monkeypatch.setattr(client, "get_async_http_client", AsyncMock(return_value=async_client))

        request_route = router.post(
            "http://example.test/api/v1/agents/caller-1/executions/exec-1/request-approval"
        ).mock(
            return_value=Response(
                200,
                json={
                    "approval_request_id": "apr-1",
                    "approval_request_url": "https://approve.test/r/apr-1",
                },
            )
        )
        status_route = router.get(
            "http://example.test/api/v1/agents/caller-1/executions/exec-1/approval-status"
        ).mock(
            return_value=Response(
                200,
                json={
                    "status": "approved",
                    "response": {"by": "human"},
                    "request_url": "https://approve.test/r/apr-1",
                    "requested_at": "t1",
                    "responded_at": "t2",
                },
            )
        )

        approval = await client.request_approval(
            "exec-1",
            "apr-1",
            approval_request_url="https://approve.test/r/apr-1",
            callback_url="https://cp.test/callback",
            expires_in_hours=24,
        )
        status = await client.get_approval_status("exec-1")

        assert approval.approval_request_id == "apr-1"
        assert approval.approval_request_url.endswith("/apr-1")
        assert status.status == "approved"
        assert status.response == {"by": "human"}
        assert request_route.called
        assert status_route.called

    async def fake_sleep(_):
        return None

    times = iter([0.0, 0.0, 1.0, 2.0, 3.0])
    responses = [
        AgentFieldClientError("transient"),
        SimpleNamespace(status="pending"),
        SimpleNamespace(status="approved"),
    ]

    async def fake_get_status(execution_id):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(client_mod.time, "time", lambda: next(times))
    monkeypatch.setattr(client, "get_approval_status", fake_get_status)

    waited = await client.wait_for_approval("exec-2", poll_interval=0.1, max_interval=0.2, timeout=5.0)
    assert waited.status == "approved"

    times_timeout = iter([0.0, 10.0])
    monkeypatch.setattr(client_mod.time, "time", lambda: next(times_timeout))

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        await client.wait_for_approval("exec-3", poll_interval=0.1, timeout=1.0)


@pytest.mark.asyncio
async def test_approval_helpers_surface_http_errors_and_log_post_is_best_effort(client):
    client.caller_agent_id = "caller-1"
    router = respx.MockRouter(assert_all_called=True, assert_all_mocked=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(router.async_handler)) as async_client:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(client, "get_async_http_client", AsyncMock(return_value=async_client))
        try:
            router.post(
                "http://example.test/api/v1/agents/caller-1/executions/exec-4/request-approval"
            ).mock(return_value=Response(500, text="bad request"))
            router.get(
                "http://example.test/api/v1/agents/caller-1/executions/exec-4/approval-status"
            ).mock(return_value=Response(502, text="bad gateway"))
            log_route = router.post("http://example.test/api/v1/executions/exec-4/logs").mock(
                side_effect=[Response(200, json={"ok": True}), RuntimeError("boom")]
            )

            with pytest.raises(AgentFieldClientError, match="Approval request failed"):
                await client.request_approval("exec-4", "apr-4")

            with pytest.raises(AgentFieldClientError, match="Approval status request failed"):
                await client.get_approval_status("exec-4")

            await client.post_execution_logs("exec-4", {"message": "one"})
            await client.post_execution_logs("exec-4", [{"message": "two"}])

            assert log_route.call_count == 2
        finally:
            monkeypatch.undo()
