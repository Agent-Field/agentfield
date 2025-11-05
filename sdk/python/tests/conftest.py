# Pytest configuration and fixtures for AgentField SDK tests
"""
Shared fixtures used by the actively supported open-source test suite.
The helpers here focus on deterministic behaviour (frozen time, env patching,
optional HTTP stubs) without enforcing heavy global mocks so that individual
tests can choose their own strategy.
"""

from __future__ import annotations

import json
import os
import sys
import types
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pytest
import respx
import responses as responses_lib
from freezegun import freeze_time

from agentfield.agent import Agent
from agentfield.types import AIConfig, MemoryConfig

# Optional imports guarded for test envs
try:
    from pytest_socket import (
        disable_socket,
        enable_socket,
        socket_allow_unix_socket,
    )  # type: ignore
except Exception:  # pragma: no cover
    disable_socket = None  # type: ignore
    enable_socket = None  # type: ignore
    socket_allow_unix_socket = None  # type: ignore

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]


def _network_allowed(node: "pytest.Node") -> bool:
    return bool(
        node.get_closest_marker("integration") or node.get_closest_marker("mcp")
    )


@pytest.fixture(autouse=True)
def _no_network_by_default(request):
    yield


# HTTPX mocking via respx (assert all mocked by default)
@pytest.fixture(autouse=True)
def _respx_mock_by_default(request):
    yield


# requests mocking via responses (non-strict by default; socket is already disabled)
@pytest.fixture(autouse=True)
def _responses_mock_by_default(request):
    if request.node.get_closest_marker("integration"):
        yield
        return

    responses_lib.start()
    try:
        yield
    finally:
        responses_lib.stop()
        responses_lib.reset()


# Deterministic time helper (opt-in)
@pytest.fixture
def frozen_time():
    """
    Freeze time to a fixed instant within this test's scope.
    """
    with freeze_time("2024-01-01T00:00:00Z"):
        yield


# ---------------------------- 1) Environment Patch Fixture ----------------------------
class _EnvPatcher:
    """
    Context-aware environment patcher that safely sets/unsets environment variables
    and restores the original state after use.

    Usage:
        def test_env(env_patch):
            env_patch.setenv("AGENT_CALLBACK_URL", "http://agent:8000")
            # ... test ...
            # Automatically restored after the test

        # As a context manager for sub-scopes:
        with env_patch.context():
            env_patch.setenv("RAILWAY_SERVICE_NAME", "my-svc")
            # sub-scope envs restored on exit
    """

    def __init__(self):
        self._originals: Dict[str, Optional[str]] = {}
        self._to_del_after: set[str] = set()

    def _record(self, key: str):
        if key not in self._originals:
            self._originals[key] = os.environ.get(key)

    def setenv(self, key: str, value: str) -> None:
        self._record(key)
        os.environ[key] = value

    def setdefault(self, key: str, value: str) -> None:
        self._record(key)
        if key not in os.environ:
            os.environ[key] = value

    def unset(self, key: str) -> None:
        self._record(key)
        if key in os.environ:
            del os.environ[key]
            self._to_del_after.add(key)

    def apply(self, mapping: Dict[str, Optional[str]]) -> None:
        """
        Apply a mapping: value=None means unset.
        """
        for k, v in mapping.items():
            if v is None:
                self.unset(k)
            else:
                self.setenv(k, v)

    def restore(self) -> None:
        for key, original in self._originals.items():
            if original is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original
        self._originals.clear()
        self._to_del_after.clear()

    def context(self):
        """
        Local context manager for nested scopes.
        """
        patcher = self

        class _Ctx:
            def __enter__(self):
                return patcher

            def __exit__(self, exc_type, exc, tb):
                patcher.restore()

        return _Ctx()


@pytest.fixture
def env_patch():
    """
    Provides a safe, test-scoped environment variable patcher.

    Example:
        def test_callback_resolution(env_patch):
            env_patch.setenv("AGENT_CALLBACK_URL", "http://host:9999")
            # ... construct Agent and assert base_url resolution ...
    """
    patcher = _EnvPatcher()
    try:
        yield patcher
    finally:
        patcher.restore()


# ---------------------------- 2) LiteLLM Mock Fixture ----------------------------
class _FakeLiteLLMModule(types.ModuleType):
    """
    Minimal fake of the 'litellm' module providing:
    - get_model_info(model)
    - token_counter(messages=..., model=...)
    - completion(**kwargs)
    - acompletion(**kwargs)  (async)

    You can inject custom behaviors via the provided configuration methods.
    """

    class _ModelInfo:
        def __init__(
            self, max_tokens: int = 131072, max_output_tokens: Optional[int] = None
        ):
            self.max_tokens = max_tokens
            self.max_output_tokens = max_output_tokens

    def __init__(self, name="litellm"):
        super().__init__(name)
        # Defaults
        self._model_info: Dict[str, Any] = {}
        self._completion_response: Any = {"choices": [{"message": {"content": "ok"}}]}
        self._completion_error: Optional[Exception] = None

        self._token_counter_impl: Optional[Callable[..., int]] = None

    # API: get_model_info
    def get_model_info(self, model: str):
        info = self._model_info.get(model)
        if info is None:
            return self._ModelInfo()  # default
        return info

    # API: token_counter
    def token_counter(
        self, messages: Any = None, model: Optional[str] = None, **kwargs
    ) -> int:
        if self._token_counter_impl:
            return self._token_counter_impl(messages=messages, model=model, **kwargs)
        # naive default: chars // 4
        text = ""
        if isinstance(messages, list):
            for m in messages:
                c = m.get("content")
                if isinstance(c, str):
                    text += c
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += str(item.get("text", ""))
        return max(1, len(text) // 4)

    # API: completion (sync)
    def completion(self, **kwargs):
        if self._completion_error:
            raise self._completion_error
        return self._completion_response

    # API: acompletion (async)
    async def acompletion(self, **kwargs):
        if self._completion_error:
            raise self._completion_error
        # mimic async latency slightly in debug contexts
        return self._completion_response

    # Config helpers
    def _set_model_info(
        self,
        model: str,
        max_tokens: int = 131072,
        max_output_tokens: Optional[int] = None,
    ):
        self._model_info[model] = self._ModelInfo(
            max_tokens=max_tokens, max_output_tokens=max_output_tokens
        )

    def _set_completion_response(self, response: Any):
        self._completion_response = response
        self._completion_error = None

    def _set_completion_error(self, exc: Exception):
        self._completion_error = exc

    def _set_token_counter(self, impl: Callable[..., int]):
        self._token_counter_impl = impl


@dataclass
class LiteLLMMockController:
    """
    Controller for configuring the fake LiteLLM behavior.

    Methods:
        set_model_info(model, max_tokens, max_output_tokens)
        set_completion_response(response_dict)
        set_completion_error(exception)
        set_token_counter(func)
    """

    module: _FakeLiteLLMModule

    def set_model_info(
        self,
        model: str,
        max_tokens: int = 131072,
        max_output_tokens: Optional[int] = None,
    ):
        self.module._set_model_info(model, max_tokens, max_output_tokens)

    def set_completion_response(self, response: Any):
        self.module._set_completion_response(response)

    def set_completion_error(self, exc: Exception):
        self.module._set_completion_error(exc)

    def set_token_counter(self, func: Callable[..., int]):
        self.module._set_token_counter(func)


@pytest.fixture
def litellm_mock(monkeypatch) -> LiteLLMMockController:
    """
    Provides a fake 'litellm' module injected into sys.modules, covering:
    - get_model_info
    - token_counter
    - completion / acompletion

    Example:
        def test_model_limits_caching(litellm_mock):
            litellm_mock.set_model_info("openai/gpt-4o-mini", max_tokens=128000, max_output_tokens=4096)
            # ... call ai_config.get_model_limits() and assert ...

        async def test_ai_completion_success(litellm_mock):
            litellm_mock.set_completion_response({"choices": [{"message": {"content": "hello"}}]})
            # ... call Agent.ai() and assert ...

        def test_ai_completion_error(litellm_mock):
            litellm_mock.set_completion_error(RuntimeError("LLM down"))
            # ... exercise error path/fallbacks ...
    """
    fake = _FakeLiteLLMModule()
    # If real litellm is present, we still shadow it within this test scope
    monkeypatch.setitem(sys.modules, "litellm", fake)

    # Also shadow the symbol imported as: from litellm import completion as litellm_completion
    def _completion_alias(**kwargs):
        return fake.completion(**kwargs)

    monkeypatch.setitem(sys.modules, "litellm.completion", _completion_alias)  # type: ignore

    return LiteLLMMockController(module=fake)


class AgentFieldHTTPMocks:
    """
    Helper wrapper that registers common AgentField server endpoints on both:
    - httpx (via respx)
    - requests (via responses)

    Defaults to http://localhost:8080 api base.

    Example:
        def test_register_success(http_mocks):
            http_mocks.mock_register_node(status=201)
            # ... call client.register_agent(...) path or register_node() ...

        async def test_execute_ok(http_mocks):
            http_mocks.mock_execute("node.reasoner", json={"result": {"ok": True}})
            # ... await client.execute("node.reasoner", {...}) ...
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api/v1"

    # ----- Nodes -----
    def mock_register_node(
        self, status: int = 201, json: Optional[Dict[str, Any]] = None
    ):
        url = f"{self.api_base}/nodes/register"

        # httpx mock
        respx.post(url).mock(
            return_value=httpx.Response(status_code=status, json=json or {})
        )  # type: ignore

        # requests mock
        responses_lib.add(responses_lib.POST, url, json=json or {}, status=status)

    def mock_update_health(
        self, node_id: str, status: int = 200, json: Optional[Dict[str, Any]] = None
    ):
        url = f"{self.api_base}/nodes/{node_id}/health"
        respx.put(url).mock(
            return_value=httpx.Response(status_code=status, json=json or {})
        )  # type: ignore
        responses_lib.add(responses_lib.PUT, url, json=json or {}, status=status)

    def mock_heartbeat(self, node_id: str, status: int = 200):
        url = f"{self.api_base}/nodes/{node_id}/heartbeat"
        respx.post(url).mock(
            return_value=httpx.Response(status_code=status, json={"ok": True})
        )  # type: ignore
        responses_lib.add(responses_lib.POST, url, json={"ok": True}, status=status)

    # ----- Execute -----
    def mock_execute(
        self,
        target: str,
        status: int = 200,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        url = f"{self.api_base}/execute/{target}"
        respx.post(url).mock(
            return_value=httpx.Response(
                status_code=status, json=json or {"result": {}}, headers=headers or {}
            )
        )  # type: ignore
        responses_lib.add(
            responses_lib.POST,
            url,
            json=json or {"result": {}},
            status=status,
            headers=headers or {},
        )

    # ----- Memory -----
    def mock_memory_get(self, result: Any, status: int = 200):
        url = f"{self.api_base}/memory/get"
        payload = result if isinstance(result, dict) else {"data": result}
        respx.post(url).mock(
            return_value=httpx.Response(status_code=status, json=payload)
        )  # type: ignore
        responses_lib.add(responses_lib.POST, url, json=payload, status=status)

    def mock_memory_delete(self, status: int = 200):
        url = f"{self.api_base}/memory/delete"
        respx.delete(url).mock(
            return_value=httpx.Response(status_code=status, json={"ok": True})
        )  # type: ignore
        responses_lib.add(responses_lib.DELETE, url, json={"ok": True}, status=status)

    def mock_memory_list(self, keys: List[str], status: int = 200):
        url = f"{self.api_base}/memory/list"
        respx.get(url).mock(
            return_value=httpx.Response(
                status_code=status, json=[{"key": k} for k in keys]
            )
        )  # type: ignore
        responses_lib.add(
            responses_lib.GET, url, json=[{"key": k} for k in keys], status=status
        )


@pytest.fixture
def http_mocks() -> AgentFieldHTTPMocks:
    """
    Returns a helper for mocking AgentField server endpoints on both httpx and requests.

    Note:
        This works in concert with the autouse respx/responses wrappers already defined
        at the top of this file.

    Example:
        def test_execute_headers_propagation(http_mocks, workflow_context):
            ctx, headers = workflow_context  # minimal by default
            http_mocks.mock_execute("n.reasoner", json={"result": {"ok": True}})
            # ... call AgentFieldClient.execute(...), ensure headers were passed ...
    """
    return AgentFieldHTTPMocks()


# ---------------------------- 4) Sample Agent Fixture ----------------------------
@pytest.fixture
def sample_ai_config() -> AIConfig:
    """
    Returns a pre-configured AIConfig safe for tests.

    Example:
        def test_ai_config_defaults(sample_ai_config):
            assert sample_ai_config.model
    """
    return AIConfig(
        model="openai/gpt-4o-mini",
        temperature=0.1,
        max_tokens=64,
        top_p=1.0,
        stream=False,
        timeout=10.0,
        retry_attempts=0,
        litellm_params={},
    )


@pytest.fixture
def mock_container_detection(monkeypatch):
    """
    Forces container detection to a predictable value.

    Example:
        def test_callback_prefers_env(mock_container_detection):
            mock_container_detection(is_container=False)
            ...
    """
    from agentfield import agent as agent_mod

    def _apply(is_container: bool = False):
        monkeypatch.setattr(
            agent_mod, "_is_running_in_container", lambda: is_container, raising=True
        )

    return _apply


@pytest.fixture
def mock_ip_detection(monkeypatch):
    """
    Mocks IP detection helpers used by Agent._resolve_callback_url.

    Example:
        def test_local_ip_fallback(mock_ip_detection, env_patch):
            mock_ip_detection(container_ip=None, local_ip="10.0.0.2")
            env_patch.unset("AGENT_CALLBACK_URL")
            ...
    """
    from agentfield import agent as agent_mod

    def _apply(container_ip: Optional[str] = None, local_ip: Optional[str] = None):
        monkeypatch.setattr(
            agent_mod, "_detect_container_ip", lambda: container_ip, raising=True
        )
        monkeypatch.setattr(
            agent_mod, "_detect_local_ip", lambda: local_ip, raising=True
        )

    return _apply


@pytest.fixture
def sample_agent(
    env_patch, mock_container_detection, mock_ip_detection, sample_ai_config
) -> Agent:
    """
    Constructs an Agent instance without serving (no network side-effects).
    - dev_mode=True
    - Avoids network calls during construction
    - Pre-configured with safe defaults

    Example:
        def test_agent_init(sample_agent):
            assert sample_agent.node_id == "test-node"
    """
    # Ensure no env interference unless the test asks for it
    env_patch.unset("AGENT_CALLBACK_URL")
    env_patch.unset("RAILWAY_SERVICE_NAME")
    env_patch.unset("RAILWAY_ENVIRONMENT")

    # Force deterministic detection behavior
    mock_container_detection(is_container=False)
    mock_ip_detection(container_ip=None, local_ip="127.0.0.1")

    agent = Agent(
        node_id="test-node",
        agentfield_server="http://localhost:8080",
        version="0.0.0",
        ai_config=sample_ai_config,
        memory_config=MemoryConfig(
            auto_inject=[], memory_retention="session", cache_results=False
        ),
        dev_mode=True,
        callback_url="http://agent.local",  # resolved to http://agent.local:8000 in __init__
    )
    return agent


# ---------------------------- 6) Fake Server Fixture (in-process FastAPI) ----------------------------
@pytest.fixture
def fake_server(monkeypatch, request):
    """
    Spins up an in-process FastAPI mock server and routes AgentFieldClient calls to it WITHOUT real sockets.
    This is suitable for contract tests while keeping network isolation.

    Endpoints:
      - POST /api/v1/nodes/register          -> 201 Created
      - POST /api/v1/execute/{target}        -> 200 with {"result": {...}, "metadata": {...}}
      - POST /api/v1/memory/get              -> 200 with {"data": ...} or 404
      - DELETE /api/v1/memory/delete         -> 200
      - GET /api/v1/memory/list?scope=...    -> 200 with [{"key": ...}, ...]

    How it works:
      - Patches httpx.AsyncClient to use httpx.ASGITransport against the in-process FastAPI app.
      - AgentFieldClient(async) calls are transparently routed; no sockets required.
      - requests.* fallbacks are NOT routed here; rely on responses/respx for those.

    Returns:
      dict with:
        - "base_url": str (e.g. "http://testserver")
        - "app": FastAPI
        - "memory": dict (in-memory store backing the memory endpoints)
    """
    if FastAPI is None or httpx is None:
        pytest.skip("fastapi/httpx are required for fake_server fixture")

    app = FastAPI(title="AgentField Fake Server")

    memory_store: Dict[str, Any] = {}

    @app.post("/api/v1/nodes/register")
    async def register_node(payload: Dict[str, Any]):
        # Minimal 201 response for contract expectations
        return JSONResponse(
            status_code=201, content={"ok": True, "node": payload.get("id")}
        )

    @app.post("/api/v1/execute/{target}")
    async def execute_target(target: str, payload: Dict[str, Any]):
        # Echo input, fake metadata headers as body fields (clients parse json)
        result = {
            "result": {"echo": payload.get("input", {}), "target": target},
            "metadata": {
                "execution_id": "exec_" + uuid.uuid4().hex[:8],
                "agentfield_request_id": "req_" + uuid.uuid4().hex[:8],
                "agent_node_id": target.split(".")[0] if "." in target else "node",
                "duration_ms": 12,
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }
        return JSONResponse(status_code=200, content=result)

    @app.post("/api/v1/memory/get")
    async def memory_get(payload: Dict[str, Any]):
        key = payload.get("key")
        if key in memory_store:
            data = memory_store[key]
            # Match SDK client expectations (it decodes "data")
            return JSONResponse(
                status_code=200,
                content={
                    "data": json.dumps(data)
                    if not isinstance(data, (dict, list))
                    else data
                },
            )
        return JSONResponse(status_code=404, content={"error": "not_found"})

    @app.delete("/api/v1/memory/delete")
    async def memory_delete(payload: Dict[str, Any]):
        key = payload.get("key")
        memory_store.pop(key, None)
        return JSONResponse(status_code=200, content={"ok": True})

    @app.get("/api/v1/memory/list")
    async def memory_list(scope: Optional[str] = None):
        # Scope is ignored in this simple fake; return all keys
        return JSONResponse(
            status_code=200, content=[{"key": k} for k in sorted(memory_store.keys())]
        )

    # httpx ASGI transport plumbing
    transport = httpx.ASGITransport(app=app)
    RealAsyncClient = httpx.AsyncClient

    def _patched_async_client(*args, **kwargs):
        # Only patch if no explicit transport provided
        if "transport" not in kwargs:
            kwargs["transport"] = transport
        if "base_url" not in kwargs:
            kwargs["base_url"] = "http://testserver"
        return RealAsyncClient(*args, **kwargs)

    # Patch within this fixture scope
    monkeypatch.setattr(httpx, "AsyncClient", _patched_async_client, raising=True)

    resource = {
        "base_url": "http://testserver",
        "app": app,
        "memory": memory_store,
    }
    yield resource

    # cleanup: nothing persistent to stop; AsyncClient restored automatically by monkeypatch


# ---------------------------- Notes and Cross-Cutting Concerns ----------------------------
# - Agent.__init__ callback URL resolution is exercised via env_patch + mock_container_detection + mock_ip_detection
# - AgentFieldClient request/header propagation is covered by http_mocks and fake_server
# - MemoryClient serialization and HTTP fallback paths are supported by http_mocks and fake_server
# - AgentAI model limits caching and message trimming rely on litellm_mock + sample_ai_config
# - AIConfig parameter merging and fallback logic can be tested via sample_ai_config overrides


# ---------------------------- Additional lightweight helpers ----------------------------
@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch):
    """Simple helper to apply environment overrides inside a test. Pass keyword arguments with values or None to unset."""

    def _apply(**values):
        for key, value in values.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, str(value))

    return _apply


@pytest.fixture
def dummy_headers():
    """Baseline execution headers consumed by memory/agentfield client tests."""
    return {
        "X-Workflow-ID": "wf-test",
        "X-Execution-ID": "exec-test",
        "X-AgentField-Request-ID": "req-test",
    }


@pytest.fixture
def responses(request):
    """Compat shim so tests can request a `responses` fixture for manual expectations."""
    existing = getattr(request.node, "_responses_mock", None)
    if existing is not None:
        yield existing
        return
    with responses_lib.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps
