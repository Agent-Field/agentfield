import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

from agentfield.agent import Agent
from agentfield.agent_ai import AgentAI
from agentfield.execution_context import (
    ExecutionContext,
    reset_execution_context,
    set_execution_context,
)
from agentfield.litellm_observability import (
    _AGENTFIELD_REGISTERED,
    _AGENTFIELD_REGISTRATION_MODULES,
    agentfield_registered_callbacks,
    apply_execution_metadata,
    build_execution_metadata,
    register_callbacks,
    resolve_callbacks,
)
from agentfield.types import AIConfig
from tests.helpers import StubAgent


_LITELLM_VERSION = Version(importlib.metadata.version("litellm"))

# LiteLLM below 1.98 routes an `openrouter/` model that carries a custom
# api_base through the OpenAI SDK while still applying
# OpenrouterConfig.transform_request, which unconditionally injects a top-level
# `usage` key that AsyncCompletions.create() rejects. That is LiteLLM's own
# request shaping and has nothing to do with the metadata under test; real
# AgentField OpenRouter traffic carries no custom api_base and is unaffected.
# pyproject.toml caps LiteLLM at <1.98.0 on Python 3.10, so this route is only
# exercisable on the newer legs of the CI matrix.
_openrouter_route = pytest.param(
    "openrouter/gpt-4o-mini",
    marks=pytest.mark.skipif(
        _LITELLM_VERSION < Version("1.98.0"),
        reason=(
            "litellm <1.98 sends OpenrouterConfig's top-level `usage` through "
            "the OpenAI SDK, which rejects it; unrelated to metadata"
        ),
    ),
)


@pytest.fixture(autouse=True)
def restore_observability_globals(monkeypatch):
    saved = set(_AGENTFIELD_REGISTERED)
    saved_modules = dict(_AGENTFIELD_REGISTRATION_MODULES)
    # Start from empty. Several tests below assert that the vendor-alias keys
    # are absent, which only holds if this process-global set is empty on
    # entry. Today no other module registers a callback, so restoring on the
    # way out would be enough — clearing on the way in keeps that from becoming
    # an order-dependent failure the first time one does.
    _AGENTFIELD_REGISTERED.clear()
    _AGENTFIELD_REGISTRATION_MODULES.clear()
    # Most historical tests exercise the contents of the stamp. Explicitly opt
    # those tests in while the dedicated contract tests below pass isolated
    # env mappings to cover the new default-off behavior.
    monkeypatch.setenv("AGENTFIELD_LITELLM_METADATA", "true")
    monkeypatch.delenv("AGENTFIELD_LITELLM_CALLBACKS", raising=False)
    yield
    _AGENTFIELD_REGISTERED.clear()
    _AGENTFIELD_REGISTERED.update(saved)
    _AGENTFIELD_REGISTRATION_MODULES.clear()
    _AGENTFIELD_REGISTRATION_MODULES.update(saved_modules)


@pytest.fixture
def real_litellm_state():
    import litellm

    attributes = ("callbacks", "success_callback", "failure_callback")
    saved = {name: copy.copy(getattr(litellm, name)) for name in attributes}
    yield litellm
    for name, value in saved.items():
        current = getattr(litellm, name)
        if isinstance(current, list):
            current[:] = value
        else:
            setattr(litellm, name, value)


class CallbackManager:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def add_litellm_callback(self, name):
        if name not in self.callbacks:
            self.callbacks.append(name)


def callback_module(*, known=("langfuse", "logfire")):
    module = types.ModuleType("litellm_stub")
    module.callbacks = []
    module.success_callback = []
    module.failure_callback = []
    module._known_custom_logger_compatible_callbacks = list(known)
    module.logging_callback_manager = CallbackManager(module.callbacks)
    return module


def execution_context(**overrides):
    values = {
        "run_id": "run-1",
        "execution_id": "exec-1",
        "agent_instance": SimpleNamespace(node_id="node-1"),
        "reasoner_name": "answer",
        "agent_node_id": "node-1",
        "session_id": "session-1",
        "parent_execution_id": "parent-1",
    }
    values.update(overrides)
    return ExecutionContext(**values)


class DummyAIConfig:
    def __init__(self):
        self.model = "openai/gpt-4o-mini"
        self.temperature = 0.1
        self.max_tokens = 100
        self.top_p = 1.0
        self.stream = False
        self.response_format = "auto"
        self.fallback_models = []
        self.final_fallback_model = None
        self.enable_rate_limit_retry = False
        self.model_limits_cache = {
            self.model: {"context_length": 1000, "max_output_tokens": 100}
        }

    def copy(self, deep=False):
        return copy.deepcopy(self)

    async def get_model_limits(self, model=None):
        return self.model_limits_cache[self.model]

    def get_litellm_params(self, **overrides):
        params = {"model": self.model, "stream": False}
        params.update(overrides)
        return params


def ai_stub(monkeypatch, responses):
    module = types.ModuleType("litellm")
    captured = []

    async def acompletion(**kwargs):
        captured.append(kwargs)
        return responses[min(len(captured) - 1, len(responses) - 1)]

    module.acompletion = acompletion
    module.utils = SimpleNamespace(
        token_counter=lambda **kwargs: 10,
        trim_messages=lambda messages, model, max_tokens: messages,
    )
    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setattr("agentfield.agent_ai.litellm", module)
    return captured


def chat_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, audio=None))]
    )


def make_agent_ai():
    agent = StubAgent()
    agent.ai_config = DummyAIConfig()
    agent.memory = SimpleNamespace()
    return AgentAI(agent)


def test_resolve_callbacks_parses_env_list():
    assert resolve_callbacks(
        [" LOGFIRE "], {"AGENTFIELD_LITELLM_CALLBACKS": "langfuse, LOGFIRE ,,,langfuse"}
    ) == ["logfire", "langfuse"]
    # Env-only form, exactly as stated in the behaviour contract.
    assert resolve_callbacks(
        env={"AGENTFIELD_LITELLM_CALLBACKS": "langfuse, LOGFIRE ,,"}
    ) == ["langfuse", "logfire"]


def test_resolve_callbacks_passes_unknown_names_through(monkeypatch):
    module = callback_module(known=())
    messages = []
    monkeypatch.setattr("agentfield.litellm_observability.log_debug", messages.append)
    assert register_callbacks(["helicone"], env={}, litellm_module=module) == [
        "helicone"
    ]
    assert module.callbacks == ["helicone"]
    assert "helicone" in messages[0]


def test_register_callbacks_is_idempotent():
    module = callback_module()
    register_callbacks(["langfuse"], env={}, litellm_module=module)
    register_callbacks(["langfuse"], env={}, litellm_module=module)
    assert module.callbacks == ["langfuse"]


def test_register_callbacks_never_raises():
    module = callback_module()
    module.logging_callback_manager.add_litellm_callback = lambda name: (
        _ for _ in ()
    ).throw(RuntimeError("boom"))
    assert register_callbacks(["langfuse"], env={}, litellm_module=module) == []


def test_incompatible_langfuse_runtime_is_refused(monkeypatch):
    module = callback_module()
    module.__name__ = "litellm"
    messages = []
    monkeypatch.setattr(
        "agentfield.litellm_observability.importlib.metadata.version",
        lambda distribution: "3.15.0",
    )
    monkeypatch.setattr("agentfield.litellm_observability.log_warn", messages.append)

    assert register_callbacks(["langfuse"], env={}, litellm_module=module) == []
    assert module.callbacks == []
    assert any("incompatible" in message for message in messages)


def test_register_callbacks_without_manager_falls_back():
    module = types.SimpleNamespace(callbacks=[])
    register_callbacks(["helicone"], env={}, litellm_module=module)
    register_callbacks(["helicone"], env={}, litellm_module=module)
    assert module.callbacks == ["helicone"]


def test_no_env_no_callbacks_registered():
    module = callback_module()
    before = (
        list(module.callbacks),
        list(module.success_callback),
        list(module.failure_callback),
    )
    assert register_callbacks(env={}, litellm_module=module) == []
    assert before == (
        module.callbacks,
        module.success_callback,
        module.failure_callback,
    )


def test_agent_construction_does_not_touch_litellm_callbacks(
    monkeypatch, real_litellm_state
):
    monkeypatch.delenv("AGENTFIELD_LITELLM_CALLBACKS", raising=False)
    before = tuple(
        copy.copy(getattr(real_litellm_state, name))
        for name in ("callbacks", "success_callback", "failure_callback")
    )
    Agent(node_id="observability-test", auto_register=False, enable_did=False)
    after = tuple(
        copy.copy(getattr(real_litellm_state, name))
        for name in ("callbacks", "success_callback", "failure_callback")
    )
    assert after == before


async def test_agent_construction_survives_registration_failure(monkeypatch):
    monkeypatch.setattr(
        "agentfield.agent.register_callbacks",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert Agent(node_id="observability-test", auto_register=False, enable_did=False)
    # ... and the completion that follows still returns the model's answer.
    ai_stub(monkeypatch, [chat_response()])
    assert (await make_agent_ai().ai(user="hello")).text == "ok"


async def test_registered_callback_appears_once_across_agents_and_calls(
    monkeypatch, real_litellm_state
):
    monkeypatch.setenv("AGENTFIELD_LITELLM_CALLBACKS", "langfuse")
    Agent(node_id="observability-n1", auto_register=False, enable_did=False)
    Agent(node_id="observability-n2", auto_register=False, enable_did=False)
    # Stub the completion path only after the real module has been registered
    # against, so the two calls below exercise app.ai without network.
    ai_stub(monkeypatch, [chat_response(), chat_response()])
    agent_ai = make_agent_ai()
    await agent_ai.ai(user="one")
    await agent_ai.ai(user="two")
    assert real_litellm_state.callbacks.count("langfuse") == 1


def test_metadata_keys_from_execution_context():
    register_callbacks(["langfuse"], env={}, litellm_module=callback_module())
    metadata = build_execution_metadata(execution_context())
    assert set(metadata) == {
        "agentfield_execution_id",
        "agentfield_run_id",
        "agentfield_agent_node_id",
        "agentfield_reasoner",
        "agentfield_session_id",
        "agentfield_parent_execution_id",
        "trace_id",
        "trace_metadata",
        "session_id",
        "trace_name",
        "generation_name",
        "tags",
    }
    assert all(
        isinstance(value, str)
        for key, value in metadata.items()
        if key not in {"tags", "trace_metadata"}
    )
    assert all(isinstance(value, str) for value in metadata["tags"])
    assert metadata["trace_metadata"]["agentfield_run_id"] == "run-1"


def test_metadata_omits_absent_optional_fields():
    metadata = build_execution_metadata(
        execution_context(session_id=None, parent_execution_id=None)
    )
    assert "agentfield_session_id" not in metadata
    assert "agentfield_parent_execution_id" not in metadata
    assert "" not in metadata.values()


def test_vendor_aliases_only_when_agentfield_registered():
    context = execution_context()
    aliases = {
        "trace_id",
        "trace_metadata",
        "session_id",
        "trace_name",
        "generation_name",
        "tags",
    }
    user_module = callback_module()
    user_module.callbacks.append("langfuse")
    assert aliases.isdisjoint(build_execution_metadata(context))
    # An application-owned callback must not be silently claimed.
    assert register_callbacks(["langfuse"], env={}, litellm_module=user_module) == []
    assert aliases.isdisjoint(build_execution_metadata(context))

    agentfield_module = callback_module()
    register_callbacks(["langfuse"], env={}, litellm_module=agentfield_module)
    metadata = build_execution_metadata(context)
    assert aliases <= metadata.keys()
    assert isinstance(metadata["tags"], list)


def test_langfuse_trace_id_is_stable_w3c_compatible():
    register_callbacks(["langfuse"], env={}, litellm_module=callback_module())
    metadata = build_execution_metadata(execution_context(run_id="run/not-w3c"))
    assert metadata["trace_id"] == hashlib.sha256(b"run/not-w3c").hexdigest()[:32]
    assert len(metadata["trace_id"]) == 32
    assert all(character in "0123456789abcdef" for character in metadata["trace_id"])
    assert metadata["agentfield_run_id"] == "run/not-w3c"
    assert metadata["trace_metadata"]["agentfield_run_id"] == "run/not-w3c"


def test_vendor_aliases_not_stamped_for_a_non_langfuse_callback():
    # An application registered `langfuse` itself; the operator asked AgentField
    # for a different vendor. Stamping the LangFuse-native aliases here would
    # re-key, rename and re-tag that application's own generations.
    module = callback_module(known=())
    module.callbacks.append("langfuse")
    assert register_callbacks(["helicone"], env={}, litellm_module=module) == [
        "helicone"
    ]
    metadata = build_execution_metadata(execution_context())
    assert {
        "trace_id",
        "trace_metadata",
        "session_id",
        "trace_name",
        "generation_name",
        "tags",
    }.isdisjoint(metadata)
    assert metadata["agentfield_run_id"] == "run-1"


def test_register_callbacks_reports_nothing_when_no_branch_registers():
    # Neither an add_litellm_callback manager nor a `callbacks` list: there is
    # nowhere to install the callback, so it must not be reported as registered
    # and must not flip the vendor-alias gate.
    module = types.SimpleNamespace()
    assert register_callbacks(["langfuse"], env={}, litellm_module=module) == []
    assert _AGENTFIELD_REGISTERED == set()
    assert "trace_id" not in build_execution_metadata(execution_context())


def test_register_callbacks_does_not_claim_a_refused_callback():
    module = callback_module()
    module.logging_callback_manager.add_litellm_callback = lambda name: None

    assert register_callbacks(["langfuse"], env={}, litellm_module=module) == []
    assert module.callbacks == []
    assert agentfield_registered_callbacks() == frozenset()
    assert "trace_id" not in build_execution_metadata(execution_context())


def test_removed_callback_reconciles_ownership_and_alias_gate():
    module = callback_module()
    assert register_callbacks(["langfuse"], env={}, litellm_module=module) == [
        "langfuse"
    ]
    assert "trace_id" in build_execution_metadata(execution_context())

    module.callbacks.remove("langfuse")
    assert agentfield_registered_callbacks() == frozenset()
    assert _AGENTFIELD_REGISTRATION_MODULES == {}
    assert "trace_id" not in build_execution_metadata(execution_context())


def test_metadata_never_contains_user_id_or_requester_metadata():
    # Register a LangFuse callback so the alias branch — the one place a future
    # `user_id` / `requester_metadata` alias would plausibly be added — is live.
    register_callbacks(["langfuse"], env={}, litellm_module=callback_module())
    metadata = build_execution_metadata(execution_context())
    assert "tags" in metadata
    assert "user_id" not in metadata
    assert "requester_metadata" not in metadata


def test_stamp_never_overrides_caller_metadata():
    params = {"metadata": {"trace_id": "mine", "x": 1}}
    apply_execution_metadata(params, context=execution_context())
    assert params["metadata"]["trace_id"] == "mine"
    assert params["metadata"]["x"] == 1
    assert params["metadata"]["agentfield_run_id"] == "run-1"


def test_stamp_ignores_non_dict_metadata():
    params = {"metadata": "nope"}
    apply_execution_metadata(params, context=execution_context())
    assert params["metadata"] == "nope"


def test_stamp_without_context_is_noop_safe():
    params = {}
    apply_execution_metadata(params)
    assert params == {}


def test_build_execution_metadata_honours_opt_out_when_called_directly():
    assert (
        build_execution_metadata(
            execution_context(), env={"AGENTFIELD_LITELLM_METADATA": " OFF "}
        )
        == {}
    )


def test_apply_execution_metadata_never_raises(monkeypatch):
    class Hostile(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    params = Hostile()
    apply_execution_metadata(params, context=execution_context())
    assert params == {}


@pytest.mark.parametrize(
    "value,enabled",
    [
        ("false", False),
        ("0", False),
        ("No", False),
        (" OFF ", False),
        (None, False),
        ("", False),
        ("true", True),
        (" 1 ", True),
        ("YES", True),
        ("on", True),
        ("unexpected", False),
    ],
)
def test_metadata_opt_in_env(value, enabled):
    env = {} if value is None else {"AGENTFIELD_LITELLM_METADATA": value}
    params = {}
    apply_execution_metadata(params, context=execution_context(), env=env)
    assert ("metadata" in params) is enabled


def test_callback_configuration_opts_into_metadata():
    params = {}
    apply_execution_metadata(
        params,
        context=execution_context(),
        env={"AGENTFIELD_LITELLM_CALLBACKS": "logfire"},
    )
    assert params["metadata"]["agentfield_run_id"] == "run-1"


def test_explicit_metadata_false_overrides_callback_configuration():
    params = {}
    apply_execution_metadata(
        params,
        context=execution_context(),
        env={
            "AGENTFIELD_LITELLM_CALLBACKS": "langfuse",
            "AGENTFIELD_LITELLM_METADATA": "false",
        },
    )
    assert params == {}


async def test_agent_ai_passes_metadata_to_acompletion(monkeypatch):
    captured = ai_stub(monkeypatch, [chat_response()])
    token = set_execution_context(execution_context())
    try:
        assert (await make_agent_ai().ai(user="hello")).text == "ok"
    finally:
        reset_execution_context(token)
    assert captured[0]["metadata"]["agentfield_run_id"] == "run-1"


async def test_agent_ai_without_context_emits_no_agentfield_keys(monkeypatch):
    captured = ai_stub(monkeypatch, [chat_response()])
    assert (await make_agent_ai().ai(user="hello")).text == "ok"
    assert not any(
        key.startswith("agentfield_") for key in captured[0].get("metadata", {})
    )


async def test_tool_loop_inherits_metadata(monkeypatch):
    captured = ai_stub(monkeypatch, [chat_response()])
    # Registered so the stamp carries the `tags` list, whose per-turn identity
    # is asserted below.
    register_callbacks(["langfuse"], env={}, litellm_module=callback_module())

    async def fake_loop(*, litellm_params, make_completion, **kwargs):
        first = await make_completion({**litellm_params})
        await make_completion({**litellm_params})
        return first, []

    monkeypatch.setattr("agentfield.tool_calling.execute_tool_call_loop", fake_loop)
    monkeypatch.setattr(
        "agentfield.tool_calling._build_tool_config",
        lambda tools, agent: (
            [],
            SimpleNamespace(max_turns=2, max_tool_calls=2),
            False,
        ),
    )
    token = set_execution_context(execution_context())
    try:
        await make_agent_ai().ai(user="hello", tools=[])
    finally:
        reset_execution_context(token)
    assert [call["metadata"]["agentfield_run_id"] for call in captured] == [
        "run-1",
        "run-1",
    ]
    assert id(captured[0]["metadata"]) != id(captured[1]["metadata"])
    # ... and the container values are not shared by reference either, so a
    # LangFuse-style integration appending to one turn's tags cannot leak into
    # the next turn.
    assert captured[0]["metadata"]["tags"] == captured[1]["metadata"]["tags"]
    assert id(captured[0]["metadata"]["tags"]) != id(captured[1]["metadata"]["tags"])


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-4o-mini", "litellm_proxy/gpt-4o-mini", _openrouter_route],
)
async def test_metadata_never_reaches_the_wire(model, real_litellm_state):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            captured.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import litellm

        await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            metadata={
                "agentfield_run_id": "run_x",
                "agentfield_execution_id": "exec_x",
            },
            api_base=f"http://127.0.0.1:{server.server_port}",
            api_key="sk-test",
            timeout=10,
        )
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
    assert captured
    assert "metadata" not in captured[0]
    assert not any(key.startswith("agentfield_") for key in captured[0])


@pytest.mark.skipif(
    importlib.util.find_spec("langfuse") is None,
    reason="install the langfuse extra to exercise the real LiteLLM callback",
)
def test_langfuse_callback_sends_correlated_local_ingestion():
    """Catch compatible-install failures that LiteLLM logs but does not raise.

    A subprocess keeps LiteLLM's global callback state isolated. Both the fake
    OpenAI provider and fake LangFuse ingestion endpoint are local, so this
    canary needs neither an LLM key nor LangFuse credentials.
    """
    assert Version(importlib.metadata.version("langfuse")) < Version("3")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._respond({"data": []})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError:
                body = raw_body.decode(errors="replace")
            requests.append((self.path, body))
            if self.path.startswith("/chat/completions"):
                self._respond(
                    {
                        "id": "chatcmpl-langfuse-canary",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "local-canary-ok",
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 3,
                            "total_tokens": 5,
                        },
                    }
                )
            else:
                self._respond({})

        def _respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    run_id = "run_langfuse_local_canary"
    expected_trace_id = hashlib.sha256(run_id.encode()).hexdigest()[:32]
    script = textwrap.dedent(
        f"""
        import asyncio

        from agentfield.agent import Agent
        from agentfield.execution_context import (
            ExecutionContext,
            reset_execution_context,
            set_execution_context,
        )
        from agentfield.types import AIConfig

        async def main():
            config = AIConfig(
                model="openai/gpt-4o-mini",
                api_base={base_url!r},
                api_key="sk-local-only",
                retry_attempts=0,
                enable_rate_limit_retry=False,
                model_limits_cache={{
                    "openai/gpt-4o-mini": {{
                        "context_length": 1000,
                        "max_output_tokens": 100,
                    }}
                }},
            )
            app = Agent(
                node_id="langfuse-canary-node",
                ai_config=config,
                auto_register=False,
                enable_did=False,
            )
            context = ExecutionContext(
                run_id={run_id!r},
                execution_id="exec_langfuse_local_canary",
                agent_instance=app,
                agent_node_id=app.node_id,
                reasoner_name="local_ingestion_canary",
            )
            token = set_execution_context(context)
            try:
                response = await app.ai(user="hello from local canary")
            finally:
                reset_execution_context(token)

            # LiteLLM dispatches async success callbacks after the completion.
            # Keep this loop alive long enough for LangFuse's one-second flush.
            await asyncio.sleep(3)
            from litellm.litellm_core_utils.litellm_logging import (
                _in_memory_loggers,
            )
            for logger in list(_in_memory_loggers):
                client = getattr(logger, "Langfuse", None)
                if client is None:
                    continue
                flush = getattr(client, "flush", None)
                if callable(flush):
                    await asyncio.to_thread(flush)
                shutdown = getattr(client, "shutdown", None)
                if callable(shutdown):
                    await asyncio.to_thread(shutdown)
            print("LANGFUSE_CANARY_RESPONSE=" + str(response.text))

        asyncio.run(main())
        """
    )
    env = os.environ.copy()
    env.pop("AGENTFIELD_LITELLM_METADATA", None)
    env.update(
        {
            "AGENTFIELD_LITELLM_CALLBACKS": "langfuse",
            "LANGFUSE_PUBLIC_KEY": "pk-local-only",
            "LANGFUSE_SECRET_KEY": "sk-local-only",
            "LANGFUSE_HOST": base_url,
            "LANGFUSE_FLUSH_INTERVAL": "1",
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LANGFUSE_CANARY_RESPONSE=local-canary-ok" in result.stdout
    provider_requests = [
        body for path, body in requests if path.startswith("/chat/completions")
    ]
    ingestion_requests = [
        body for path, body in requests if path.startswith("/api/public/ingestion")
    ]
    assert provider_requests
    assert "metadata" not in provider_requests[0]
    assert not any(key.startswith("agentfield_") for key in provider_requests[0])
    assert ingestion_requests, result.stdout + result.stderr
    serialized_ingestion = json.dumps(ingestion_requests)
    assert expected_trace_id in serialized_ingestion
    assert run_id in serialized_ingestion
    assert "agentfield_run_id" in serialized_ingestion
    assert "hello from local canary" in serialized_ingestion
    assert "local-canary-ok" in serialized_ingestion


def test_tts_paths_are_not_stamped():
    # Must run with a live ExecutionContext: the regression this guards is the
    # stamp migrating into AIConfig.get_litellm_params, whose dict the two TTS
    # call sites read config["api_key"] out of — one of them builds a raw OpenAI
    # SDK client whose create() has no metadata kwarg, and the resulting
    # TypeError is swallowed into a silent degrade to a text-only response.
    token = set_execution_context(execution_context())
    try:
        params = AIConfig(model="openai/gpt-4o-mini").get_litellm_params()
    finally:
        reset_execution_context(token)
    assert "metadata" not in params
