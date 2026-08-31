import copy
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from agentfield.agent import Agent
from agentfield.agent_ai import AgentAI
from agentfield.execution_context import (
    ExecutionContext,
    reset_execution_context,
    set_execution_context,
)
from agentfield.litellm_observability import (
    _AGENTFIELD_REGISTERED,
    apply_execution_metadata,
    build_execution_metadata,
    register_callbacks,
    resolve_callbacks,
)
from agentfield.types import AIConfig
from tests.helpers import StubAgent


@pytest.fixture(autouse=True)
def restore_observability_globals():
    saved = set(_AGENTFIELD_REGISTERED)
    yield
    _AGENTFIELD_REGISTERED.clear()
    _AGENTFIELD_REGISTERED.update(saved)


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
        "session_id",
        "trace_name",
        "generation_name",
        "tags",
    }
    assert all(
        isinstance(value, str) for key, value in metadata.items() if key != "tags"
    )
    assert all(isinstance(value, str) for value in metadata["tags"])


def test_metadata_omits_absent_optional_fields():
    metadata = build_execution_metadata(
        execution_context(session_id=None, parent_execution_id=None)
    )
    assert "agentfield_session_id" not in metadata
    assert "agentfield_parent_execution_id" not in metadata
    assert "" not in metadata.values()


def test_vendor_aliases_only_when_agentfield_registered():
    context = execution_context()
    aliases = {"trace_id", "session_id", "trace_name", "generation_name", "tags"}
    user_module = callback_module()
    user_module.callbacks.append("langfuse")
    assert aliases.isdisjoint(build_execution_metadata(context))
    register_callbacks(["langfuse"], env={}, litellm_module=user_module)
    metadata = build_execution_metadata(context)
    assert aliases <= metadata.keys()
    assert isinstance(metadata["tags"], list)


def test_metadata_never_contains_user_id_or_requester_metadata():
    metadata = build_execution_metadata(execution_context())
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
        (None, True),
        ("true", True),
        ("unexpected", True),
    ],
)
def test_metadata_opt_out_env(value, enabled):
    env = {} if value is None else {"AGENTFIELD_LITELLM_METADATA": value}
    params = {}
    apply_execution_metadata(params, context=execution_context(), env=env)
    assert ("metadata" in params) is enabled


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


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-4o-mini", "litellm_proxy/gpt-4o-mini", "openrouter/gpt-4o-mini"],
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


def test_tts_paths_are_not_stamped():
    assert "metadata" not in AIConfig(model="openai/gpt-4o-mini").get_litellm_params()
