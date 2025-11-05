import asyncio
import copy
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haxen_sdk.agent_ai import AgentAI
from tests.helpers import StubAgent


class DummyAIConfig:
    def __init__(self):
        self.model = "openai/gpt-4"
        self.temperature = 0.1
        self.max_tokens = 100
        self.top_p = 1.0
        self.stream = False
        self.response_format = "auto"
        self.fallback_models = []
        self.final_fallback_model = None
        self.enable_rate_limit_retry = True
        self.rate_limit_max_retries = 2
        self.rate_limit_base_delay = 0.1
        self.rate_limit_max_delay = 1.0
        self.rate_limit_jitter_factor = 0.1
        self.rate_limit_circuit_breaker_threshold = 3
        self.rate_limit_circuit_breaker_timeout = 1
        self.auto_inject_memory = []
        self.model_limits_cache = {}
        self.audio_model = "tts-1"
        self.vision_model = "dall-e-3"

    def copy(self, deep=False):
        return copy.deepcopy(self)

    async def get_model_limits(self, model=None):
        return {"context_length": 1000, "max_output_tokens": 100}

    def get_litellm_params(self, **overrides):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        params.update(overrides)
        return params


@pytest.fixture
def agent_with_ai():
    agent = StubAgent()
    agent.ai_config = DummyAIConfig()
    agent.memory = SimpleNamespace()
    return agent


def setup_litellm_stub(monkeypatch):
    module = types.ModuleType("litellm")
    module.acompletion = AsyncMock()
    module.completion = lambda **kwargs: None
    module.aspeech = AsyncMock()
    module.aimage_generation = AsyncMock()

    utils_module = types.ModuleType("utils")
    utils_module.get_max_tokens = lambda model: 8192
    utils_module.token_counter = lambda model, messages: 10
    utils_module.trim_messages = lambda messages, model, max_tokens: messages
    module.utils = utils_module

    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setitem(sys.modules, "litellm.utils", utils_module)
    monkeypatch.setattr("haxen_sdk.agent_ai.litellm", module, raising=False)
    return module


def make_chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, audio=None))]
    )


def test_get_rate_limiter_cached(monkeypatch, agent_with_ai):
    created = {}

    class DummyLimiter:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("haxen_sdk.agent_ai.StatelessRateLimiter", DummyLimiter)

    ai = AgentAI(agent_with_ai)
    limiter1 = ai._get_rate_limiter()
    limiter2 = ai._get_rate_limiter()
    assert limiter1 is limiter2
    assert created["max_retries"] == agent_with_ai.ai_config.rate_limit_max_retries


@pytest.mark.asyncio
async def test_ensure_model_limits_cached(monkeypatch, agent_with_ai):
    calls = []

    async def fake_limits(model=None):
        calls.append(model)
        return {"context_length": 2000, "max_output_tokens": 200}

    agent_with_ai.ai_config.get_model_limits = fake_limits
    agent_with_ai.ai_config.audio_model = "openai/audio"
    agent_with_ai.ai_config.vision_model = "openai/vision"

    ai = AgentAI(agent_with_ai)
    await ai._ensure_model_limits_cached()
    assert calls == [None, "openai/audio", "openai/vision"]


@pytest.mark.asyncio
async def test_ai_simple_text(monkeypatch, agent_with_ai):
    ai = AgentAI(agent_with_ai)

    class DummyLimiter:
        async def execute_with_retry(self, func):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(ai, "_get_rate_limiter", lambda: DummyLimiter())
    monkeypatch.setattr(
        "haxen_sdk.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "haxen_sdk.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    result = await ai.ai("Hello world")
    assert hasattr(result, "text")
    assert "ok" in result.text


@pytest.mark.asyncio
async def test_ai_uses_fallback_models(monkeypatch, agent_with_ai):
    agent_with_ai.ai_config.fallback_models = ["openai/gpt-3.5"]
    stub_module = setup_litellm_stub(monkeypatch)

    call_order = []

    async def acompletion_side_effect(**params):
        call_order.append(params["model"])
        if params["model"] == agent_with_ai.ai_config.model:
            raise RuntimeError("primary failed")
        return make_chat_response("fallback")

    stub_module.acompletion.side_effect = acompletion_side_effect

    class StubLimiter:
        def __init__(self):
            self.calls = 0

        async def execute_with_retry(self, func):
            self.calls += 1
            return await func()

    limiter = StubLimiter()
    ai = AgentAI(agent_with_ai)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(ai, "_get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(
        "haxen_sdk.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "haxen_sdk.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    result = await ai.ai("hello")

    assert call_order == [agent_with_ai.ai_config.model, "openai/gpt-3.5"]
    assert limiter.calls == 1
    assert result.text == "fallback"


@pytest.mark.asyncio
async def test_ai_skips_rate_limiter_when_disabled(monkeypatch, agent_with_ai):
    agent_with_ai.ai_config.enable_rate_limit_retry = False
    stub_module = setup_litellm_stub(monkeypatch)
    stub_module.acompletion.return_value = make_chat_response("ok")

    ai = AgentAI(agent_with_ai)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "haxen_sdk.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "haxen_sdk.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    result = await ai.ai("hello")
    assert result.text == "ok"
    assert stub_module.acompletion.await_count == 1


@pytest.mark.asyncio
async def test_ai_with_audio_uses_tts_path(monkeypatch, agent_with_ai):
    ai = AgentAI(agent_with_ai)
    agent_with_ai.ai_config.audio_model = "tts-1"

    captured = {}

    async def fake_generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "audio-response"

    monkeypatch.setattr(ai, "_generate_tts_audio", fake_generate)

    result = await ai.ai_with_audio("speak", voice="nova")

    assert result == "audio-response"
    assert captured["kwargs"]["voice"] == "nova"


@pytest.mark.asyncio
async def test_ai_with_audio_non_tts_calls_ai(monkeypatch, agent_with_ai):
    ai = AgentAI(agent_with_ai)
    agent_with_ai.ai_config.audio_model = "openai/gpt-4o"

    captured = {}

    async def fake_ai(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "delegated"

    monkeypatch.setattr(ai, "ai", fake_ai)

    result = await ai.ai_with_audio(
        "hello", model="openai/gpt-4o", voice="alloy", format="mp3"
    )

    assert result == "delegated"
    assert captured["kwargs"]["modalities"] == ["text", "audio"]
    assert captured["kwargs"]["audio"] == {"voice": "alloy", "format": "mp3"}


@pytest.mark.asyncio
async def test_ai_with_audio_openai_direct(monkeypatch, agent_with_ai):
    ai = AgentAI(agent_with_ai)

    async def fake_direct(*args, **kwargs):
        return "direct-audio"

    monkeypatch.setattr(ai, "_generate_openai_direct_audio", fake_direct)

    result = await ai.ai_with_audio("hello", mode="openai_direct")
    assert result == "direct-audio"


@pytest.mark.asyncio
async def test_ai_with_multimodal_passes_modalities(monkeypatch, agent_with_ai):
    ai = AgentAI(agent_with_ai)
    captured = {}

    async def fake_ai(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "multimodal"

    monkeypatch.setattr(ai, "ai", fake_ai)

    result = await ai.ai_with_multimodal(
        "describe",
        modalities=["text", "audio"],
        audio_config={"voice": "nova"},
    )

    assert result == "multimodal"
    assert captured["kwargs"]["modalities"] == ["text", "audio"]
    assert captured["kwargs"]["audio"] == {"voice": "nova"}
    assert captured["kwargs"]["model"] == "gpt-4o-audio-preview"


@pytest.mark.asyncio
async def test_ai_with_vision_invokes_litellm(monkeypatch, agent_with_ai):
    stub_module = setup_litellm_stub(monkeypatch)
    image_item = SimpleNamespace(url="http://image", b64_json=None, revised_prompt=None)
    stub_module.aimage_generation.return_value = SimpleNamespace(data=[image_item])

    ai = AgentAI(agent_with_ai)
    result = await ai.ai_with_vision("A cat")

    stub_module.aimage_generation.assert_awaited_once()
    assert result.images[0].url == "http://image"
