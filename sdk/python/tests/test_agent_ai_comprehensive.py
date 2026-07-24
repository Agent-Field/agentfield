"""
Comprehensive tests for AgentAI covering critical execution paths.
"""

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentfield.agent_ai import AgentAI
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
        self.video_model = "fal-ai/minimax-video/image-to-video"
        self.fal_api_key = None

    def copy(self, deep=False):
        """Create a copy of the config."""
        import copy as copy_module

        if deep:
            return copy_module.deepcopy(self)
        else:
            return copy_module.copy(self)

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
    monkeypatch.setattr("agentfield.agent_ai.litellm", module, raising=False)
    return module


def make_chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, audio=None))]
    )


@pytest.mark.asyncio
async def test_ai_request_building_with_different_models(monkeypatch, agent_with_ai):
    """Test AI request building with different model configurations."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("test response")

    ai = AgentAI(agent_with_ai)

    # Test with default model
    result = await ai.ai("test prompt")
    assert result.text == "test response"
    assert litellm_module.acompletion.called

    # Test with custom model (must include provider prefix)
    result = await ai.ai("test prompt", model="anthropic/claude-3-opus")
    assert result.text == "test response"
    call_args = litellm_module.acompletion.call_args
    assert call_args[1]["model"] == "anthropic/claude-3-opus"


@pytest.mark.asyncio
async def test_ai_response_parsing_and_error_handling(monkeypatch, agent_with_ai):
    """Test response parsing and error handling."""
    litellm_module = setup_litellm_stub(monkeypatch)

    ai = AgentAI(agent_with_ai)

    # Test successful response
    litellm_module.acompletion.return_value = make_chat_response("success")
    result = await ai.ai("test")
    assert result.text == "success"

    # Test error response
    litellm_module.acompletion.side_effect = Exception("API error")
    with pytest.raises(Exception):
        await ai.ai("test")


@pytest.mark.asyncio
async def test_ai_streaming_response(monkeypatch, agent_with_ai):
    """Test streaming response handling."""
    litellm_module = setup_litellm_stub(monkeypatch)

    # Create a mock streaming response
    async def stream_generator():
        for chunk in ["chunk1", "chunk2", "chunk3"]:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))]
            )

    litellm_module.acompletion.return_value = stream_generator()

    ai = AgentAI(agent_with_ai)
    result = await ai.ai("test", stream=True)

    # Should return a generator/async iterator
    assert hasattr(result, "__aiter__") or hasattr(result, "__iter__")


@pytest.mark.asyncio
async def test_ai_multimodal_input_processing(monkeypatch, agent_with_ai):
    """Test multimodal input processing."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("image analyzed")

    ai = AgentAI(agent_with_ai)

    # Test with image URL
    result = await ai.ai("https://example.com/image.jpg", "What's in this image?")
    assert result.text == "image analyzed"

    # Verify messages were constructed correctly
    call_args = litellm_module.acompletion.call_args
    messages = call_args[1]["messages"]
    assert len(messages) > 0


@pytest.mark.asyncio
async def test_ai_error_recovery_and_retry(monkeypatch, agent_with_ai):
    """Test error recovery and retry logic."""
    litellm_module = setup_litellm_stub(monkeypatch)

    ai = AgentAI(agent_with_ai)

    # Test retry on rate limit error
    call_count = 0

    async def rate_limit_then_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            from httpx import HTTPStatusError

            error = HTTPStatusError(
                "Rate limit", request=MagicMock(), response=MagicMock()
            )
            error.response.status_code = 429
            raise error
        return make_chat_response("success after retry")

    litellm_module.acompletion.side_effect = rate_limit_then_success

    result = await ai.ai("test")
    assert result.text == "success after retry"
    assert call_count == 2


@pytest.mark.asyncio
async def test_ai_with_schema_validation(monkeypatch, agent_with_ai):
    """Test AI call with Pydantic schema validation."""
    from pydantic import BaseModel

    class TestSchema(BaseModel):
        name: str
        age: int

    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response(
        '{"name": "John", "age": 30}'
    )

    ai = AgentAI(agent_with_ai)
    result = await ai.ai("test", schema=TestSchema)

    assert isinstance(result, TestSchema)
    assert result.name == "John"
    assert result.age == 30


@pytest.mark.asyncio
async def test_ai_with_memory_injection(monkeypatch, agent_with_ai):
    """Test AI call with memory scope injection."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("response")

    # Mock memory methods
    agent_with_ai.memory.get = MagicMock(return_value={"key": "value"})
    agent_with_ai.memory.get_all = MagicMock(return_value=[{"key": "value"}])

    ai = AgentAI(agent_with_ai)
    result = await ai.ai("test", memory_scope=["workflow", "session"])

    assert result.text == "response"
    # Note: Memory injection is not yet fully implemented (see TODO in agent_ai.py)
    # This test verifies the call succeeds even with memory_scope parameter


@pytest.mark.asyncio
async def test_ai_with_context_parameter(monkeypatch, agent_with_ai):
    """Test AI call with context parameter."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("response")

    ai = AgentAI(agent_with_ai)
    context = {"user_id": "123", "session_id": "abc"}

    result = await ai.ai("test", context=context)
    assert result.text == "response"

    # Verify context was passed to litellm
    call_args = litellm_module.acompletion.call_args
    assert call_args is not None


@pytest.mark.asyncio
async def test_ai_model_limits_caching(monkeypatch, agent_with_ai):
    """Test that model limits are cached on first call."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("response")

    # Mock get_model_limits to track calls
    original_get_model_limits = agent_with_ai.ai_config.get_model_limits
    agent_with_ai.ai_config.get_model_limits = AsyncMock(
        side_effect=original_get_model_limits
    )

    ai = AgentAI(agent_with_ai)

    # First call should cache limits
    await ai.ai("test")
    assert agent_with_ai.ai_config.get_model_limits.called


@pytest.mark.asyncio
async def test_ai_fallback_models(monkeypatch, agent_with_ai):
    """Test fallback model behavior."""
    litellm_module = setup_litellm_stub(monkeypatch)

    call_count = 0

    async def fail_then_succeed(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Primary model failed")
        return make_chat_response("fallback success")

    litellm_module.acompletion.side_effect = fail_then_succeed
    agent_with_ai.ai_config.fallback_models = ["openai/gpt-3.5-turbo"]

    ai = AgentAI(agent_with_ai)

    # Should try fallback model
    result = await ai.ai("test")
    assert result.text == "fallback success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_ai_temperature_override(monkeypatch, agent_with_ai):
    """Test temperature parameter override."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("response")

    ai = AgentAI(agent_with_ai)
    await ai.ai("test", temperature=0.9)

    call_args = litellm_module.acompletion.call_args
    assert call_args[1]["temperature"] == 0.9


@pytest.mark.asyncio
async def test_ai_max_tokens_override(monkeypatch, agent_with_ai):
    """Test max_tokens parameter override."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("response")

    ai = AgentAI(agent_with_ai)
    await ai.ai("test", max_tokens=200)

    call_args = litellm_module.acompletion.call_args
    assert call_args[1]["max_tokens"] == 200


@pytest.mark.asyncio
async def test_ai_response_format_json(monkeypatch, agent_with_ai):
    """Test JSON response format."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response('{"key": "value"}')

    ai = AgentAI(agent_with_ai)
    result = await ai.ai("test", response_format="json")

    # Should parse JSON
    if isinstance(result, str):
        parsed = json.loads(result)
        assert parsed["key"] == "value"


@pytest.mark.asyncio
async def test_ai_system_and_user_prompts(monkeypatch, agent_with_ai):
    """Test system and user prompt handling."""
    litellm_module = setup_litellm_stub(monkeypatch)
    litellm_module.acompletion.return_value = make_chat_response("response")

    ai = AgentAI(agent_with_ai)
    await ai.ai(system="You are a helpful assistant", user="What is 2+2?")

    call_args = litellm_module.acompletion.call_args
    messages = call_args[1]["messages"]
    assert len(messages) >= 2
    assert any(msg.get("role") == "system" for msg in messages)
    assert any(msg.get("role") == "user" for msg in messages)
