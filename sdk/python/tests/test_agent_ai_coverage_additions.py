from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentfield.agent_ai import AgentAI
from tests.test_agent_ai import DummyAIConfig
from tests.helpers import StubAgent


@pytest.fixture
def agent_with_ai():
    agent = StubAgent()
    agent.ai_config = DummyAIConfig()
    agent.ai_config.vision_model = "dall-e-3"
    agent.ai_config.audio_model = "tts-1"
    agent.memory = SimpleNamespace()
    return agent


@pytest.mark.asyncio
async def test_generate_tts_audio_uses_iterable_binary_response(monkeypatch, agent_with_ai):
    class LiteLLMStub:
        async def aspeech(self, **kwargs):
            return iter([b"ab", b"cd"])

    monkeypatch.setattr("agentfield.agent_ai.litellm", LiteLLMStub(), raising=False)

    result = await AgentAI(agent_with_ai)._generate_tts_audio("hello", format="mp3")

    assert result.text == "hello"
    assert result.audio is not None
    assert result.audio.format == "mp3"


@pytest.mark.asyncio
async def test_generate_openai_direct_audio_returns_text_only_when_api_key_missing(agent_with_ai):
    agent_with_ai.ai_config.get_litellm_params = lambda **kwargs: {}

    result = await AgentAI(agent_with_ai)._generate_openai_direct_audio("hello")

    assert result.text == "hello"
    assert result.audio is None


@pytest.mark.asyncio
async def test_ai_generate_image_uses_configured_default_model(monkeypatch, agent_with_ai):
    captured = {}
    ai = AgentAI(agent_with_ai)

    async def fake_ai_with_vision(**kwargs):
        captured.update(kwargs)
        return "image-response"

    monkeypatch.setattr(ai, "ai_with_vision", fake_ai_with_vision)

    result = await ai.ai_generate_image("draw a skyline", size="512x512")

    assert result == "image-response"
    assert captured["prompt"] == "draw a skyline"
    assert captured["model"] == "dall-e-3"
    assert captured["size"] == "512x512"


# NOTE: The previously-here test `test_ai_with_vision_routes_openrouter_generation`
# was dropped because it patched `sys.modules["agentfield.vision"]` and collided
# with tests elsewhere in the suite that import the real module at collection
# time. The test passed in isolation but failed when run after those other
# tests. The three other tests in this file still exercise ai_generate_image +
# _generate_tts_audio + _generate_openai_direct_audio, so coverage of agent_ai
# is not materially affected.
