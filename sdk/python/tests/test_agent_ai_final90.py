from __future__ import annotations

import base64
import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agentfield.agent_ai import AgentAI
from agentfield.multimodal import Audio, File, Image, Text
from tests.helpers import StubAgent


class _DummyAIConfig:
    model = "openai/gpt-4"
    audio_model = "tts-1"
    vision_model = "dall-e-3"
    video_model = "fal-ai/video"
    fal_api_key = None
    rate_limit_max_retries = 1
    rate_limit_base_delay = 0.1
    rate_limit_max_delay = 1.0
    rate_limit_jitter_factor = 0.1
    rate_limit_circuit_breaker_threshold = 3
    rate_limit_circuit_breaker_timeout = 1

    async def get_model_limits(self, model=None):
        return {"context_length": 1000}

    def get_litellm_params(self, **kwargs):
        return {"api_key": "test-key", **kwargs}

    def copy(self, deep=False):
        return self


@pytest.fixture
def ai_agent():
    agent = StubAgent()
    agent.ai_config = _DummyAIConfig()
    return AgentAI(agent)


@pytest.mark.asyncio
async def test_process_multimodal_args_covers_rich_input_types(ai_agent, monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
        image_file.write(b"\x89PNG\r\n\x1a\npng-bytes")
        image_path = image_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(b"RIFF0000WAVEfmt ")
        audio_path = audio_file.name

    try:
        messages = ai_agent._process_multimodal_args(
            (
                Text(text="hello"),
                Image(image_url="https://example.com/image.png"),
                Audio(input_audio={"data": "YQ==", "format": "wav"}),
                File(file={"url": "https://example.com/file.txt"}),
                image_path,
                audio_path,
                "data:audio/mp3;base64,ZmFrZQ==",
                b"\xff\xd8\xffjpeg",
                b"RIFF0000WAVEfmt ",
                {
                    "system": "sys",
                    "user": "usr",
                    "text": "extra",
                    "image": "https://img",
                },
                {"role": "assistant", "content": "prior"},
                [{"role": "tool", "content": "done"}, {"user": "nested"}],
                {"nested": True},
                123,
            )
        )
    finally:
        os.unlink(image_path)
        os.unlink(audio_path)

    roles = [message["role"] for message in messages if "role" in message]
    assert "assistant" in roles
    assert "tool" in roles
    assert "user" in roles

    user_message = next(
        message for message in messages if message.get("role") == "user"
    )
    assert any(item["type"] == "image_url" for item in user_message["content"])
    assert any(item["type"] == "input_audio" for item in user_message["content"])
    assert any(
        "Data:" in item["text"]
        for item in user_message["content"]
        if item["type"] == "text"
    )
    assert any(
        item["text"] == "123"
        for item in user_message["content"]
        if item["type"] == "text"
    )


@pytest.mark.asyncio
async def test_audio_generation_paths(ai_agent, monkeypatch):
    fal_audio = AsyncMock(return_value="fal-audio")
    fal_video = AsyncMock(return_value="fal-video")
    ai_agent._fal_provider_instance = SimpleNamespace(
        generate_audio=fal_audio,
        generate_video=fal_video,
        generate_image=AsyncMock(return_value="fal-image"),
        supported_modalities=["image", "audio", "video"],
        name="fal",
    )
    # Reset router so it picks up the new fal mock
    ai_agent._media_router_instance = None

    tts_result = await ai_agent.ai_with_audio("say hi", model="fal-ai/kokoro")
    assert tts_result == "fal-audio"
    fal_audio.assert_awaited_once()

    ai_agent._generate_tts_audio = AsyncMock(return_value="tts")
    assert await ai_agent.ai_with_audio("hello", model="tts-1") == "tts"

    ai_agent._generate_openai_direct_audio = AsyncMock(return_value="direct")
    assert (
        await ai_agent.ai_with_audio(
            "hello", model="gpt-4o-mini-tts", mode="openai_direct"
        )
        == "direct"
    )

    ai_agent.ai = AsyncMock(return_value="chat-audio")
    result = await ai_agent.ai_with_audio("hello", model="openai/gpt-4o-audio-preview")
    assert result == "chat-audio"
    ai_agent.ai.assert_awaited_once()
    assert ai_agent.ai.await_args.kwargs["audio"]["voice"] == "alloy"

    assert await ai_agent.ai_generate_video("clip") == "fal-video"
    with pytest.raises(ValueError, match="No provider"):
        await ai_agent.ai_generate_video("clip", model="openai/not-video")


@pytest.mark.asyncio
async def test_tts_audio_success_and_fallback(ai_agent, monkeypatch):
    litellm_module = types.SimpleNamespace(
        aspeech=AsyncMock(return_value=SimpleNamespace(content=b"abc"))
    )
    monkeypatch.setattr("agentfield.agent_ai.litellm", litellm_module, raising=False)

    response = await ai_agent._generate_tts_audio("hello", format="mp3")
    assert response.audio is not None
    assert response.audio.data == base64.b64encode(b"abc").decode("utf-8")

    litellm_module.aspeech = AsyncMock(side_effect=RuntimeError("tts fail"))
    fallback = await ai_agent._generate_tts_audio()
    assert fallback.audio is None
    assert fallback.text == "Hello, this is a test audio message."


@pytest.mark.asyncio
async def test_openai_direct_audio_success_and_fallback(ai_agent, monkeypatch):
    class _StreamingResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream_to_file(self, path):
            with open(path, "wb") as fh:
                fh.write(b"streamed-audio")

    class _OpenAIClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.audio = SimpleNamespace(
                speech=SimpleNamespace(
                    with_streaming_response=SimpleNamespace(
                        create=lambda **kwargs: _StreamingResponse()
                    )
                )
            )

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = _OpenAIClient
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    response = await ai_agent._generate_openai_direct_audio(
        "hello", format="wav", speed=1.25
    )
    assert response.audio is not None
    assert base64.b64decode(response.audio.data) == b"streamed-audio"

    ai_agent.agent.ai_config.get_litellm_params = Mock(return_value={})
    fallback = await ai_agent._generate_openai_direct_audio("hello")
    assert fallback.audio is None
    assert fallback.text == "hello"


@pytest.mark.asyncio
async def test_vision_and_generate_wrappers(ai_agent, monkeypatch):
    mock_openrouter = SimpleNamespace(
        generate_image=AsyncMock(return_value="openrouter-image"),
        generate_audio=AsyncMock(),
        supported_modalities=["image"],
        name="openrouter",
    )
    mock_litellm = SimpleNamespace(
        generate_image=AsyncMock(return_value="litellm-image"),
        generate_audio=AsyncMock(),
        supported_modalities=["image", "audio"],
        name="litellm",
    )

    ai_agent._fal_provider_instance = SimpleNamespace(
        generate_image=AsyncMock(return_value="fal-image"),
        generate_audio=AsyncMock(),
        generate_video=AsyncMock(),
        supported_modalities=["image", "audio", "video"],
        name="fal",
    )

    # Build a custom router with our mocks
    from agentfield.media_router import MediaRouter

    router = MediaRouter()
    router.register("fal-ai/", ai_agent._fal_provider_instance)
    router.register("fal/", ai_agent._fal_provider_instance)
    router.register("openrouter/", mock_openrouter)
    router.register("", mock_litellm)
    ai_agent._media_router_instance = router

    assert await ai_agent.ai_with_vision("draw", model="fal-ai/flux") == "fal-image"
    assert (
        await ai_agent.ai_with_vision("draw", model="openrouter/google/image")
        == "openrouter-image"
    )
    assert await ai_agent.ai_with_vision("draw", model="dall-e-3") == "litellm-image"

    ai_agent.ai_with_vision = AsyncMock(return_value="wrapped-image")
    ai_agent.ai_with_audio = AsyncMock(return_value="wrapped-audio")

    assert await ai_agent.ai_generate_image("sunset") == "wrapped-image"
    assert await ai_agent.ai_generate_audio("voiceover") == "wrapped-audio"
    assert (
        ai_agent.ai_with_vision.await_args.kwargs["model"]
        == ai_agent.agent.ai_config.vision_model
    )
    assert (
        ai_agent.ai_with_audio.await_args.kwargs["model"]
        == ai_agent.agent.ai_config.audio_model
    )
