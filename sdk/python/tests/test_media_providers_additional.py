import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentfield.media_providers import (
    FalProvider,
    MediaProvider,
    OpenRouterProvider,
    register_provider,
)


class DummyProvider(MediaProvider):
    @property
    def name(self):
        return "dummy"

    @property
    def supported_modalities(self):
        return ["image"]

    async def generate_image(self, prompt, **kwargs):
        return prompt

    async def generate_audio(self, text, **kwargs):
        return text


def test_media_provider_generate_video_and_register_validation():
    provider = DummyProvider()
    with pytest.raises(NotImplementedError, match="dummy does not support video generation"):
        __import__("asyncio").run(provider.generate_video("prompt"))

    with pytest.raises(TypeError, match="provider_class must be a MediaProvider subclass"):
        register_provider("bad", object)


def test_fal_provider_get_client_sets_env_and_raises_importerror(monkeypatch):
    provider = FalProvider(api_key="fal-key")
    fake_module = SimpleNamespace()
    monkeypatch.setitem(__import__("sys").modules, "fal_client", fake_module)

    client = provider._get_client()
    assert client is fake_module
    assert __import__("os").environ["FAL_KEY"] == "fal-key"

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fal_client":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    provider = FalProvider()
    monkeypatch.delitem(__import__("sys").modules, "fal_client", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="fal-client is not installed"):
        provider._get_client()
    monkeypatch.setattr(builtins, "__import__", original_import)


@pytest.mark.asyncio
async def test_fal_provider_generate_image_audio_video_and_transcription(monkeypatch):
    provider = FalProvider(api_key="fal-key")
    mock_client = MagicMock()
    mock_client.subscribe_async = AsyncMock(
        side_effect=[
            {"image": {"url": "https://fal.media/single.png"}},
            {"audio": {"url": "https://fal.media/audio.wav"}},
            {"video": {"url": "https://fal.media/video.mp4"}},
            {"transcription": "hello transcript"},
        ]
    )
    monkeypatch.setattr(provider, "_client", mock_client)

    image_result = await provider.generate_image(
        prompt="A city",
        model=None,
        size="badxsize",
        quality="hd",
        seed=7,
        guidance_scale=3.5,
        extra="value",
    )
    image_call = mock_client.subscribe_async.await_args_list[0]
    assert image_call.args[0] == "fal-ai/flux/dev"
    assert image_call.kwargs["arguments"] == {
        "prompt": "A city",
        "image_size": "square_hd",
        "num_images": 1,
        "num_inference_steps": 50,
        "seed": 7,
        "guidance_scale": 3.5,
        "extra": "value",
    }
    assert image_result.images[0].url == "https://fal.media/single.png"

    audio_result = await provider.generate_audio(
        text="Speak",
        model="fal-ai/tts",
        voice="https://example.com/ref.wav",
        ref_audio_url="https://example.com/original.wav",
        format="mp3",
        gen_text="override",
    )
    audio_call = mock_client.subscribe_async.await_args_list[1]
    assert audio_call.kwargs["arguments"] == {
        "ref_audio_url": "https://example.com/ref.wav",
        "gen_text": "override",
    }
    assert audio_result.audio.url == "https://fal.media/audio.wav"
    assert audio_result.audio.format == "mp3"

    video_result = await provider.generate_video(
        prompt="Animate",
        model=None,
        duration=2.5,
    )
    video_call = mock_client.subscribe_async.await_args_list[2]
    assert video_call.args[0] == "fal-ai/minimax-video/image-to-video"
    assert video_call.kwargs["arguments"] == {"prompt": "Animate", "duration": 2.5}
    assert video_result.files[0].filename == "generated_video.mp4"

    transcription_result = await provider.transcribe_audio(
        audio_url="https://example.com/audio.wav",
        language="en",
        task="translate",
    )
    transcription_call = mock_client.subscribe_async.await_args_list[3]
    assert transcription_call.kwargs["arguments"] == {
        "audio_url": "https://example.com/audio.wav",
        "language": "en",
        "task": "translate",
    }
    assert transcription_result.text == "hello transcript"


@pytest.mark.asyncio
async def test_fal_provider_logs_and_reraises(monkeypatch):
    provider = FalProvider()
    mock_client = MagicMock()
    mock_client.subscribe_async = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(provider, "_client", mock_client)

    logged = []
    monkeypatch.setattr("agentfield.logger.log_error", logged.append)

    with pytest.raises(RuntimeError, match="boom"):
        await provider.generate_image(prompt="A city")

    with pytest.raises(RuntimeError, match="boom"):
        await provider.generate_audio(text="Speak")

    with pytest.raises(RuntimeError, match="boom"):
        await provider.generate_video(prompt="Animate")

    with pytest.raises(RuntimeError, match="boom"):
        await provider.transcribe_audio(audio_url="https://example.com/a.wav")

    assert any("Fal image generation failed: boom" == entry for entry in logged)
    assert any("Fal audio generation failed: boom" == entry for entry in logged)
    assert any("Fal video generation failed: boom" == entry for entry in logged)
    assert any("Fal transcription failed: boom" == entry for entry in logged)


@pytest.mark.asyncio
async def test_openrouter_audio_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider()
    with pytest.raises(ValueError, match="API key required"):
        await provider.generate_audio("hello")
