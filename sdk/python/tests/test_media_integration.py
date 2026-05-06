"""
Integration tests for the Media Generation milestone (#470).

Verifies cross-component correctness of MediaRouter, providers,
AgentAI routing, MultimodalResponse consistency, error propagation,
and provider caching — all without live API calls.
"""

from __future__ import annotations

import base64
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield.media_providers import (
    FalProvider,
    MediaProvider,
    OpenRouterProvider,
)
from agentfield.media_router import MediaRouter
from agentfield.multimodal_response import (
    AudioOutput,
    FileOutput,
    ImageOutput,
    MultimodalResponse,
    VideoOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubProvider(MediaProvider):
    """Minimal stub for router tests."""

    def __init__(self, name: str, modalities: List[str]):
        self._name = name
        self._modalities = modalities

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_modalities(self) -> List[str]:
        return self._modalities

    async def generate_image(self, prompt, **kw):
        return MultimodalResponse(text=f"{self._name}:image")

    async def generate_audio(self, text, **kw):
        return MultimodalResponse(text=f"{self._name}:audio")

    async def generate_video(self, prompt, **kw):
        return MultimodalResponse(text=f"{self._name}:video")


def _fake_agent(video_model="fal-ai/minimax-video/image-to-video"):
    """Build a minimal mock Agent suitable for AgentAI construction."""
    ai_config = MagicMock()
    ai_config.fal_api_key = None
    ai_config.video_model = video_model
    ai_config.audio_model = "tts-1"
    ai_config.vision_model = "dall-e-3"
    ai_config.model = "openai/gpt-4o-mini"
    ai_config.copy = MagicMock(return_value=ai_config)
    ai_config.get_litellm_params = MagicMock(
        return_value={"model": "openai/gpt-4o-mini"}
    )

    agent = MagicMock()
    agent.ai_config = ai_config
    agent.async_config = MagicMock()
    agent.async_config.llm_call_timeout = 120.0
    return agent


# ===========================================================================
# 1. MediaRouter integration — prefix-based routing
# ===========================================================================


class TestMediaRouterIntegration:
    def test_register_and_resolve_single_provider(self):
        router = MediaRouter()
        prov = StubProvider("fal", ["image", "audio", "video"])
        router.register("fal-ai/", prov)

        resolved = router.resolve("fal-ai/flux/dev", "image")
        assert resolved is prov

    def test_longest_prefix_wins(self):
        router = MediaRouter()
        generic = StubProvider("generic", ["image", "video"])
        specific = StubProvider("specific", ["image", "video"])

        router.register("openrouter/", generic)
        router.register("openrouter/google/", specific)

        assert router.resolve("openrouter/google/veo-3", "video") is specific
        assert router.resolve("openrouter/openai/dall-e", "image") is generic

    def test_capability_filter(self):
        router = MediaRouter()
        img_only = StubProvider("img", ["image"])
        router.register("img/", img_only)

        assert router.resolve("img/model", "image") is img_only
        with pytest.raises(ValueError, match="No provider"):
            router.resolve("img/model", "video")

    def test_no_match_raises(self):
        router = MediaRouter()
        with pytest.raises(ValueError, match="No provider"):
            router.resolve("unknown/model", "video")

    def test_empty_prefix_catch_all(self):
        router = MediaRouter()
        fallback = StubProvider("fallback", ["image", "audio"])
        router.register("", fallback)

        assert router.resolve("dall-e-3", "image") is fallback
        assert router.resolve("tts-1", "audio") is fallback

    def test_multiple_providers_correct_routing(self):
        router = MediaRouter()
        fal = StubProvider("fal", ["image", "audio", "video"])
        openrouter = StubProvider("openrouter", ["image", "video", "audio", "music"])
        litellm = StubProvider("litellm", ["image", "audio"])

        router.register("fal-ai/", fal)
        router.register("openrouter/", openrouter)
        router.register("", litellm)

        assert router.resolve("fal-ai/flux/dev", "image") is fal
        assert router.resolve("openrouter/google/veo-3", "video") is openrouter
        assert router.resolve("dall-e-3", "image") is litellm
        assert router.resolve("tts-1", "audio") is litellm


# ===========================================================================
# 2. OpenRouterProvider video end-to-end (mock aiohttp)
# ===========================================================================


class TestOpenRouterVideoE2E:
    @pytest.mark.asyncio
    async def test_video_lifecycle_submit_poll_download(self):
        """Full video lifecycle: submit → poll (pending) → poll (completed) → download."""
        provider = OpenRouterProvider(api_key="test-key")

        video_bytes = b"fake-video-content"
        video_b64 = base64.b64encode(video_bytes).decode()

        # Mock aiohttp.ClientSession as a context manager
        mock_session = AsyncMock()

        # Submit response
        submit_resp = AsyncMock()
        submit_resp.status = 202
        submit_resp.json = AsyncMock(return_value={"id": "job-123"})

        # Poll response 1: pending
        poll_resp_1 = AsyncMock()
        poll_resp_1.status = 200
        poll_resp_1.json = AsyncMock(return_value={"status": "pending"})

        # Poll response 2: completed
        poll_resp_2 = AsyncMock()
        poll_resp_2.status = 200
        poll_resp_2.json = AsyncMock(
            return_value={
                "status": "completed",
                "unsigned_urls": ["https://cdn.example.com/video.mp4"],
                "usage": {"cost": 0.05},
            }
        )

        # Download response
        download_resp = AsyncMock()
        download_resp.status = 200
        download_resp.headers = {"Content-Length": str(len(video_bytes))}
        download_resp.read = AsyncMock(return_value=video_bytes)

        # Wire up context managers
        submit_cm = AsyncMock()
        submit_cm.__aenter__ = AsyncMock(return_value=submit_resp)
        submit_cm.__aexit__ = AsyncMock(return_value=False)

        poll_cm_1 = AsyncMock()
        poll_cm_1.__aenter__ = AsyncMock(return_value=poll_resp_1)
        poll_cm_1.__aexit__ = AsyncMock(return_value=False)

        poll_cm_2 = AsyncMock()
        poll_cm_2.__aenter__ = AsyncMock(return_value=poll_resp_2)
        poll_cm_2.__aexit__ = AsyncMock(return_value=False)

        download_cm = AsyncMock()
        download_cm.__aenter__ = AsyncMock(return_value=download_resp)
        download_cm.__aexit__ = AsyncMock(return_value=False)

        post_calls = [submit_cm]
        get_calls = [poll_cm_1, poll_cm_2, download_cm]

        mock_session.post = MagicMock(side_effect=post_calls)
        mock_session.get = MagicMock(side_effect=get_calls)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await provider.generate_video(
                prompt="A sunset timelapse",
                model="openrouter/google/veo-2.0-generate-001",
                poll_interval=0.01,
                timeout=5.0,
            )

        # Validate result
        assert result.has_videos
        assert len(result.videos) == 1
        assert result.videos[0].url == "https://cdn.example.com/video.mp4"
        assert result.videos[0].data == video_b64
        assert result.videos[0].mime_type == "video/mp4"

        # Also in files for backward compat
        assert result.has_files
        assert result.files[0].url == "https://cdn.example.com/video.mp4"

    @pytest.mark.asyncio
    async def test_video_submit_failure(self):
        """Verify submit failure raises RuntimeError."""
        provider = OpenRouterProvider(api_key="test-key")

        submit_resp = AsyncMock()
        submit_resp.status = 402
        submit_resp.text = AsyncMock(return_value="Insufficient credits")

        submit_cm = AsyncMock()
        submit_cm.__aenter__ = AsyncMock(return_value=submit_resp)
        submit_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=submit_cm)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            with pytest.raises(RuntimeError, match="video submit failed"):
                await provider.generate_video(
                    prompt="test",
                    model="openrouter/google/veo-2.0-generate-001",
                )

    @pytest.mark.asyncio
    async def test_video_rejects_non_https_url(self):
        """SSRF protection: refuse non-HTTPS download URL."""
        provider = OpenRouterProvider(api_key="test-key")

        submit_resp = AsyncMock()
        submit_resp.status = 202
        submit_resp.json = AsyncMock(return_value={"id": "job-http"})

        poll_resp = AsyncMock()
        poll_resp.status = 200
        poll_resp.json = AsyncMock(
            return_value={
                "status": "completed",
                "unsigned_urls": ["http://insecure.example.com/video.mp4"],
            }
        )

        submit_cm = AsyncMock()
        submit_cm.__aenter__ = AsyncMock(return_value=submit_resp)
        submit_cm.__aexit__ = AsyncMock(return_value=False)

        poll_cm = AsyncMock()
        poll_cm.__aenter__ = AsyncMock(return_value=poll_resp)
        poll_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=submit_cm)
        mock_session.get = MagicMock(return_value=poll_cm)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            with pytest.raises(RuntimeError, match="non-HTTPS"):
                await provider.generate_video(
                    prompt="test",
                    model="openrouter/google/veo-2.0-generate-001",
                    poll_interval=0.01,
                    timeout=2.0,
                )

    @pytest.mark.asyncio
    async def test_video_rejects_private_download_url(self):
        """SSRF protection: refuse private IP download URL."""
        provider = OpenRouterProvider(api_key="test-key")

        submit_resp = AsyncMock()
        submit_resp.status = 202
        submit_resp.json = AsyncMock(return_value={"id": "job-private"})

        poll_resp = AsyncMock()
        poll_resp.status = 200
        poll_resp.json = AsyncMock(
            return_value={
                "status": "completed",
                "unsigned_urls": ["https://192.168.1.10/video.mp4"],
            }
        )

        submit_cm = AsyncMock()
        submit_cm.__aenter__ = AsyncMock(return_value=submit_resp)
        submit_cm.__aexit__ = AsyncMock(return_value=False)

        poll_cm = AsyncMock()
        poll_cm.__aenter__ = AsyncMock(return_value=poll_resp)
        poll_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=submit_cm)
        mock_session.get = MagicMock(return_value=poll_cm)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            with pytest.raises(RuntimeError, match="private IP"):
                await provider.generate_video(
                    prompt="test",
                    model="openrouter/google/veo-2.0-generate-001",
                    poll_interval=0.01,
                    timeout=2.0,
                )

    @pytest.mark.asyncio
    async def test_video_generation_failed_status(self):
        """Provider returns status=failed."""
        provider = OpenRouterProvider(api_key="test-key")

        submit_resp = AsyncMock()
        submit_resp.status = 202
        submit_resp.json = AsyncMock(return_value={"id": "job-fail"})

        poll_resp = AsyncMock()
        poll_resp.status = 200
        poll_resp.json = AsyncMock(
            return_value={"status": "failed", "error": "content policy"}
        )

        submit_cm = AsyncMock()
        submit_cm.__aenter__ = AsyncMock(return_value=submit_resp)
        submit_cm.__aexit__ = AsyncMock(return_value=False)

        poll_cm = AsyncMock()
        poll_cm.__aenter__ = AsyncMock(return_value=poll_resp)
        poll_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=submit_cm)
        mock_session.get = MagicMock(return_value=poll_cm)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            with pytest.raises(RuntimeError, match="generation failed"):
                await provider.generate_video(
                    prompt="test",
                    model="openrouter/google/veo-2.0-generate-001",
                    poll_interval=0.01,
                    timeout=2.0,
                )


# ===========================================================================
# 3. OpenRouterProvider audio end-to-end (mock aiohttp SSE)
# ===========================================================================


class TestOpenRouterAudioE2E:
    @pytest.mark.asyncio
    async def test_audio_sse_stream_to_audio_output(self):
        """Mock SSE stream -> AudioOutput with transcript."""
        provider = OpenRouterProvider(api_key="test-key")

        # Simulate SSE lines
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello world"}}]}\n',
            b'data: {"choices":[{"delta":{"audio":{"data":"AAAA"}}}]}\n',
            b'data: {"choices":[{"delta":{"audio":{"data":"BBBB","transcript":"Hi"}}}]}\n',
            b"data: [DONE]\n",
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = MagicMock()

        async def fake_iter_any():
            for line in sse_lines:
                yield line

        mock_resp.content.iter_any = fake_iter_any

        mock_session = AsyncMock()

        post_cm = AsyncMock()
        post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        post_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=post_cm)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await provider.generate_audio(
                text="Say hello",
                model="openai/gpt-4o-mini-tts",
                voice="nova",
            )

        assert result.audio is not None
        assert result.audio.data == "AAAABBBB"
        assert result.audio.format == "wav"

    @pytest.mark.asyncio
    async def test_audio_api_key_required(self):
        """Missing API key raises ValueError."""
        provider = OpenRouterProvider.__new__(OpenRouterProvider)
        provider._api_key = None
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                await provider.generate_audio(text="test")


# ===========================================================================
# 4. OpenRouterProvider music end-to-end
# ===========================================================================


class TestOpenRouterMusicE2E:
    @pytest.mark.asyncio
    async def test_music_generation_returns_audio(self):
        """Music generation routes through SSE stream -> AudioOutput."""
        provider = OpenRouterProvider(api_key="test-key")

        sse_lines = [
            b'data: {"choices":[{"delta":{"audio":{"data":"MUSIC"}}}]}\n',
            b"data: [DONE]\n",
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = MagicMock()

        async def fake_iter_any():
            for line in sse_lines:
                yield line

        mock_resp.content.iter_any = fake_iter_any

        mock_session = AsyncMock()
        post_cm = AsyncMock()
        post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        post_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=post_cm)

        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_cm):
            result = await provider.generate_music(
                prompt="upbeat jazz piano solo",
                model="google/lyria-3-pro",
                duration=30,
            )

        assert result.audio is not None
        assert result.audio.data == "MUSIC"

    @pytest.mark.asyncio
    async def test_music_invalid_duration_rejected(self):
        """Duration validation: <= 0 or > 600 raises ValueError."""
        provider = OpenRouterProvider(api_key="test-key")

        with pytest.raises(ValueError, match="duration must be"):
            await provider.generate_music(prompt="test", duration=0)

        with pytest.raises(ValueError, match="duration must be"):
            await provider.generate_music(prompt="test", duration=700)


# ===========================================================================
# 5. AgentAI routing — verify methods route to correct providers
# ===========================================================================


class TestAgentAIRouting:
    @pytest.mark.asyncio
    async def test_ai_generate_video_routes_through_media_router(self):
        """ai_generate_video resolves provider via MediaRouter."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        mock_response = MultimodalResponse(
            text="video prompt",
            videos=[
                VideoOutput(
                    url="https://example.com/v.mp4",
                    data="fakedata",
                    mime_type="video/mp4",
                )
            ],
        )

        # Replace _media_router with a mock that returns our mock provider
        mock_provider = AsyncMock()
        mock_provider.generate_video = AsyncMock(return_value=mock_response)

        mock_router = MagicMock()
        mock_router.resolve = MagicMock(return_value=mock_provider)
        ai._media_router_instance = mock_router

        result = await ai.ai_generate_video(
            "A sunset", model="fal-ai/minimax-video/image-to-video"
        )

        mock_router.resolve.assert_called_once_with(
            "fal-ai/minimax-video/image-to-video", "video"
        )
        mock_provider.generate_video.assert_awaited_once()
        assert result.has_videos

    @pytest.mark.asyncio
    async def test_ai_generate_image_routes_through_media_router(self):
        """ai_generate_image resolves provider via MediaRouter."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        mock_response = MultimodalResponse(
            text="image prompt",
            images=[ImageOutput(url="https://example.com/img.png")],
        )

        mock_provider = AsyncMock()
        mock_provider.generate_image = AsyncMock(return_value=mock_response)

        mock_router = MagicMock()
        mock_router.resolve = MagicMock(return_value=mock_provider)
        ai._media_router_instance = mock_router

        result = await ai.ai_generate_image("A cat", model="fal-ai/flux/dev")

        mock_router.resolve.assert_called_once_with("fal-ai/flux/dev", "image")
        assert result.has_images

    @pytest.mark.asyncio
    async def test_ai_generate_audio_uses_media_router_for_fal(self):
        """ai_generate_audio for fal model routes through MediaRouter."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        mock_response = MultimodalResponse(
            text="audio",
            audio=AudioOutput(data="AAAA", format="wav"),
        )

        mock_provider = MagicMock()
        mock_provider.name = "fal"
        mock_provider.generate_audio = AsyncMock(return_value=mock_response)

        mock_router = MagicMock()
        mock_router.resolve = MagicMock(return_value=mock_provider)
        ai._media_router_instance = mock_router

        result = await ai.ai_generate_audio("Hello", model="fal-ai/kokoro/tts")

        mock_router.resolve.assert_called_once_with("fal-ai/kokoro/tts", "audio")
        assert result.has_audio

    @pytest.mark.asyncio
    async def test_ai_generate_music_routes_to_openrouter(self):
        """ai_generate_music goes directly to _openrouter_provider."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        mock_response = MultimodalResponse(
            text="music",
            audio=AudioOutput(data="MUSIC", format="wav"),
        )

        mock_or_provider = AsyncMock()
        mock_or_provider.generate_music = AsyncMock(return_value=mock_response)
        ai._openrouter_provider_instance = mock_or_provider

        result = await ai.ai_generate_music("jazz piano")

        mock_or_provider.generate_music.assert_awaited_once()
        assert result.has_audio


# ===========================================================================
# 6. MultimodalResponse consistency
# ===========================================================================


class TestMultimodalResponseConsistency:
    def test_video_output_save_and_get_bytes(self, tmp_path):
        video_bytes = b"fake-video-data-12345"
        b64 = base64.b64encode(video_bytes).decode()
        vo = VideoOutput(url="https://x.com/v.mp4", data=b64, mime_type="video/mp4")

        path = tmp_path / "test.mp4"
        vo.save(path)
        assert path.read_bytes() == video_bytes
        assert vo.get_bytes() == video_bytes

    def test_audio_output_save_and_get_bytes(self, tmp_path):
        audio_bytes = b"audio-data-wav"
        b64 = base64.b64encode(audio_bytes).decode()
        ao = AudioOutput(data=b64, format="wav")

        path = tmp_path / "test.wav"
        ao.save(path)
        assert path.read_bytes() == audio_bytes
        assert ao.get_bytes() == audio_bytes

    def test_image_output_save_and_get_bytes(self, tmp_path):
        img_bytes = b"png-image-data"
        b64 = base64.b64encode(img_bytes).decode()
        io_obj = ImageOutput(b64_json=b64)

        path = tmp_path / "test.png"
        io_obj.save(path)
        assert path.read_bytes() == img_bytes
        assert io_obj.get_bytes() == img_bytes

    def test_file_output_save_and_get_bytes(self, tmp_path):
        file_bytes = b"generic-file"
        b64 = base64.b64encode(file_bytes).decode()
        fo = FileOutput(data=b64, mime_type="application/octet-stream")

        path = tmp_path / "test.bin"
        fo.save(path)
        assert path.read_bytes() == file_bytes
        assert fo.get_bytes() == file_bytes

    def test_multimodal_response_has_flags(self):
        resp = MultimodalResponse(
            text="test",
            audio=AudioOutput(data="x", format="wav"),
            images=[ImageOutput(url="http://img")],
            videos=[VideoOutput(url="http://vid")],
            files=[FileOutput(data="y")],
        )
        assert resp.has_audio
        assert resp.has_images
        assert resp.has_videos
        assert resp.has_files
        assert resp.is_multimodal

    def test_multimodal_response_empty(self):
        resp = MultimodalResponse(text="plain text")
        assert not resp.has_audio
        assert not resp.has_images
        assert not resp.has_videos
        assert not resp.has_files
        assert not resp.is_multimodal

    def test_video_output_no_data_raises(self):
        vo = VideoOutput()
        with pytest.raises(ValueError, match="No video data"):
            vo.save("/tmp/test.mp4")
        with pytest.raises(ValueError, match="No video data"):
            vo.get_bytes()

    def test_audio_output_no_data_raises(self):
        ao = AudioOutput()
        with pytest.raises(ValueError, match="No audio data"):
            ao.save("/tmp/test.wav")
        with pytest.raises(ValueError, match="No audio data"):
            ao.get_bytes()


# ===========================================================================
# 7. Error propagation
# ===========================================================================


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_provider_error_propagates_through_ai_generate_video(self):
        """RuntimeError from provider surfaces through AgentAI.ai_generate_video."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        mock_provider = AsyncMock()
        mock_provider.generate_video = AsyncMock(
            side_effect=RuntimeError("Provider exploded")
        )

        mock_router = MagicMock()
        mock_router.resolve = MagicMock(return_value=mock_provider)
        ai._media_router_instance = mock_router

        with pytest.raises(RuntimeError, match="Provider exploded"):
            await ai.ai_generate_video(
                "test", model="fal-ai/minimax-video/image-to-video"
            )

    @pytest.mark.asyncio
    async def test_provider_error_propagates_through_ai_generate_image(self):
        """Error from provider surfaces through AgentAI.ai_generate_image."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        mock_provider = AsyncMock()
        mock_provider.generate_image = AsyncMock(
            side_effect=ConnectionError("Network down")
        )

        mock_router = MagicMock()
        mock_router.resolve = MagicMock(return_value=mock_provider)
        ai._media_router_instance = mock_router

        with pytest.raises(ConnectionError, match="Network down"):
            await ai.ai_generate_image("test", model="fal-ai/flux/dev")

    @pytest.mark.asyncio
    async def test_no_api_key_raises_on_video(self):
        """OpenRouterProvider.generate_video raises without API key."""
        provider = OpenRouterProvider.__new__(OpenRouterProvider)
        provider._api_key = None

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                await provider.generate_video(prompt="test")


# ===========================================================================
# 8. Provider caching
# ===========================================================================


class TestProviderCaching:
    def test_fal_provider_cached(self):
        """_fal_provider returns same instance on repeated access."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        p1 = ai._fal_provider
        p2 = ai._fal_provider
        assert p1 is p2
        assert isinstance(p1, FalProvider)

    def test_openrouter_provider_cached(self):
        """_openrouter_provider returns same instance on repeated access."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        p1 = ai._openrouter_provider
        p2 = ai._openrouter_provider
        assert p1 is p2
        assert isinstance(p1, OpenRouterProvider)

    def test_media_router_cached(self):
        """_media_router returns same instance on repeated access."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)

        r1 = ai._media_router
        r2 = ai._media_router
        assert r1 is r2
        assert isinstance(r1, MediaRouter)

    def test_media_router_has_expected_providers(self):
        """Default MediaRouter has fal, openrouter, and litellm catch-all."""
        from agentfield.agent_ai import AgentAI

        agent = _fake_agent()
        ai = AgentAI(agent)
        router = ai._media_router

        # Verify routing by prefix
        fal_prov = router.resolve("fal-ai/flux/dev", "image")
        assert fal_prov.name == "fal"

        or_prov = router.resolve("openrouter/google/veo-3", "video")
        assert or_prov.name == "openrouter"

        litellm_prov = router.resolve("dall-e-3", "image")
        assert litellm_prov.name == "litellm"
