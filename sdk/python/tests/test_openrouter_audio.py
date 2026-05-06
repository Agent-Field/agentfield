"""
Tests for OpenRouter audio output and music generation.

Covers:
- SSE stream parsing for generate_audio
- Audio chunk concatenation
- Transcript extraction
- Model prefix stripping
- generate_music on ABC (raises NotImplementedError by default)
- supported_modalities includes "audio" and "music"
- ai_generate_music convenience method on AgentAI
"""

import base64
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield.agent_ai import AgentAI
from agentfield.media_providers import (
    FalProvider,
    LiteLLMProvider,
    OpenRouterProvider,
)
from agentfield.multimodal_response import AudioOutput, MultimodalResponse


# =============================================================================
# Helpers
# =============================================================================


def _make_sse_lines(events: list[dict], done: bool = True) -> list[bytes]:
    """Build raw SSE byte lines from a list of event dicts."""
    lines = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}\n".encode())
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


def _audio_event(b64_chunk: str = "", transcript: str = "") -> dict:
    """Build a single SSE audio delta event."""
    audio = {}
    if b64_chunk:
        audio["data"] = b64_chunk
    if transcript:
        audio["transcript"] = transcript
    return {"choices": [{"delta": {"audio": audio}}]}


class _FakeContent:
    """Fake aiohttp StreamReader supporting iter_any()."""

    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def iter_any(self):
        for line in self._lines:
            yield line


class _FakeStreamResponse:
    """Fake aiohttp response supporting readline-based SSE parsing."""

    def __init__(self, lines: list[bytes], status: int = 200):
        self.status = status
        self._lines = lines
        self.content = _FakeContent(lines)

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """Fake aiohttp.ClientSession."""

    def __init__(self, response: _FakeStreamResponse, **kwargs):
        self._response = response
        # Accept timeout and other kwargs like real ClientSession
        self._init_kwargs = kwargs

    def post(self, url, **kwargs):
        self._last_post_kwargs = kwargs
        self._last_post_url = url
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


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
        self.fal_api_key = None
        self.video_model = "fal-ai/minimax-video/image-to-video"

    def copy(self, deep=False):
        return copy.deepcopy(self)

    async def get_model_limits(self, model=None):
        return {"context_length": 1000, "max_output_tokens": 100}

    def get_litellm_params(self, **overrides):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        params.update(overrides)
        return params


class StubAgent:
    def __init__(self):
        self.node_id = "test-agent"
        self.ai_config = DummyAIConfig()
        self.memory = SimpleNamespace()


# =============================================================================
# OpenRouterProvider.generate_audio tests
# =============================================================================


class TestOpenRouterGenerateAudio:
    """Tests for OpenRouterProvider.generate_audio SSE streaming."""

    @pytest.mark.asyncio
    async def test_sse_stream_parsing_and_concatenation(self, monkeypatch):
        """Audio base64 chunks from SSE should be concatenated correctly."""
        chunk1 = base64.b64encode(b"audio_part_1").decode()
        chunk2 = base64.b64encode(b"audio_part_2").decode()

        events = [
            _audio_event(b64_chunk=chunk1, transcript="Hello "),
            _audio_event(b64_chunk=chunk2, transcript="world"),
        ]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_audio(
                text="Say hello",
                model="openai/gpt-4o-mini-tts",
                voice="alloy",
                format="wav",
            )

        assert result.has_audio
        assert result.audio.data == chunk1 + chunk2
        assert result.audio.format == "wav"
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcript_extraction(self, monkeypatch):
        """Transcript text should be accumulated from SSE events."""
        events = [
            _audio_event(b64_chunk="AAAA", transcript="First "),
            _audio_event(b64_chunk="BBBB", transcript="second "),
            _audio_event(b64_chunk="CCCC", transcript="third."),
        ]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_audio(text="test")

        assert result.text == "First second third."

    @pytest.mark.asyncio
    async def test_model_prefix_stripping(self, monkeypatch):
        """openrouter/ prefix should be stripped from model before sending."""
        events = [_audio_event(b64_chunk="AAAA")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_audio(
                text="test",
                model="openrouter/openai/gpt-4o-mini-tts",
            )

        # Check the payload sent
        post_kwargs = fake_session._last_post_kwargs
        payload = post_kwargs["json"]
        assert payload["model"] == "openai/gpt-4o-mini-tts"
        assert not payload["model"].startswith("openrouter/")

    @pytest.mark.asyncio
    async def test_empty_stream_returns_no_audio(self, monkeypatch):
        """Empty SSE stream should return response with no audio."""
        lines = _make_sse_lines([])
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_audio(text="test")

        assert not result.has_audio
        assert result.text == "test"

    @pytest.mark.asyncio
    async def test_invalid_voice_defaults_to_alloy(self, monkeypatch):
        """Invalid voice should fall back to alloy."""
        events = [_audio_event(b64_chunk="AAAA")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_audio(text="test", voice="invalid_voice")

        payload = fake_session._last_post_kwargs["json"]
        assert payload["audio"]["voice"] == "alloy"

    @pytest.mark.asyncio
    async def test_invalid_format_defaults_to_wav(self, monkeypatch):
        """Invalid format should fall back to wav."""
        events = [_audio_event(b64_chunk="AAAA")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_audio(text="test", format="invalid_fmt")

        payload = fake_session._last_post_kwargs["json"]
        assert payload["audio"]["format"] == "wav"

    @pytest.mark.asyncio
    async def test_http_error_raises(self, monkeypatch):
        """Non-200 response should raise RuntimeError."""
        fake_resp = _FakeStreamResponse([], status=400)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            with pytest.raises(RuntimeError, match="failed.*400"):
                await provider.generate_audio(text="test")

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        """Missing API key should raise ValueError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        provider = OpenRouterProvider()
        with pytest.raises(ValueError, match="API key required"):
            await provider.generate_audio(text="test")

    @pytest.mark.asyncio
    async def test_malformed_sse_lines_skipped(self, monkeypatch):
        """Non-JSON SSE lines and non-data lines should be safely skipped."""
        lines = [
            b"event: ping\n",
            b"data: not_json\n",
            b'data: {"choices": []}\n',
            f"data: {json.dumps(_audio_event(b64_chunk='QUFB'))}\n".encode(),
            b"data: [DONE]\n",
        ]
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_audio(text="test")

        assert result.has_audio
        assert result.audio.data == "QUFB"


# =============================================================================
# generate_music tests
# =============================================================================


class TestGenerateMusic:
    """Tests for music generation."""

    @pytest.mark.asyncio
    async def test_abc_generate_music_raises_not_implemented(self):
        """MediaProvider.generate_music should raise NotImplementedError by default."""
        # FalProvider and LiteLLMProvider don't override generate_music
        fal = FalProvider()
        litellm_p = LiteLLMProvider()

        with pytest.raises(NotImplementedError, match="does not support music"):
            await fal.generate_music("test")

        with pytest.raises(NotImplementedError, match="does not support music"):
            await litellm_p.generate_music("test")

    @pytest.mark.asyncio
    async def test_openrouter_generate_music_streams(self, monkeypatch):
        """OpenRouterProvider.generate_music should parse SSE like generate_audio."""
        chunk = base64.b64encode(b"music_data").decode()
        events = [_audio_event(b64_chunk=chunk, transcript="Generated music")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(
                prompt="jazz piano",
                model="google/lyria-3-pro",
                duration=30,
            )

        assert result.has_audio
        assert result.audio.data == chunk
        assert result.text == "Generated music"

        # Verify duration was included in prompt
        payload = fake_session._last_post_kwargs["json"]
        assert "30 seconds" in payload["messages"][0]["content"]
        assert payload["model"] == "google/lyria-3-pro"

    @pytest.mark.asyncio
    async def test_openrouter_generate_music_default_model(self, monkeypatch):
        """generate_music should default to google/lyria-3-pro."""
        events = [_audio_event(b64_chunk="AAAA")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_music(prompt="test")

        payload = fake_session._last_post_kwargs["json"]
        assert payload["model"] == "google/lyria-3-pro"

    @pytest.mark.asyncio
    async def test_openrouter_generate_music_strips_prefix(self, monkeypatch):
        """openrouter/ prefix should be stripped from music model."""
        events = [_audio_event(b64_chunk="AAAA")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_music(
                prompt="test",
                model="openrouter/google/lyria-3-pro",
            )

        payload = fake_session._last_post_kwargs["json"]
        assert payload["model"] == "google/lyria-3-pro"


# =============================================================================
# supported_modalities tests
# =============================================================================


class TestSupportedModalities:
    """Test supported_modalities property."""

    def test_openrouter_includes_audio_and_music(self):
        """OpenRouterProvider.supported_modalities should include audio and music."""
        provider = OpenRouterProvider()
        mods = provider.supported_modalities
        assert "audio" in mods
        assert "music" in mods
        assert "image" in mods

    def test_fal_modalities_unchanged(self):
        """FalProvider modalities should still be image, audio, video."""
        provider = FalProvider()
        assert set(provider.supported_modalities) == {"image", "audio", "video"}

    def test_litellm_modalities_unchanged(self):
        """LiteLLMProvider modalities should still be image, audio."""
        provider = LiteLLMProvider()
        assert set(provider.supported_modalities) == {"image", "audio"}


# =============================================================================
# AgentAI.ai_generate_music tests
# =============================================================================


class TestAgentAIGenerateMusic:
    """Tests for ai_generate_music convenience method."""

    def test_method_exists(self):
        """AgentAI should have ai_generate_music method."""
        agent = StubAgent()
        ai = AgentAI(agent)
        assert hasattr(ai, "ai_generate_music")
        assert callable(ai.ai_generate_music)

    @pytest.mark.asyncio
    async def test_ai_generate_music_delegates_to_provider(self, monkeypatch):
        """ai_generate_music should delegate to cached OpenRouterProvider.generate_music."""
        expected = MultimodalResponse(
            text="music",
            audio=AudioOutput(data="AAAA", format="wav"),
            images=[],
            files=[],
        )
        mock_generate = AsyncMock(return_value=expected)

        agent = StubAgent()
        ai = AgentAI(agent)

        # Inject a mock into the cached provider slot
        mock_provider = MagicMock()
        mock_provider.generate_music = mock_generate
        ai._openrouter_provider_instance = mock_provider

        result = await ai.ai_generate_music(
            prompt="jazz", model="google/lyria-3-pro", duration=15
        )

        mock_generate.assert_called_once_with(
            prompt="jazz", model="google/lyria-3-pro", duration=15
        )
        assert result.has_audio
