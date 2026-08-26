"""
Tests for OpenRouter audio output and music generation.

Covers:
- SSE stream parsing for generate_audio
- generate_audio rejecting containers OpenRouter cannot deliver
- Audio chunk concatenation
- Transcript extraction
- Model prefix stripping
- generate_music transport (always SSE) and container sniffing
- generate_music on ABC (raises NotImplementedError by default)
- supported_modalities includes "audio" and "music"
- ai_generate_music convenience method on AgentAI
"""

import base64
import copy
import io
import json
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield.agent_ai import AgentAI
from agentfield.media_providers import (
    FalProvider,
    LiteLLMProvider,
    OpenRouterProvider,
    _label_streamed_audio,
    _sniff_audio_container,
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


def _prime_chat_audio_cache(provider: OpenRouterProvider, model: str) -> None:
    """Pre-populate model metadata so audio requests route via chat-completions.

    Without this, the provider tries to GET /api/v1/models/{id}/endpoints to
    discover routing — that's already mocked away in these tests.
    """
    stripped = model.removeprefix("openrouter/")
    provider._model_meta_cache[stripped] = {
        "id": stripped,
        "output_modalities": ["text", "audio"],
        "input_modalities": ["text"],
    }


class _BytesResponse:
    """Fake aiohttp response for /audio/speech (returns raw bytes)."""

    def __init__(self, body: bytes, status: int = 200, content_type: str = "audio/pcm"):
        self.status = status
        self._body = body
        self.headers = {"Content-Type": content_type}

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestOpenRouterGenerateAudio:
    """Tests for OpenRouterProvider.generate_audio (chat-completions SSE path)."""

    @pytest.mark.asyncio
    async def test_sse_stream_parsing_and_concatenation(self, monkeypatch):
        """Audio chunks from SSE should be concatenated correctly for chat-audio models."""
        # Two valid base64 chunks decode to "audio_part_1" + "audio_part_2".
        chunk1 = base64.b64encode(b"audio_part_1").decode()
        chunk2 = base64.b64encode(b"audio_part_2").decode()
        merged_b64 = base64.b64encode(b"audio_part_1" + b"audio_part_2").decode()

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
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="Say hello",
                model="openai/gpt-audio-mini",
                voice="alloy",
                format="pcm16",  # avoid pcm→wav rewrap so we can assert raw concat
            )

        assert result.has_audio
        assert result.audio.data == merged_b64
        assert result.audio.format == "pcm16"
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcript_extraction(self, monkeypatch):
        """Transcript text should be accumulated from SSE events."""
        # Use valid base64 chunks so the new merged-bytes path doesn't error.
        chunks = [
            base64.b64encode(b"AAAA").decode(),
            base64.b64encode(b"BBBB").decode(),
            base64.b64encode(b"CCCC").decode(),
        ]
        events = [
            _audio_event(b64_chunk=chunks[0], transcript="First "),
            _audio_event(b64_chunk=chunks[1], transcript="second "),
            _audio_event(b64_chunk=chunks[2], transcript="third."),
        ]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="test", model="openai/gpt-audio-mini", format="pcm16"
            )

        assert result.text == "First second third."

    @pytest.mark.asyncio
    async def test_model_prefix_stripping(self, monkeypatch):
        """openrouter/ prefix should be stripped from model before sending (chat path)."""
        events = [_audio_event(b64_chunk=base64.b64encode(b"AAAA").decode())]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            await provider.generate_audio(
                text="test",
                model="openrouter/openai/gpt-audio-mini",
                format="pcm16",
            )

        payload = fake_session._last_post_kwargs["json"]
        assert payload["model"] == "openai/gpt-audio-mini"
        assert not payload["model"].startswith("openrouter/")

    @pytest.mark.asyncio
    async def test_empty_stream_returns_no_audio(self, monkeypatch):
        """Empty SSE stream should return response with no audio (chat-audio path)."""
        lines = _make_sse_lines([])
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="test", model="openai/gpt-audio-mini", format="pcm16"
            )

        assert not result.has_audio
        assert result.text == "test"

    @pytest.mark.asyncio
    async def test_http_error_raises(self, monkeypatch):
        """Non-200 response should raise RuntimeError (chat-audio path)."""
        fake_resp = _FakeStreamResponse([], status=400)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            with pytest.raises(RuntimeError, match="failed.*400"):
                await provider.generate_audio(
                    text="test", model="openai/gpt-audio-mini", format="pcm16"
                )

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
        valid_b64 = base64.b64encode(b"AAA").decode()
        lines = [
            b"event: ping\n",
            b"data: not_json\n",
            b'data: {"choices": []}\n',
            f"data: {json.dumps(_audio_event(b64_chunk=valid_b64))}\n".encode(),
            b"data: [DONE]\n",
        ]
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="test", model="openai/gpt-audio-mini", format="pcm16"
            )

        assert result.has_audio
        assert result.audio.data == valid_b64


class TestOpenRouterChatAudioFormatGuard:
    """Chat-audio models can only deliver pcm16, so anything else fails fast.

    OpenRouter's gateway rejects ``stream: false`` for *any* audio-output
    request (whatever the model), and the streaming transport only ever emits
    pcm16 deltas — there is no combination that yields mp3/flac/opus from a
    chat-audio model, so asking for one must raise before a request is spent.
    """

    @pytest.mark.parametrize("fmt", ["mp3", "flac", "opus", "pcm"])
    @pytest.mark.asyncio
    async def test_unsupported_format_raises_before_any_request(self, monkeypatch, fmt):
        """A1: ValueError naming the format, and nothing goes over the wire."""
        fake_session = _FakeSession(_FakeStreamResponse(_make_sse_lines([])))

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            with pytest.raises(ValueError) as excinfo:
                await provider.generate_audio(
                    text="Say hi", model="openai/gpt-audio-mini", format=fmt
                )

        message = str(excinfo.value)
        assert fmt in message
        assert "pcm16" in message
        assert "wav" in message
        assert not hasattr(fake_session, "_last_post_kwargs"), (
            "no request may be spent on a format that cannot come back"
        )

    @pytest.mark.asyncio
    async def test_unknown_format_still_falls_back_to_wav(self, monkeypatch):
        """A1: an unrecognised format is normalised to wav, not rejected."""
        pcm = b"\x00\x01" * 32
        events = [_audio_event(b64_chunk=base64.b64encode(pcm).decode())]
        fake_session = _FakeSession(_FakeStreamResponse(_make_sse_lines(events)))

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="hi", model="openai/gpt-audio-mini", format="aac"
            )

        assert fake_session._last_post_kwargs["json"]["audio"]["format"] == "pcm16"
        assert result.audio.format == "wav"

    @pytest.mark.asyncio
    async def test_wav_streams_as_pcm16_and_returns_riff(self, monkeypatch):
        """A3: wav is still requested as pcm16 and wrapped client-side."""
        pcm = b"\x00\x01" * 32
        chunk = base64.b64encode(pcm).decode()
        events = [_audio_event(b64_chunk=chunk, transcript="wav path")]
        fake_session = _FakeSession(_FakeStreamResponse(_make_sse_lines(events)))

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="hi", model="openai/gpt-audio-mini", format="wav"
            )

        sent = fake_session._last_post_kwargs["json"]
        assert sent["stream"] is True
        assert sent["audio"]["format"] == "pcm16"

        assert result.audio.format == "wav"
        wav_bytes = base64.b64decode(result.audio.data)
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        with wave.open(io.BytesIO(wav_bytes)) as handle:
            assert handle.getframerate() == 24000
            assert handle.getnchannels() == 1
        assert pcm in wav_bytes

    @pytest.mark.asyncio
    async def test_pcm16_format_still_uses_sse_streaming(self, monkeypatch):
        """A2: pcm16 keeps streaming and returns the base64 verbatim."""
        chunk1 = base64.b64encode(b"pcm_audio").decode()
        events = [_audio_event(b64_chunk=chunk1, transcript="SSE path")]
        sse_lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(sse_lines, status=200)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "openai/gpt-audio-mini")
            result = await provider.generate_audio(
                text="SSE", model="openai/gpt-audio-mini", format="pcm16"
            )

        sent_payload = fake_session._last_post_kwargs["json"]
        assert sent_payload["stream"] is True, "pcm16 should use SSE streaming"
        assert sent_payload["audio"]["format"] == "pcm16"
        assert result.has_audio
        assert result.audio.data == chunk1
        assert result.audio.format == "pcm16"
        assert result.text == "SSE path"


class TestOpenRouterAudioSpeechEndpoint:
    """Tests for OpenRouterProvider.generate_audio routing to /audio/speech."""

    @pytest.mark.asyncio
    async def test_tts_only_model_routes_to_audio_speech(self, monkeypatch):
        """A model with output_modalities=['speech'] (e.g. Kokoro) hits /audio/speech."""
        pcm_body = b"\x00\x01" * 1000  # 2KB of fake PCM
        speech_resp = _BytesResponse(pcm_body, status=200, content_type="audio/pcm")
        fake_session = _FakeSession(speech_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            # Pre-populate cache as a TTS-only model so routing is deterministic.
            provider._model_meta_cache["hexgrad/kokoro-82m"] = {
                "id": "hexgrad/kokoro-82m",
                "output_modalities": ["speech"],
                "input_modalities": ["text"],
            }
            result = await provider.generate_audio(
                text="hello",
                model="openrouter/hexgrad/kokoro-82m",
                voice="af_bella",
                format="wav",
            )

        # Verify endpoint and payload.
        assert "audio/speech" in fake_session._last_post_url
        payload = fake_session._last_post_kwargs["json"]
        assert payload["model"] == "hexgrad/kokoro-82m"
        assert payload["voice"] == "af_bella"
        assert payload["response_format"] == "pcm"  # wav→pcm wire, wrapped client-side
        assert payload["input"] == "hello"

        # WAV wrapping should produce a RIFF/WAVE container.
        assert result.has_audio
        decoded = base64.b64decode(result.audio.data)
        assert decoded[:4] == b"RIFF"
        assert decoded[8:12] == b"WAVE"

    @pytest.mark.asyncio
    async def test_audio_speech_passes_speed_and_extra(self, monkeypatch):
        """speed and extra are forwarded to /audio/speech body."""
        speech_resp = _BytesResponse(b"\x00\x01" * 100)
        fake_session = _FakeSession(speech_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            provider._model_meta_cache["openai/gpt-4o-mini-tts"] = {
                "id": "openai/gpt-4o-mini-tts",
                "output_modalities": ["speech"],
                "input_modalities": ["text"],
            }
            await provider.generate_audio(
                text="hi",
                model="openai/gpt-4o-mini-tts",
                speed=1.25,
                extra={"language": "en-US"},
                format="mp3",
            )

        payload = fake_session._last_post_kwargs["json"]
        assert payload["speed"] == 1.25
        assert payload["language"] == "en-US"
        # A4: the chat-audio pcm16 guard must not reach the /audio/speech
        # route, which serves mp3/flac/opus natively.
        assert payload["response_format"] == "mp3"


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
            _prime_chat_audio_cache(provider, "google/lyria-3-pro-preview")
            result = await provider.generate_music(
                prompt="jazz piano",
                model="google/lyria-3-pro-preview",
                duration=30,
                # Ask for pcm16 so the payload (which carries no container
                # header) is returned verbatim and we can compare base64.
                format="pcm16",
            )

        assert result.has_audio
        assert result.audio.data == chunk
        assert result.text == "Generated music"

        # Verify duration was included in prompt
        payload = fake_session._last_post_kwargs["json"]
        assert payload["stream"] is True
        assert "30 seconds" in payload["messages"][0]["content"]
        assert payload["model"] == "google/lyria-3-pro-preview"

    @pytest.mark.asyncio
    async def test_openrouter_generate_music_default_model(self, monkeypatch):
        """generate_music should default to google/lyria-3-pro-preview."""
        events = [_audio_event(b64_chunk="AAAA")]
        lines = _make_sse_lines(events)
        fake_resp = _FakeStreamResponse(lines)
        fake_session = _FakeSession(fake_resp)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            _prime_chat_audio_cache(provider, "google/lyria-3-pro-preview")
            await provider.generate_music(prompt="test")

        payload = fake_session._last_post_kwargs["json"]
        assert payload["model"] == "google/lyria-3-pro-preview"

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
            _prime_chat_audio_cache(provider, "google/lyria-3-pro-preview")
            await provider.generate_music(
                prompt="test",
                model="openrouter/google/lyria-3-pro-preview",
            )

        payload = fake_session._last_post_kwargs["json"]
        assert payload["model"] == "google/lyria-3-pro-preview"


class TestSniffAudioContainer:
    """_sniff_audio_container identifies a container from its magic bytes."""

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b"ID3\x03\x00\x00\x00\x00tag", "mp3"),
            (b"\xff\xfb\x90\x64", "mp3"),
            (b"\xff\xf3\x90\x64", "mp3"),
            (b"\xff\xf2\x90\x64", "mp3"),
            (b"\xff\xfa\x90\x64", "mp3"),
            (b"RIFF\x24\x00\x00\x00WAVEfmt ", "wav"),
            (b"fLaC\x00\x00\x00\x22", "flac"),
            (b"OggS\x00\x02\x00\x00", "ogg"),
            # Raw little-endian PCM16 frames carry no signature at all.
            (b"\x00\x01\x00\x01\x00\x01", None),
            # RIFF that is not WAVE must not be claimed as wav.
            (b"RIFF\x24\x00\x00\x00AVI LIST", None),
            (b"", None),
            (b"ID", None),
        ],
    )
    def test_signature_detection(self, payload, expected):
        assert _sniff_audio_container(payload) == expected


class TestLabelStreamedAudio:
    """_label_streamed_audio degrades safely on inputs it cannot inspect."""

    def test_empty_payload_keeps_requested_format(self):
        assert _label_streamed_audio("", "wav") == ("", "wav")

    def test_undecodable_base64_is_passed_through(self):
        # "abcde" is not valid base64 (length % 4 == 1) — the payload must be
        # handed back untouched rather than crashing the caller.
        assert _label_streamed_audio("abcde", "wav") == ("abcde", "wav")


class TestOpenRouterMusicTransportAndLabelling:
    """generate_music always streams, and labels the result by the real bytes.

    OpenRouter's music models require ``stream: true`` for any audio output
    ("Audio output requires stream: true" otherwise) and ignore
    ``audio.format`` — ``google/lyria-3-*`` answers with MP3 whatever was
    asked for — so the container is sniffed from the payload instead of being
    assumed from the request.
    """

    @staticmethod
    def _sse_session(payload: bytes, transcript: str = ""):
        lines = _make_sse_lines(
            [
                _audio_event(
                    b64_chunk=base64.b64encode(payload).decode(),
                    transcript=transcript,
                )
            ]
        )
        return _FakeSession(_FakeStreamResponse(lines))

    @staticmethod
    def _capture_warnings(monkeypatch) -> list:
        captured: list = []
        monkeypatch.setattr(
            "agentfield.logger.log_warn",
            lambda message, **kwargs: captured.append(message),
        )
        return captured

    @pytest.mark.parametrize("fmt", [None, "wav", "pcm16", "mp3", "flac", "opus"])
    @pytest.mark.asyncio
    async def test_every_format_streams_with_the_requested_wire_format(
        self, monkeypatch, fmt
    ):
        """M1/M2/M10: stream is always True and audio.format is untouched."""
        self._capture_warnings(monkeypatch)
        fake_session = self._sse_session(b"ID3\x03\x00\x00\x00\x00lyria")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        call_kwargs = {} if fmt is None else {"format": fmt}
        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_music(prompt="a jingle", **call_kwargs)

        sent = fake_session._last_post_kwargs["json"]
        assert fake_session._last_post_url.endswith("/chat/completions")
        assert sent["stream"] is True, "music models reject stream=false"
        assert sent["audio"]["format"] == (fmt or "wav")

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b"ID3\x03\x00\x00\x00\x00lyria-mp3-bytes", "mp3"),
            (b"\xff\xfb\x90\x64frame-sync-mp3", "mp3"),
            (b"RIFF\x24\x00\x00\x00WAVEfmt payload", "wav"),
            (b"fLaC\x00\x00\x00\x22flac-bytes", "flac"),
            (b"OggS\x00\x02\x00\x00ogg-bytes", "ogg"),
        ],
    )
    @pytest.mark.asyncio
    async def test_detected_container_wins_over_requested_format(
        self, monkeypatch, payload, expected
    ):
        """M4/M6: the container found in the bytes is what gets reported."""
        warnings = self._capture_warnings(monkeypatch)
        fake_session = self._sse_session(payload, "take 1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            # No format -> "wav" is requested; lyria ignores it.
            result = await provider.generate_music(prompt="a jingle")

        assert result.has_audio
        assert result.audio.format == expected
        assert base64.b64decode(result.audio.data) == payload, (
            "a payload that already has a container must not be re-wrapped"
        )
        assert result.text == "take 1"

        if expected == "wav":
            assert warnings == []
        else:
            assert len(warnings) == 1, "exactly one mismatch warning"
            assert expected in warnings[0]
            assert "wav" in warnings[0]

    @pytest.mark.asyncio
    async def test_raw_pcm_with_wav_requested_is_wrapped_at_24khz(self, monkeypatch):
        """M5/M6: no container + wav asked for -> 24 kHz mono RIFF, no warning."""
        warnings = self._capture_warnings(monkeypatch)
        pcm = b"\x00\x01" * 64
        fake_session = self._sse_session(pcm, "pcm take")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(prompt="ambient pads")

        assert result.audio.format == "wav"
        wav_bytes = base64.b64decode(result.audio.data)
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        with wave.open(io.BytesIO(wav_bytes)) as handle:
            assert handle.getframerate() == 24000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
        assert pcm in wav_bytes, "the streamed pcm payload must survive the wrap"
        assert warnings == [], (
            "wrapping raw pcm into the requested wav is not a mismatch"
        )

    @pytest.mark.asyncio
    async def test_raw_pcm_with_pcm16_requested_is_returned_verbatim(self, monkeypatch):
        """M5/M6: no container + pcm16 asked for -> untouched base64, no warning."""
        warnings = self._capture_warnings(monkeypatch)
        pcm = b"\x00\x01" * 64
        chunk = base64.b64encode(pcm).decode()
        fake_session = self._sse_session(pcm, "pcm take")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(
                prompt="ambient pads", format="pcm16"
            )

        assert result.audio.format == "pcm16"
        assert result.audio.data == chunk
        assert warnings == []

    @pytest.mark.asyncio
    async def test_raw_pcm_with_mp3_requested_passes_through_and_warns(
        self, monkeypatch
    ):
        """M5/M6: no container + mp3 asked for -> passthrough, labelled pcm16, one warning."""
        warnings = self._capture_warnings(monkeypatch)
        pcm = b"\x00\x01" * 64
        chunk = base64.b64encode(pcm).decode()
        fake_session = self._sse_session(pcm)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(prompt="a riff", format="mp3")

        assert result.audio.data == chunk, "raw pcm must not be re-encoded"
        assert result.audio.format == "pcm16", "raw frames are not mp3, say so"
        assert len(warnings) == 1
        assert "pcm16" in warnings[0]
        assert "mp3" in warnings[0]

    @pytest.mark.asyncio
    async def test_opus_request_satisfied_by_ogg_container_logs_nothing(
        self, monkeypatch
    ):
        """M6: Opus travels in an Ogg container - OggS for format=opus is no mismatch."""
        warnings = self._capture_warnings(monkeypatch)
        ogg = b"OggS" + b"\x00" * 60
        fake_session = self._sse_session(ogg)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(prompt="a riff", format="opus")

        assert result.audio.format == "ogg"
        assert warnings == []

    @pytest.mark.asyncio
    async def test_matching_container_logs_nothing(self, monkeypatch):
        """M6: mp3 asked for and mp3 delivered -> silence."""
        warnings = self._capture_warnings(monkeypatch)
        fake_session = self._sse_session(b"ID3\x03\x00\x00\x00\x00lyria-mp3")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(prompt="a riff", format="mp3")

        assert result.audio.format == "mp3"
        assert warnings == []

    @pytest.mark.parametrize("fmt", ["wav", "pcm16", "mp3", "opus"])
    @pytest.mark.asyncio
    async def test_request_shape_is_identical_across_formats(self, monkeypatch, fmt):
        """M7: prompt/messages/model/modalities do not vary with the format."""
        self._capture_warnings(monkeypatch)
        fake_session = self._sse_session(b"\x00\x01\x00\x01")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            await provider.generate_music(
                prompt="lo-fi beats",
                model="openrouter/google/lyria-3-pro-preview",
                duration=45,
                format=fmt,
            )

        sent = fake_session._last_post_kwargs["json"]
        assert sent["model"] == "google/lyria-3-pro-preview"
        assert sent["messages"] == [
            {"role": "user", "content": "lo-fi beats (duration: 45 seconds)"}
        ]
        assert sent["modalities"] == ["text", "audio"]
        assert sent["stream"] is True

    @pytest.mark.asyncio
    async def test_duration_validation_runs_before_any_request(self, monkeypatch):
        """M7: duration validation still precedes the HTTP call."""
        fake_session = self._sse_session(b"\x00\x01\x00\x01")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            with pytest.raises(ValueError, match="duration must be"):
                await provider.generate_music(prompt="x", duration=601, format="mp3")

        assert not hasattr(fake_session, "_last_post_kwargs")

    @pytest.mark.asyncio
    async def test_empty_stream_returns_no_audio(self, monkeypatch):
        """M8: a stream with no audio deltas yields no audio and does not crash."""
        self._capture_warnings(monkeypatch)
        fake_session = _FakeSession(_FakeStreamResponse(_make_sse_lines([])))
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            result = await provider.generate_music(prompt="silence please")

        assert not result.has_audio
        assert result.audio is None
        assert result.text == "silence please"

    @pytest.mark.asyncio
    async def test_http_error_mentions_music_and_status(self, monkeypatch):
        """M9: a non-200 raises RuntimeError naming the label and the status."""
        fake_session = _FakeSession(_FakeStreamResponse([], status=402))
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with patch("aiohttp.ClientSession", return_value=fake_session):
            provider = OpenRouterProvider()
            with pytest.raises(RuntimeError) as excinfo:
                await provider.generate_music(prompt="a riff", format="opus")

        message = str(excinfo.value)
        assert "music" in message
        assert "402" in message
        # M1: even the failing request went out as a stream.
        assert fake_session._last_post_kwargs["json"]["stream"] is True


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
