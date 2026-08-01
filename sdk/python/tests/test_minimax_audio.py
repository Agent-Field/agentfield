from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.agent_ai import AgentAI
from agentfield.media_providers import (
    MINIMAX_CN_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    MiniMaxProvider,
)
from agentfield.media_router import MediaRouter


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class CaptureSession:
    def __init__(self, response):
        self.response = response
        self.request = None

    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_minimax_audio_hex_output_uses_global_endpoint_and_request_fields(
    monkeypatch,
):
    audio_bytes = b"generated-speech"
    payload = {
        "data": {"audio": audio_bytes.hex(), "status": 2},
        "extra_info": {"audio_format": "mp3"},
        "base_resp": {"status_code": 0},
    }
    session = CaptureSession(FakeResponse(payload))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value")

    result = await provider.generate_audio(
        text="Read this sentence clearly",
        model="minimax/speech-2.8-turbo",
        voice="English_Graceful_Lady",
        format="mp3",
        speed=1.25,
        language_boost="English",
        output_format="hex",
        pronunciation_dict={"tone": ["read/reed"]},
        audio_setting={"sample_rate": 32000},
        voice_modify={"pitch": 10},
        subtitle_enable=True,
    )

    assert session.request[0] == f"{MINIMAX_GLOBAL_BASE_URL}/t2a_v2"
    assert session.request[1]["json"] == {
        "model": "speech-2.8-turbo",
        "text": "Read this sentence clearly",
        "stream": False,
        "output_format": "hex",
        "audio_setting": {"sample_rate": 32000, "format": "mp3"},
        "language_boost": "English",
        "voice_setting": {
            "voice_id": "English_Graceful_Lady",
            "speed": 1.25,
        },
        "pronunciation_dict": {"tone": ["read/reed"]},
        "voice_modify": {"pitch": 10},
        "subtitle_enable": True,
    }
    assert session.request[1]["headers"]["Authorization"] == "Bearer unit-value"
    assert result.audio.format == "mp3"
    assert result.audio.url is None
    assert result.audio.get_bytes() == audio_bytes
    assert result.raw_response["data"]["status"] == 2


@pytest.mark.asyncio
async def test_minimax_audio_url_output_uses_cn_endpoint_and_default_model(monkeypatch):
    audio_url = "https://example.test/generated.flac"
    session = CaptureSession(
        FakeResponse(
            {
                "data": {"audio": audio_url, "status": 2},
                "base_resp": {"status_code": 0},
            }
        )
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value", base_url=MINIMAX_CN_BASE_URL)

    result = await provider.generate_audio(
        text="Generate regional speech",
        format="flac",
        output_format="url",
        voice_setting={"voice_id": "regional-voice"},
    )

    assert session.request[0] == f"{MINIMAX_CN_BASE_URL}/t2a_v2"
    assert session.request[1]["json"] == {
        "model": "speech-2.8-hd",
        "text": "Generate regional speech",
        "stream": False,
        "output_format": "url",
        "audio_setting": {"format": "flac"},
        "voice_setting": {"voice_id": "regional-voice"},
    }
    assert result.audio.data is None
    assert result.audio.url == audio_url
    assert result.audio.format == "flac"


@pytest.mark.asyncio
async def test_minimax_audio_validates_inputs_and_api_errors(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    provider = MiniMaxProvider()
    with pytest.raises(ValueError, match="API key required"):
        await provider.generate_audio("Audio")

    provider = MiniMaxProvider(api_key="unit-value")
    with pytest.raises(ValueError, match="format must be"):
        await provider.generate_audio("Audio", format="aac")
    with pytest.raises(ValueError, match="streaming TTS"):
        await provider.generate_audio("Audio", stream=True)

    session = CaptureSession(
        FakeResponse(
            {
                "base_resp": {
                    "status_code": 1004,
                    "status_msg": "authentication failed",
                }
            }
        )
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="authentication failed"):
        await provider.generate_audio("Audio")


@pytest.mark.asyncio
async def test_agent_ai_routes_minimax_audio_models():
    agent = SimpleNamespace(
        ai_config=SimpleNamespace(audio_model="minimax/speech-2.8-hd")
    )
    ai = AgentAI(agent)
    generate_audio = AsyncMock(return_value="generated-audio")
    provider = SimpleNamespace(
        name="minimax",
        supported_modalities=["audio"],
        generate_audio=generate_audio,
    )
    router = MediaRouter()
    router.register("minimax/", provider)
    ai._media_router_instance = router

    result = await ai.ai_generate_audio(
        "Route this speech",
        voice="target-voice",
        format="pcm",
        speed=1.1,
        language_boost="English",
    )

    assert result == "generated-audio"
    generate_audio.assert_awaited_once_with(
        text="Route this speech",
        model="minimax/speech-2.8-hd",
        voice="target-voice",
        format="pcm",
        speed=1.1,
        language_boost="English",
    )
