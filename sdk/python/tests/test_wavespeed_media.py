import itertools
from types import SimpleNamespace

import pytest

from agentfield.agent_ai import AgentAI
from agentfield.types import AIConfig
from agentfield.media_providers import (
    WAVESPEED_BASE_URL,
    WaveSpeedProvider,
    get_provider,
)


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


class ScriptedSession:
    """Fake aiohttp session returning one POST response and queued GET responses."""

    def __init__(self, post_response, get_responses=()):
        self.post_response = post_response
        self.get_responses = list(get_responses)
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.post_response

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def envelope(data, code=200, message="success"):
    return {"code": code, "message": message, "data": data}


def created(prediction_id="pred-1"):
    return FakeResponse(
        envelope({"id": prediction_id, "status": "created", "outputs": []})
    )


def completed(outputs, prediction_id="pred-1"):
    return FakeResponse(
        envelope({"id": prediction_id, "status": "completed", "outputs": outputs})
    )


@pytest.fixture
def no_sleep(monkeypatch):
    async def _sleep(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)


def install(monkeypatch, session):
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)


@pytest.mark.asyncio
async def test_generate_image_submits_default_model_and_polls_result(
    monkeypatch, no_sleep
):
    session = ScriptedSession(
        created(),
        [
            FakeResponse(
                envelope({"id": "pred-1", "status": "processing", "outputs": []})
            ),
            completed(["https://cdn.example.test/out.png"]),
        ],
    )
    install(monkeypatch, session)
    provider = WaveSpeedProvider(api_key="unit-value")

    result = await provider.generate_image(
        "A lighthouse at sunrise", aspect_ratio="16:9", resolution="2k"
    )

    url, kwargs = session.posts[0]
    assert url == f"{WAVESPEED_BASE_URL}/api/v3/bytedance/seedream-v5.5"
    assert kwargs["json"] == {
        "prompt": "A lighthouse at sunrise",
        "aspect_ratio": "16:9",
        "resolution": "2k",
    }
    assert kwargs["headers"]["Authorization"] == "Bearer unit-value"
    assert len(session.gets) == 2
    assert session.gets[0][0] == (
        f"{WAVESPEED_BASE_URL}/api/v3/predictions/pred-1/result"
    )
    assert [image.url for image in result.images] == [
        "https://cdn.example.test/out.png"
    ]
    assert result.raw_response["status"] == "completed"


@pytest.mark.asyncio
async def test_generate_image_strips_route_prefix_and_reads_env_key(
    monkeypatch, no_sleep
):
    session = ScriptedSession(created(), [completed(["https://x.test/a.jpeg"])])
    install(monkeypatch, session)
    monkeypatch.setenv("WAVESPEED_API_KEY", "env-value")
    provider = WaveSpeedProvider(base_url="https://proxy.example.test/")

    await provider.generate_image("A cat", model="wavespeed/vendor/some-model")

    url, kwargs = session.posts[0]
    assert url == "https://proxy.example.test/api/v3/vendor/some-model"
    assert kwargs["headers"]["Authorization"] == "Bearer env-value"


@pytest.mark.asyncio
async def test_missing_api_key_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    session = ScriptedSession(created())
    install(monkeypatch, session)

    with pytest.raises(ValueError, match="WAVESPEED_API_KEY"):
        await WaveSpeedProvider().generate_image("A cat")
    assert session.posts == []


@pytest.mark.asyncio
async def test_envelope_error_with_http_200_is_raised_without_polling(
    monkeypatch, no_sleep
):
    session = ScriptedSession(
        FakeResponse(envelope(None, code=400, message="prompt is required"))
    )
    install(monkeypatch, session)

    with pytest.raises(
        RuntimeError, match=r"submit failed \(400\): prompt is required"
    ):
        await WaveSpeedProvider(api_key="k").generate_image("")
    assert len(session.posts) == 1
    assert session.gets == []


@pytest.mark.asyncio
async def test_http_error_on_submit_is_not_retried(monkeypatch, no_sleep):
    session = ScriptedSession(FakeResponse({"message": "unauthorized"}, status=401))
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match=r"submit failed \(401\)"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")
    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_failed_prediction_raises_with_error(monkeypatch, no_sleep):
    session = ScriptedSession(
        created("pred-9"),
        [
            FakeResponse(
                envelope(
                    {
                        "id": "pred-9",
                        "status": "failed",
                        "outputs": [],
                        "error": "content rejected",
                    }
                )
            )
        ],
    )
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match="pred-9 failed: content rejected"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")


@pytest.mark.asyncio
async def test_transient_poll_errors_are_retried(monkeypatch, no_sleep):
    import aiohttp

    session = ScriptedSession(
        created(),
        [
            FakeResponse({"message": "bad gateway"}, status=502),
            aiohttp.ClientConnectionError("reset"),
            completed(["https://x.test/a.png"]),
        ],
    )
    install(monkeypatch, session)

    result = await WaveSpeedProvider(api_key="k").generate_image("A cat")

    assert len(session.posts) == 1
    assert len(session.gets) == 3
    assert result.images[0].url == "https://x.test/a.png"


@pytest.mark.asyncio
async def test_client_error_while_polling_is_raised(monkeypatch, no_sleep):
    session = ScriptedSession(
        created(), [FakeResponse({"message": "not found"}, status=404)]
    )
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match=r"result for pred-1 failed \(404\)"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")


@pytest.mark.asyncio
async def test_poll_timeout_raises(monkeypatch, no_sleep):
    processing = FakeResponse(
        envelope({"id": "pred-1", "status": "processing", "outputs": []})
    )
    session = ScriptedSession(created(), [processing] * 50)
    install(monkeypatch, session)
    clock = itertools.count(0.0, 1.0)
    monkeypatch.setattr("time.monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="pred-1"):
        await WaveSpeedProvider(api_key="k", timeout=2.0).generate_image("A cat")


@pytest.mark.asyncio
async def test_completed_without_outputs_raises(monkeypatch, no_sleep):
    session = ScriptedSession(created(), [completed([])])
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match="no outputs"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")


@pytest.mark.asyncio
async def test_invalid_submit_payload_raises(monkeypatch, no_sleep):
    session = ScriptedSession(FakeResponse(["not", "a", "dict"]))
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match="invalid response"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")


@pytest.mark.asyncio
async def test_generate_audio_uses_default_tts_inputs(monkeypatch, no_sleep):
    session = ScriptedSession(created(), [completed(["https://x.test/speech.mp3"])])
    install(monkeypatch, session)

    result = await WaveSpeedProvider(api_key="k").generate_audio(
        "Hello there", speed=1.0
    )

    url, kwargs = session.posts[0]
    assert url.endswith("/api/v3/wavespeed-ai/qwen3-tts/text-to-speech")
    assert kwargs["json"] == {
        "text": "Hello there",
        "language": "auto",
        "voice": "Vivian",
    }
    assert result.audio.url == "https://x.test/speech.mp3"
    assert result.audio.format == "mp3"


@pytest.mark.asyncio
async def test_generate_audio_passes_voice_and_model_fields(monkeypatch, no_sleep):
    session = ScriptedSession(created(), [completed(["https://x.test/speech"])])
    install(monkeypatch, session)

    result = await WaveSpeedProvider(api_key="k").generate_audio(
        "Hallo",
        model="wavespeed/vendor/other-tts",
        voice="Ryan",
        format="wav",
        language="German",
        speed=1.25,
    )

    url, kwargs = session.posts[0]
    assert url.endswith("/api/v3/vendor/other-tts")
    assert kwargs["json"] == {
        "text": "Hallo",
        "voice": "Ryan",
        "speed": 1.25,
        "language": "German",
    }
    assert result.audio.format == "wav"


@pytest.mark.asyncio
async def test_generate_video_text_and_image_defaults(monkeypatch, no_sleep):
    session = ScriptedSession(created(), [completed(["https://x.test/clip.mp4"])])
    install(monkeypatch, session)

    result = await WaveSpeedProvider(api_key="k").generate_video(
        "Waves at night", duration=5.0, resolution="720p", generate_audio=False
    )

    url, kwargs = session.posts[0]
    assert url.endswith("/api/v3/alibaba/wan-3.0/text-to-video")
    assert kwargs["json"] == {
        "prompt": "Waves at night",
        "duration": 5,
        "resolution": "720p",
        "generate_audio": False,
    }
    assert result.videos[0].url == "https://x.test/clip.mp4"
    assert result.videos[0].filename == "clip.mp4"
    assert result.files[0].mime_type == "video/mp4"

    session = ScriptedSession(created(), [completed(["https://x.test/i2v.mp4"])])
    install(monkeypatch, session)

    await WaveSpeedProvider(api_key="k").generate_video(
        "Slow zoom", image_url="https://x.test/in.png"
    )

    url, kwargs = session.posts[0]
    assert url.endswith("/api/v3/alibaba/wan-3.0/image-to-video")
    assert kwargs["json"] == {"prompt": "Slow zoom", "image": "https://x.test/in.png"}


def test_registry_and_metadata():
    provider = get_provider("wavespeed", api_key="k")
    assert isinstance(provider, WaveSpeedProvider)
    assert provider.name == "wavespeed"
    assert set(provider.supported_modalities) == {"image", "audio", "video"}


def _agent_ai():
    ai_config = AIConfig(wavespeed_api_key="config-value")
    return AgentAI(SimpleNamespace(ai_config=ai_config))


@pytest.mark.asyncio
async def test_agent_ai_routes_wavespeed_prefix_for_image(monkeypatch, no_sleep):
    session = ScriptedSession(created(), [completed(["https://x.test/a.png"])])
    install(monkeypatch, session)
    ai = _agent_ai()

    result = await ai.ai_generate_image(
        "A cat", model="wavespeed/bytedance/seedream-v5.5"
    )

    url, kwargs = session.posts[0]
    assert url.endswith("/api/v3/bytedance/seedream-v5.5")
    assert kwargs["headers"]["Authorization"] == "Bearer config-value"
    assert result.images[0].url == "https://x.test/a.png"


@pytest.mark.asyncio
async def test_agent_ai_routes_wavespeed_prefix_for_audio_and_video(
    monkeypatch, no_sleep
):
    ai = _agent_ai()

    session = ScriptedSession(created(), [completed(["https://x.test/s.wav"])])
    install(monkeypatch, session)
    audio = await ai.ai_generate_audio(
        "Hello", model="wavespeed/wavespeed-ai/qwen3-tts/text-to-speech"
    )
    assert session.posts[0][1]["json"] == {
        "text": "Hello",
        "language": "auto",
        "voice": "Vivian",
    }
    assert audio.audio.format == "wav"

    session = ScriptedSession(created(), [completed(["https://x.test/v.mp4"])])
    install(monkeypatch, session)
    video = await ai.ai_generate_video(
        "Waves", model="wavespeed/alibaba/wan-3.0/text-to-video"
    )
    assert session.posts[0][0].endswith("/api/v3/alibaba/wan-3.0/text-to-video")
    assert video.videos[0].url == "https://x.test/v.mp4"


@pytest.mark.asyncio
async def test_empty_model_after_prefix_raises(monkeypatch):
    session = ScriptedSession(created())
    install(monkeypatch, session)

    with pytest.raises(ValueError, match="requires a model"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat", model="wavespeed/")
    assert session.posts == []


@pytest.mark.asyncio
async def test_submit_without_prediction_id_raises(monkeypatch, no_sleep):
    session = ScriptedSession(FakeResponse(envelope({"status": "created"})))
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match="no prediction id"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")


@pytest.mark.asyncio
async def test_envelope_without_data_object_raises(monkeypatch, no_sleep):
    session = ScriptedSession(FakeResponse({"code": 200, "message": "success"}))
    install(monkeypatch, session)

    with pytest.raises(RuntimeError, match="invalid response"):
        await WaveSpeedProvider(api_key="k").generate_image("A cat")
