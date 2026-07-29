from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield import AIConfig, MiniMaxProvider, get_provider
from agentfield.agent_ai import AgentAI
from agentfield.media_providers import (
    MINIMAX_CN_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
)
from tests.helpers import StubAgent


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
    def __init__(self, submit_response, get_responses):
        self.submit_response = submit_response
        self.get_responses = list(get_responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.submit_response

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.get_responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_minimax_video_lifecycle_uses_cn_endpoint_and_request_shape(monkeypatch):
    session = CaptureSession(
        FakeResponse({"task_id": "task-123", "base_resp": {"status_code": 0}}),
        [
            FakeResponse(
                {
                    "status": "Success",
                    "file_id": "file-123",
                    "base_resp": {"status_code": 0},
                }
            ),
            FakeResponse(
                {
                    "file": {
                        "filename": "result.mp4",
                        "download_url": "https://cdn.example.com/result.mp4",
                    },
                    "base_resp": {"status_code": 0},
                }
            ),
        ],
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)

    provider = MiniMaxProvider(api_key="unit-value", base_url=MINIMAX_CN_BASE_URL)
    result = await provider.generate_video(
        prompt="A camera moves through a city",
        model="minimax/video-model",
        image_url="https://cdn.example.com/frame.png",
        duration=6.0,
        resolution="1080p",
        extra={"prompt_optimizer": False},
        poll_interval=0,
    )

    assert session.calls[0][0:2] == (
        "post",
        f"{MINIMAX_CN_BASE_URL}/video_generation",
    )
    assert session.calls[0][2]["json"] == {
        "model": "video-model",
        "prompt": "A camera moves through a city",
        "first_frame_image": "https://cdn.example.com/frame.png",
        "duration": 6,
        "resolution": "1080P",
        "prompt_optimizer": False,
    }
    assert session.calls[1][0:2] == (
        "get",
        f"{MINIMAX_CN_BASE_URL}/query/video_generation",
    )
    assert session.calls[1][2]["params"] == {"task_id": "task-123"}
    assert session.calls[2][0:2] == (
        "get",
        f"{MINIMAX_CN_BASE_URL}/files/retrieve",
    )
    assert session.calls[2][2]["params"] == {"file_id": "file-123"}
    assert result.files[0].url == "https://cdn.example.com/result.mp4"
    assert result.videos[0].filename == "result.mp4"
    assert result.videos[0].resolution == "1080P"


@pytest.mark.asyncio
async def test_minimax_video_checks_api_errors_and_failed_tasks(monkeypatch):
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    error_session = CaptureSession(
        FakeResponse(
            {
                "base_resp": {
                    "status_code": 1004,
                    "status_msg": "authentication failed",
                }
            }
        ),
        [],
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: error_session)
    provider = MiniMaxProvider(api_key="unit-value")

    with pytest.raises(RuntimeError, match="authentication failed"):
        await provider.generate_video(
            prompt="A landscape",
            model="minimax/video-model",
            poll_interval=0,
        )
    assert error_session.calls[0][1] == f"{MINIMAX_GLOBAL_BASE_URL}/video_generation"

    failed_session = CaptureSession(
        FakeResponse({"task_id": "task-456", "base_resp": {"status_code": 0}}),
        [
            FakeResponse(
                {
                    "status": "Fail",
                    "error_message": "generation rejected",
                    "base_resp": {"status_code": 0},
                }
            )
        ],
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: failed_session)

    with pytest.raises(RuntimeError, match="generation rejected"):
        await provider.generate_video(
            prompt="A landscape",
            model="minimax/video-model",
            poll_interval=0,
        )


@pytest.mark.asyncio
async def test_minimax_video_validates_credentials_model_and_duration(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    provider = MiniMaxProvider()

    with pytest.raises(ValueError, match="API key required"):
        await provider.generate_video("A landscape", model="minimax/video-model")

    provider = MiniMaxProvider(api_key="unit-value")
    with pytest.raises(ValueError, match="explicit model"):
        await provider.generate_video("A landscape")
    with pytest.raises(ValueError, match="whole number"):
        await provider.generate_video(
            "A landscape",
            model="minimax/video-model",
            duration=6.5,
        )
    with pytest.raises(NotImplementedError, match="image generation"):
        await provider.generate_image("An image")
    with pytest.raises(NotImplementedError, match="audio generation"):
        await provider.generate_audio("Audio")


@pytest.mark.asyncio
async def test_minimax_video_rejects_extra_and_kwargs_overriding_validated_fields():
    provider = MiniMaxProvider(api_key="unit-value")

    with pytest.raises(ValueError, match="duration"):
        await provider.generate_video(
            "A landscape",
            model="minimax/video-model",
            extra={"duration": 3.5},
        )
    with pytest.raises(ValueError, match="first_frame_image"):
        await provider.generate_video(
            "A landscape",
            model="minimax/video-model",
            image_url="https://cdn.example.com/frame.png",
            first_frame_image="https://cdn.example.com/other.png",
        )
    with pytest.raises(ValueError, match="model"):
        await provider.generate_video(
            "A landscape",
            model="minimax/video-model",
            extra={"model": "other-model"},
        )


@pytest.mark.asyncio
async def test_agent_ai_routes_minimax_video_models():
    agent = StubAgent()
    agent.ai_config = SimpleNamespace(
        fal_api_key=None,
        minimax_api_key="unit-value",
        minimax_base_url=MINIMAX_GLOBAL_BASE_URL,
        video_model="minimax/video-model",
    )
    ai = AgentAI(agent)
    generate_video = AsyncMock(return_value="minimax-video")
    ai._minimax_provider_instance = SimpleNamespace(
        name="minimax",
        supported_modalities=["video"],
        generate_video=generate_video,
    )

    result = await ai.ai_generate_video("A landscape")

    assert result == "minimax-video"
    generate_video.assert_awaited_once_with(
        prompt="A landscape",
        model="minimax/video-model",
        image_url=None,
        duration=None,
    )


def test_minimax_provider_configuration_and_registry():
    config = AIConfig(
        minimax_api_key="unit-value",
        minimax_base_url=MINIMAX_CN_BASE_URL,
    )
    assert config.minimax_api_key == "unit-value"
    assert config.minimax_base_url == MINIMAX_CN_BASE_URL

    provider = get_provider(
        "minimax",
        api_key="unit-value",
        base_url=MINIMAX_GLOBAL_BASE_URL,
    )
    assert isinstance(provider, MiniMaxProvider)
    assert provider.supported_modalities == ["video"]
