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
    def __init__(self, submit_response, get_responses, delete_response=None):
        self.submit_response = submit_response
        self.get_responses = list(get_responses)
        self.delete_response = delete_response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.submit_response

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.get_responses.pop(0)

    def delete(self, url, **kwargs):
        self.calls.append(("delete", url, kwargs))
        return self.delete_response

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
async def test_minimax_music_legacy_request_shape(monkeypatch):
    session = CaptureSession(
        FakeResponse(
            {
                "data": {"audio": "https://cdn.example.com/music.mp3"},
                "base_resp": {"status_code": 0},
            }
        ),
        [],
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)

    provider = MiniMaxProvider(api_key="unit-value", base_url=MINIMAX_CN_BASE_URL)
    result = await provider.generate_music(
        "music prompt",
        model="music-custom",
        duration=30,
        output_format="url",
        stream=False,
        sample_rate=48000,
        bitrate=320000,
        format="mp3",
        lyrics="la",
        aigc_watermark=True,
    )

    assert session.calls[0][0:2] == (
        "post",
        f"{MINIMAX_CN_BASE_URL}/music_generation",
    )
    assert session.calls[0][2]["json"] == {
        "model": "music-custom",
        "prompt": "music prompt",
        "output_format": "url",
        "stream": False,
        "audio_setting": {
            "sample_rate": 48000,
            "bitrate": 320000,
            "format": "mp3",
            "duration": 30,
        },
        "lyrics": "la",
        "aigc_watermark": True,
    }
    assert result.audio.url == "https://cdn.example.com/music.mp3"


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
async def test_minimax_video_validates_credentials_and_duration(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    provider = MiniMaxProvider()

    with pytest.raises(ValueError, match="API key required"):
        await provider.generate_video("A landscape", model="minimax/video-model")

    provider = MiniMaxProvider(api_key="unit-value")
    with pytest.raises(ValueError, match="requires duration"):
        await provider.generate_video("A landscape")
    with pytest.raises(ValueError, match="whole number"):
        await provider.generate_video(
            "A landscape",
            model="minimax/video-model",
            duration=6.5,
        )


@pytest.mark.asyncio
async def test_minimax_video_legacy_forwards_v1_optional_fields(monkeypatch):
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

    provider = MiniMaxProvider(api_key="unit-value")
    await provider.generate_video(
        prompt="A landscape",
        model="minimax/MiniMax-Hailuo-02",
        duration=6.0,
        ratio="16:9",
        callback_url="https://hook.example/cb",
        aigc_watermark=True,
        poll_interval=0,
    )

    assert session.calls[0][2]["json"] == {
        "model": "MiniMax-Hailuo-02",
        "prompt": "A landscape",
        "duration": 6,
        "ratio": "16:9",
        "callback_url": "https://hook.example/cb",
        "aigc_watermark": True,
    }


@pytest.mark.asyncio
async def test_minimax_video_legacy_rejects_h3_structured_content():
    provider = MiniMaxProvider(api_key="unit-value")

    with pytest.raises(ValueError, match="structured content requires the MiniMax-H3"):
        await provider.generate_video(
            "A landscape",
            model="minimax/MiniMax-Hailuo-02",
            duration=6.0,
            content=[{"type": "text", "text": "A landscape"}],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolution", "normalized_resolution", "expected_cost"),
    [("2k", "2K", 0.95), ("768p", "768P", 0.6)],
)
async def test_minimax_h3_video_lifecycle_uses_v2_endpoint_and_pricing(
    monkeypatch, resolution, normalized_resolution, expected_cost
):
    session = CaptureSession(
        FakeResponse({"task_id": "task-h3"}),
        [
            FakeResponse(
                {
                    "task": {
                        "id": "task-h3",
                        "model": "MiniMax-H3",
                        "status": "succeeded",
                        "content": {"url": "https://cdn.example.com/h3.mp4"},
                        "resolution": normalized_resolution,
                        "duration": 5,
                        "ratio": "16:9",
                        "usage": {
                            "total_seconds": 7,
                            "input_seconds": 2,
                            "output_seconds": 5,
                            "input_image_count": 6,
                        },
                    }
                }
            )
        ],
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)

    provider = MiniMaxProvider(api_key="unit-value")
    result = await provider.generate_video(
        prompt="A city wakes at dawn",
        duration=5,
        resolution=resolution,
        ratio="16:9",
        poll_interval=0,
    )

    assert session.calls[0][0:2] == (
        "post",
        "https://api.minimax.io/v2/video_generation",
    )
    assert session.calls[0][2]["json"] == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A city wakes at dawn"}],
        "resolution": normalized_resolution,
        "duration": 5,
        "ratio": "16:9",
    }
    assert session.calls[1][0:2] == (
        "get",
        "https://api.minimax.io/v2/query/video_generation/task-h3",
    )
    assert result.files[0].url == "https://cdn.example.com/h3.mp4"
    assert result.videos[0].has_audio is True
    assert result.videos[0].cost_usd == pytest.approx(expected_cost)
    assert result.cost_usd == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_minimax_h3_exposes_create_query_list_and_delete(monkeypatch):
    session = CaptureSession(
        FakeResponse({"task_id": "task-h3"}),
        [
            FakeResponse({"task": {"id": "task-h3", "status": "running"}}),
            FakeResponse({"items": [], "total": 0}),
        ],
        delete_response=FakeResponse(
            {"task_id": "task-h3", "action": "cancel", "status": "cancelled"}
        ),
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value", base_url=MINIMAX_CN_BASE_URL)
    content = [
        {"type": "text", "text": "Follow the reference performance"},
        {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/reference.png"},
            "role": "reference_image",
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://cdn.example.com/reference.mp4"},
            "role": "reference_video",
        },
        {
            "type": "audio_url",
            "audio_url": {"url": "https://cdn.example.com/reference.mp3"},
            "role": "reference_audio",
        },
    ]

    created = await provider.create_video_task(
        content=content,
        duration=8,
        aigc_watermark=True,
    )
    queried = await provider.query_video_task("task/h3")
    listed = await provider.list_video_tasks(
        page_num=2,
        page_size=10,
        status="running",
        task_ids=["task-h3", "task-h4"],
        model="minimax/MiniMax-H3",
        task_type="generation",
    )
    deleted = await provider.delete_video_task("task/h3")

    assert created == {"task_id": "task-h3"}
    assert queried["task"]["status"] == "running"
    assert listed == {"items": [], "total": 0}
    assert deleted["action"] == "cancel"
    assert session.calls[0][0:2] == (
        "post",
        "https://api.minimaxi.com/v2/video_generation",
    )
    assert session.calls[0][2]["json"] == {
        "model": "MiniMax-H3",
        "content": content,
        "resolution": "2K",
        "duration": 8,
        "ratio": "adaptive",
        "aigc_watermark": True,
    }
    assert session.calls[1][1].endswith("/v2/query/video_generation/task%2Fh3")
    assert session.calls[2][1] == "https://api.minimaxi.com/v2/query/video_generation"
    assert session.calls[2][2]["params"] == {
        "page_num": 2,
        "page_size": 10,
        "filter.status": "running",
        "filter.task_ids": ["task-h3", "task-h4"],
        "filter.model": "MiniMax-H3",
        "filter.task_type": "generation",
    }
    assert session.calls[3][0:2] == (
        "delete",
        "https://api.minimaxi.com/v2/video_generation/task%2Fh3",
    )


def test_minimax_h3_validates_content_duration_resolution_and_ratio(monkeypatch):
    provider = MiniMaxProvider(api_key="unit-value")
    text_content = [{"type": "text", "text": "A landscape"}]

    with pytest.raises(ValueError, match="between 4 and 15"):
        provider._build_h3_request(content=text_content, duration=3, ratio="16:9")
    with pytest.raises(ValueError, match="whole number"):
        provider._build_h3_request(content=text_content, duration=4.5, ratio="16:9")
    for resolution in ("768p", "2k"):
        body = provider._build_h3_request(
            content=text_content,
            duration=5,
            resolution=resolution,
            ratio="16:9",
        )
        assert body["resolution"] == resolution.upper()
    for resolution in ("720p", "4k"):
        with pytest.raises(ValueError, match="resolution must be 768P or 2K"):
            provider._build_h3_request(
                content=text_content,
                duration=5,
                resolution=resolution,
                ratio="16:9",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "expired"}, "task status"),
        ({"task_type": "bogus"}, "task type"),
    ],
)
async def test_minimax_h3_rejects_unsupported_list_filters(kwargs, message):
    provider = MiniMaxProvider(api_key="unit-value")

    with pytest.raises(ValueError, match=message):
        await provider.list_video_tasks(**kwargs)


def test_minimax_h3_rejects_unsupported_content(monkeypatch):
    provider = MiniMaxProvider(api_key="unit-value")
    text_content = [{"type": "text", "text": "A landscape"}]

    with pytest.raises(ValueError, match="resolution must be 768P or 2K"):
        provider._build_h3_request(
            content=text_content,
            duration=5,
            resolution="1080p",
            ratio="16:9",
        )
    with pytest.raises(ValueError, match="non-adaptive ratio"):
        provider._build_h3_request(content=text_content, duration=5)
    with pytest.raises(ValueError, match="requires a text entry"):
        provider._build_h3_request(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example.com/frame.png"},
                }
            ],
            duration=5,
        )
    with pytest.raises(ValueError, match="reference_audio requires"):
        provider._build_h3_request(
            content=[
                *text_content,
                {
                    "type": "audio_url",
                    "audio_url": {"url": "https://cdn.example.com/reference.mp3"},
                    "role": "reference_audio",
                },
            ],
            duration=5,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        provider._build_h3_request(
            content=[
                *text_content,
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example.com/first.png"},
                    "role": "first_frame",
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://cdn.example.com/reference.mp4"},
                    "role": "reference_video",
                },
            ],
            duration=5,
        )
    with pytest.raises(ValueError, match="China endpoint"):
        provider._build_h3_request(
            content=text_content,
            duration=5,
            ratio="16:9",
            aigc_watermark=True,
        )

    monkeypatch.setattr(provider, "H3_MAX_REQUEST_BODY_BYTES", 100)
    with pytest.raises(ValueError, match="64 MB"):
        provider._build_h3_request(
            content=[
                *text_content,
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + "a" * 200},
                },
            ],
            duration=5,
        )


def test_minimax_h3_exposes_current_model_and_pricing_metadata():
    provider = MiniMaxProvider(api_key="unit-value")
    assert provider.DEFAULT_VIDEO_MODEL == "MiniMax-H3"
    assert provider.H3_TASK_STATUSES == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    }
    assert provider.H3_TASK_TYPES == {
        "generation",
        "h3_context_ir",
        "regeneration",
    }
    metadata = provider.video_model_metadata["MiniMax-H3"]

    assert metadata["release_date"] == "2026-07-31"
    assert metadata["api_version"] == "v2"
    assert metadata["input_modalities"] == ["text", "image", "video", "audio"]
    assert metadata["output_modalities"] == ["video", "audio"]
    assert metadata["resolutions"] == ["768P", "2K"]
    assert metadata["duration_seconds"] == {"min": 4, "max": 15, "integer": True}
    assert metadata["pricing"]["global_en"]["output_video"]["rates"] == {
        "768P": 0.08,
        "2K": 0.13,
    }
    assert metadata["pricing"]["global_en"]["input_reference_video"]["rates"] == {
        "768P": 0.08,
        "2K": 0.13,
    }
    assert (
        metadata["pricing"]["global_en"]["input_reference_images"]["additional_image"]
        == 0.04
    )
    assert metadata["pricing"]["cn_zh"]["output_video"]["rates"] == {
        "768P": 0.5,
        "2K": 0.8,
    }
    assert metadata["pricing"]["cn_zh"]["input_reference_video"]["rates"] == {
        "768P": 0.5,
        "2K": 0.8,
    }
    assert (
        metadata["pricing"]["cn_zh"]["input_reference_images"]["additional_image"]
        == 0.2
    )


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
    assert provider.supported_modalities == ["video", "music", "audio", "image"]
