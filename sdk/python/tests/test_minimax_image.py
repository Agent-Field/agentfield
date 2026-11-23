import asyncio
import base64
from types import SimpleNamespace

import aiohttp
import pytest

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
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_minimax_image_url_output_uses_global_endpoint_and_request_fields(
    monkeypatch,
):
    payload = {
        "base_resp": {"status_code": 0},
        "data": {
            "image_urls": [
                "https://example.test/one.png",
                "https://example.test/two.png",
            ]
        },
        "metadata": {"success_count": 2, "failed_count": 0},
    }
    session = CaptureSession(FakeResponse(payload))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value")

    result = await provider.generate_image(
        prompt="A luminous city at dusk",
        model="minimax/image-01-live",
        subject_reference=["reference-image"],
        aspect_ratio="16:9",
        response_format="url",
        seed=17,
        n=2,
        prompt_optimizer=False,
    )

    assert session.calls[0][0] == f"{MINIMAX_GLOBAL_BASE_URL}/image_generation"
    assert session.calls[0][1]["json"] == {
        "model": "image-01-live",
        "prompt": "A luminous city at dusk",
        "response_format": "url",
        "subject_reference": ["reference-image"],
        "aspect_ratio": "16:9",
        "seed": 17,
        "n": 2,
        "prompt_optimizer": False,
    }
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer unit-value"
    assert [image.url for image in result.images] == payload["data"]["image_urls"]
    assert result.raw_response["metadata"]["success_count"] == 2


@pytest.mark.asyncio
async def test_minimax_image_base64_output_uses_cn_endpoint_and_default_model(
    monkeypatch,
):
    image_bytes = b"generated-image"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    session = CaptureSession(
        FakeResponse(
            {
                "base_resp": {"status_code": 0},
                "data": {"image_base64": [encoded]},
            }
        )
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value", base_url=MINIMAX_CN_BASE_URL)

    result = await provider.generate_image(
        prompt="A geometric landscape",
        size="1280x720",
        response_format="b64_json",
    )

    assert session.calls[0][0] == f"{MINIMAX_CN_BASE_URL}/image_generation"
    assert session.calls[0][1]["json"] == {
        "model": "image-01",
        "prompt": "A geometric landscape",
        "response_format": "base64",
        "width": 1280,
        "height": 720,
    }
    assert result.images[0].url is None
    assert result.images[0].get_bytes() == image_bytes


@pytest.mark.asyncio
async def test_minimax_image_validates_credentials_format_and_api_errors(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    provider = MiniMaxProvider()
    with pytest.raises(ValueError, match="API key required"):
        await provider.generate_image("An image")

    provider = MiniMaxProvider(api_key="unit-value")
    for model in ("", " ", "minimax/   "):
        with pytest.raises(ValueError, match="requires a model"):
            await provider.generate_image("An image", model=model)
    with pytest.raises(ValueError, match="response_format"):
        await provider.generate_image("An image", response_format="binary")

    session = CaptureSession(
        FakeResponse(
            {
                "base_resp": {
                    "status_code": 1004,
                    "status_msg": "generation rejected",
                }
            }
        )
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="generation rejected"):
        await provider.generate_image("An image")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("not a dictionary", "invalid response"),
        ({"base_resp": "invalid"}, "invalid response"),
        ({"data": "invalid"}, "invalid response"),
        ({"data": {}}, "no image_urls"),
        ({"data": {"image_urls": []}}, "no images"),
    ],
)
async def test_minimax_image_rejects_invalid_responses(monkeypatch, payload, error):
    session = CaptureSession(FakeResponse(payload))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value")

    with pytest.raises(RuntimeError, match=error):
        await provider.generate_image("An image")


@pytest.mark.asyncio
async def test_minimax_image_checks_http_status(monkeypatch):
    session = CaptureSession(FakeResponse("upstream unavailable", status=503))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value")

    with pytest.raises(RuntimeError, match=r"failed \(503\): upstream unavailable"):
        await provider.generate_image("An image")


def test_minimax_image_models_route_to_provider():
    provider = MiniMaxProvider(api_key="unit-value")
    router = MediaRouter()
    router.register("minimax/", provider)

    assert router.resolve("minimax/image-01", "image") is provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "base_url", "expected_base", "response_format", "field", "value"),
    [
        (
            "MiniMax-H3-Image",
            MINIMAX_GLOBAL_BASE_URL,
            MINIMAX_GLOBAL_BASE_URL,
            "url",
            "url",
            "https://example.test/sunset.jpg",
        ),
        (
            "MiniMax-H3-Image-Highspeed",
            MINIMAX_CN_BASE_URL,
            "https://api.minimax.cn/v1",
            "base64",
            "b64_json",
            "c3Vuc2V0",
        ),
        (
            "MiniMax-H3-Image",
            "https://images.example.test/v1",
            "https://images.example.test/v1",
            "b64_json",
            "b64_json",
            "c3Vuc2V0",
        ),
    ],
)
async def test_h3_images_through_agent_api(
    monkeypatch, model, base_url, expected_base, response_format, field, value
):
    from types import SimpleNamespace

    from agentfield.agent_ai import AgentAI
    from agentfield.types import AIConfig

    payload = {
        "data": [
            {"error": {"code": "1027", "message": "image withheld"}},
            {field: value, "revised_prompt": "A golden sunset"},
        ],
        "usage": {"generated_images": 1},
    }
    session = CaptureSession(FakeResponse(payload))

    def client_session(timeout):
        assert timeout.total >= 600
        return session

    monkeypatch.setattr("aiohttp.ClientSession", client_session)
    config = AIConfig(
        vision_model=f"minimax/{model}",
        minimax_api_key="unit-value",
        minimax_base_url=base_url,
    )
    result = await AgentAI(SimpleNamespace(ai_config=config)).ai_generate_image(
        "Sunset",
        response_format=response_format,
        size="auto",
        area="2K",
        aspect_ratio="16:9",
        output_format="webp",
        output_compression=0,
        prompt_optimizer=False,
        n=2,
    )
    [(url, request)] = session.calls
    assert url == f"{expected_base}/images/generations"
    assert request["headers"]["Authorization"] == "Bearer unit-value"
    assert request["allow_redirects"] is False
    assert request["json"] == {
        "model": model,
        "prompt": "Sunset",
        "response_format": field,
        "size": "auto",
        "area": "2K",
        "aspect_ratio": "16:9",
        "output_format": "webp",
        "output_compression": 0,
        "disable_prompt_enhanced": True,
        "n": 2,
    }
    [image] = result.images
    assert getattr(image, field) == value
    assert image.revised_prompt == "A golden sunset"
    assert result.raw_response == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"subject_reference": ["ref"]}, "subject_reference"),
        ({"seed": 42}, "seed"),
        ({"width": 1024}, "size"),
        ({"stream": True}, "streaming"),
        ({"aspect_ratio": "1:1"}, "size='auto'"),
        ({"prompt_optimizer": True, "disable_prompt_enhanced": True}, "either"),
        ({"style": "vivid"}, "Unsupported"),
    ],
)
async def test_h3_rejects_unsupported_combinations_before_submission(options, message):
    with pytest.raises(ValueError, match=message):
        await MiniMaxProvider(api_key="unit-value").generate_image(
            "Sunset", model="MiniMax-H3-Image", **options
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (
            422,
            {
                "error": {"message": "all images withheld"},
                "data": [{"error": {"code": "1027"}}],
            },
            "all images withheld",
        ),
        (500, {}, "HTTP 500"),
        (200, [], "invalid image list"),
        (200, {"data": {}}, "invalid image list"),
        (200, {"data": [{"error": {"code": "1027"}}, {"url": None}]}, "no images"),
    ],
)
async def test_h3_reports_api_errors(monkeypatch, status, payload, message):
    session = CaptureSession(FakeResponse(payload, status))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match=message):
        await MiniMaxProvider(api_key="unit-value").generate_image(
            "Sunset", model="MiniMax-H3-Image"
        )
    [(_, request)] = session.calls
    assert request["json"]["size"] == "1024x1024"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ValueError(), "invalid JSON"),
        (TimeoutError(), "may still complete"),
        (asyncio.TimeoutError(), "may still complete"),
    ],
)
async def test_h3_does_not_retry_interrupted_or_invalid_responses(
    monkeypatch, error, message
):
    class BrokenResponse(FakeResponse):
        async def json(self):
            raise error

    session = CaptureSession(BrokenResponse(None))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match=message):
        await MiniMaxProvider(api_key="unit-value").generate_image(
            "Sunset", model="MiniMax-H3-Image"
        )
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        asyncio.TimeoutError(),
        TimeoutError(),
        aiohttp.ServerDisconnectedError(),
        aiohttp.ClientPayloadError("truncated body"),
        aiohttp.ClientResponseError(SimpleNamespace(real_url="x"), (), status=400),
    ],
)
async def test_h3_failures_before_the_response_is_read_are_not_retried(
    monkeypatch, error
):
    class FailingPost:
        async def __aenter__(self):
            raise error

        async def __aexit__(self, *args):
            return False

    class FailingSession(CaptureSession):
        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FailingPost()

    session = FailingSession(None)
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="may still complete"):
        await MiniMaxProvider(api_key="unit-value").generate_image(
            "Sunset", model="MiniMax-H3-Image"
        )
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": [{"error": {"code": "1027"}}]},
        [],
    ],
)
async def test_h3_unusable_success_bodies_warn_about_charges(monkeypatch, payload):
    session = CaptureSession(FakeResponse(payload))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="do not retry automatically"):
        await MiniMaxProvider(api_key="unit-value").generate_image(
            "Sunset", model="MiniMax-H3-Image"
        )
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_h3_all_failed_response_reports_the_per_image_reason(monkeypatch):
    payload = {
        "data": [{"error": {"code": "1027", "message": "image withheld"}}],
        "usage": {"generated_images": 0},
    }
    session = CaptureSession(FakeResponse(payload, status=422))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match=r"HTTP 422\): image withheld"):
        await MiniMaxProvider(api_key="unit-value").generate_image(
            "Sunset", model="MiniMax-H3-Image"
        )


@pytest.mark.asyncio
async def test_h3_malformed_revised_prompt_does_not_discard_the_batch(monkeypatch):
    payload = {
        "data": [
            {"url": "https://example.test/one.png", "revised_prompt": {}},
            {"url": "https://example.test/two.png", "revised_prompt": "A calm sea"},
        ]
    }
    session = CaptureSession(FakeResponse(payload))
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    result = await MiniMaxProvider(api_key="unit-value").generate_image(
        "Sunset", model="MiniMax-H3-Image"
    )
    assert [image.url for image in result.images] == [
        "https://example.test/one.png",
        "https://example.test/two.png",
    ]
    assert [image.revised_prompt for image in result.images] == [None, "A calm sea"]
