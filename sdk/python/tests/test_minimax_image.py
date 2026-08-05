import base64

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
