"""OpenRouter image-generation retry behaviour on the provider path.

Issues #586 / #588: OpenRouter answers a 404 carrying "No endpoints found that
support the requested output modalities" when routing lands on an upstream
replica without the image modality (transient) or when no upstream provider in
the model's matrix accepts the requested ``image_config`` (deterministic).

``app.ai_generate_image()`` reaches OpenRouter through
``OpenRouterProvider.generate_image``, so the retry has to live there — these
tests exercise that entry point end to end over a fake aiohttp session.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agentfield import openrouter_retry
from agentfield.media_providers import OpenRouterProvider


NO_ENDPOINTS_BODY = json.dumps(
    {
        "error": {
            "message": (
                "No endpoints found that support the requested output "
                "modalities: image, text"
            ),
            "code": 404,
        }
    }
)

OTHER_404_BODY = json.dumps(
    {"error": {"message": "No allowed providers are available", "code": 404}}
)

SUCCESS_PAYLOAD = {
    "choices": [
        {
            "message": {
                "content": "here you go",
                "images": [
                    {"image_url": {"url": "data:image/png;base64,Zm9v"}},
                ],
            }
        }
    ]
}

IMAGE_CONFIG = {"aspect_ratio": "9:16"}


def _response(status, *, payload=None, text=""):
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=payload if payload is not None else {})
    resp.text = AsyncMock(return_value=text)
    return resp


def _ok():
    return _response(200, payload=SUCCESS_PAYLOAD)


def _no_endpoints_404():
    return _response(404, text=NO_ENDPOINTS_BODY)


class RecordingSession:
    """aiohttp.ClientSession double: records POST bodies, replays responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bodies = []

    def post(self, url, **kwargs):
        # Copy: the provider must be free to reuse/replace the body dict.
        body = kwargs.get("json")
        self.bodies.append(dict(body) if isinstance(body, dict) else body)
        resp = self._responses[len(self.bodies) - 1]
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace the retry backoff so tests do not wait real seconds."""
    sleeper = AsyncMock()
    monkeypatch.setattr(openrouter_retry, "sleep", sleeper)
    return sleeper


async def _generate(session, **kwargs):
    provider = OpenRouterProvider(api_key="sk-test-key")
    with patch("aiohttp.ClientSession", return_value=session):
        return await provider.generate_image(prompt="a vertical portrait", **kwargs)


@pytest.mark.asyncio
async def test_transient_no_endpoints_404_is_retried_until_it_succeeds(no_sleep):
    session = RecordingSession([_no_endpoints_404(), _no_endpoints_404(), _ok()])

    result = await _generate(session)

    assert result.has_images
    assert result.images[0].get_bytes() == b"foo"
    assert len(session.bodies) == 3
    # Backoff between attempts, not a hot loop.
    assert [c.args[0] for c in no_sleep.await_args_list] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_first_attempt_success_issues_a_single_request(no_sleep):
    session = RecordingSession([_ok()])

    result = await _generate(session, image_config=IMAGE_CONFIG)

    assert result.has_images
    assert len(session.bodies) == 1
    assert no_sleep.await_count == 0


@pytest.mark.asyncio
async def test_exhausted_retries_retry_once_with_image_config_stripped(no_sleep):
    session = RecordingSession(
        [_no_endpoints_404(), _no_endpoints_404(), _no_endpoints_404(), _ok()]
    )

    result = await _generate(session, image_config=IMAGE_CONFIG)

    assert result.has_images
    assert result.images[0].get_bytes() == b"foo"
    assert len(session.bodies) == 4
    # The three retried attempts asked for the caller's image_config...
    for body in session.bodies[:3]:
        assert body["image_config"] == IMAGE_CONFIG
    # ...and only the final fallback drops it.
    assert "image_config" not in session.bodies[3]
    # Everything else about the request is unchanged.
    assert session.bodies[3]["messages"] == session.bodies[0]["messages"]
    assert session.bodies[3]["modalities"] == ["image"]
    assert [c.args[0] for c in no_sleep.await_args_list] == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_exhausted_retries_without_image_config_raise(no_sleep):
    session = RecordingSession(
        [_no_endpoints_404(), _no_endpoints_404(), _no_endpoints_404()]
    )

    with pytest.raises(RuntimeError) as exc:
        await _generate(session)

    assert "No endpoints found" in str(exc.value)
    assert "404" in str(exc.value)
    assert len(session.bodies) == 3


@pytest.mark.asyncio
async def test_empty_image_config_is_not_worth_a_strip_attempt(no_sleep):
    """An empty image_config produces an identical request once stripped."""
    session = RecordingSession(
        [_no_endpoints_404(), _no_endpoints_404(), _no_endpoints_404()]
    )

    with pytest.raises(RuntimeError):
        await _generate(session, image_config={})

    assert len(session.bodies) == 3


@pytest.mark.asyncio
async def test_failure_of_the_stripped_attempt_is_surfaced(no_sleep):
    session = RecordingSession(
        [
            _no_endpoints_404(),
            _no_endpoints_404(),
            _no_endpoints_404(),
            _no_endpoints_404(),
        ]
    )

    with pytest.raises(RuntimeError) as exc:
        await _generate(session, image_config=IMAGE_CONFIG)

    assert "No endpoints found" in str(exc.value)
    assert len(session.bodies) == 4


@pytest.mark.asyncio
async def test_unrelated_404_raises_on_the_first_response(no_sleep):
    session = RecordingSession([_response(404, text=OTHER_404_BODY), _ok()])

    with pytest.raises(RuntimeError) as exc:
        await _generate(session, image_config=IMAGE_CONFIG)

    assert "No allowed providers are available" in str(exc.value)
    assert len(session.bodies) == 1
    assert no_sleep.await_count == 0


@pytest.mark.asyncio
async def test_server_error_raises_on_the_first_response(no_sleep):
    session = RecordingSession([_response(500, text="upstream exploded"), _ok()])

    with pytest.raises(RuntimeError) as exc:
        await _generate(session)

    assert "upstream exploded" in str(exc.value)
    assert "500" in str(exc.value)
    assert len(session.bodies) == 1
    assert no_sleep.await_count == 0
