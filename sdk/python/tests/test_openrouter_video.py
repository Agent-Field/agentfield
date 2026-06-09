"""Tests for OpenRouterProvider.generate_video() — async polling video API."""

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest

from agentfield.media_providers import OpenRouterProvider, _assert_safe_download_url
from agentfield.multimodal_response import (
    MultimodalResponse,
    VideoOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VIDEO_BYTES = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100  # fake MP4 header
FAKE_VIDEO_B64 = base64.b64encode(FAKE_VIDEO_BYTES).decode()

JOB_ID = "job-abc123"
UNSIGNED_URL = "https://cdn.openrouter.ai/videos/job-abc123/0.mp4"


def _make_response(status, json_data=None, body=None, headers=None):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.headers = headers or {}
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    if body is not None:
        resp.read = AsyncMock(return_value=body)
    resp.text = AsyncMock(return_value=json.dumps(json_data) if json_data else "error")
    return resp


class FakeSession:
    """Mock aiohttp.ClientSession that returns pre-configured responses."""

    def __init__(self, responses):
        """responses: list of (method, url_substring, mock_response) tuples."""
        self._responses = list(responses)
        self._call_index = 0

    def post(self, url, **kwargs):
        return self._match("post", url)

    def get(self, url, **kwargs):
        return self._match("get", url)

    def _match(self, method, url):
        resp = self._responses[self._call_index]
        self._call_index += 1
        # Return async context manager
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenRouterVideoHappyPath:
    """Submit -> poll -> download -> return MultimodalResponse."""

    @pytest.mark.asyncio
    async def test_happy_path(self):
        submit_resp = _make_response(
            202,
            {
                "id": JOB_ID,
                "status": "pending",
                "polling_url": f"/api/v1/videos/{JOB_ID}",
            },
        )
        poll_resp = _make_response(
            200,
            {
                "status": "completed",
                "unsigned_urls": [UNSIGNED_URL],
                "usage": {"cost": 0.25},
            },
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        session = FakeSession([submit_resp, poll_resp, download_resp])

        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            result = await provider.generate_video(
                prompt="A cat walking",
                model="openrouter/google/veo-2.0-generate-001",
                duration=5,
                resolution="1080p",
                aspect_ratio="16:9",
                poll_interval=0.01,
                timeout=5.0,
            )

        assert isinstance(result, MultimodalResponse)
        assert result.has_files
        assert result.has_videos
        assert len(result.files) == 1
        assert len(result.videos) == 1
        assert result.files[0].mime_type == "video/mp4"
        assert result.videos[0].mime_type == "video/mp4"
        assert result.videos[0].url == UNSIGNED_URL
        assert result.cost_usd == 0.25
        assert result.text == "A cat walking"

    @pytest.mark.asyncio
    async def test_poll_pending_then_completed(self):
        """Status goes pending -> in_progress -> completed."""
        submit_resp = _make_response(202, {"id": JOB_ID, "status": "pending"})
        poll_pending = _make_response(200, {"status": "pending"})
        poll_progress = _make_response(200, {"status": "in_progress"})
        poll_done = _make_response(
            200,
            {
                "status": "completed",
                "unsigned_urls": [UNSIGNED_URL],
                "usage": {"cost": 0.10},
            },
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        session = FakeSession(
            [submit_resp, poll_pending, poll_progress, poll_done, download_resp]
        )
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            result = await provider.generate_video(
                prompt="test", poll_interval=0.01, timeout=5.0
            )

        assert result.has_videos
        assert result.cost_usd == 0.10


class TestOpenRouterVideoDownloadSafety:
    @pytest.mark.parametrize(
        "url",
        [
            "https://127.0.0.1/video.mp4",
            "https://localhost/video.mp4",
            "https://10.0.0.5/video.mp4",
            "https://172.16.0.1/video.mp4",
            "https://192.168.1.10/video.mp4",
            "https://169.254.1.1/video.mp4",
        ],
    )
    def test_rejects_localhost_and_private_download_urls(self, url):
        with pytest.raises(RuntimeError, match="Refusing to download"):
            _assert_safe_download_url(url)

    def test_allows_https_public_download_url(self):
        _assert_safe_download_url(UNSIGNED_URL)


class TestOpenRouterVideoTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        submit_resp = _make_response(202, {"id": JOB_ID, "status": "pending"})

        # Build a session that always returns pending on GET
        class AlwaysPendingSession(FakeSession):
            def get(self, url, **kwargs):
                cm = AsyncMock()
                cm.__aenter__ = AsyncMock(
                    return_value=_make_response(200, {"status": "pending"})
                )
                cm.__aexit__ = AsyncMock(return_value=False)
                return cm

        session = AlwaysPendingSession([submit_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(TimeoutError, match="timed out"):
                await provider.generate_video(
                    prompt="test", poll_interval=0.01, timeout=0.05
                )


class TestOpenRouterVideoErrorStatus:
    @pytest.mark.asyncio
    async def test_failed_status(self):
        submit_resp = _make_response(202, {"id": JOB_ID, "status": "pending"})
        poll_failed = _make_response(
            200, {"status": "failed", "error": "content policy violation"}
        )

        session = FakeSession([submit_resp, poll_failed])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RuntimeError, match="content policy violation"):
                await provider.generate_video(
                    prompt="test", poll_interval=0.01, timeout=5.0
                )

    @pytest.mark.asyncio
    async def test_submit_400(self):
        submit_resp = _make_response(400, {"error": "invalid model"})

        session = FakeSession([submit_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RuntimeError, match="Bad request"):
                await provider.generate_video(prompt="test", poll_interval=0.01)

    @pytest.mark.asyncio
    async def test_submit_401(self):
        submit_resp = _make_response(401, {"error": "unauthorized"})

        session = FakeSession([submit_resp])
        provider = OpenRouterProvider(api_key="sk-bad-key")

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RuntimeError, match="Invalid API key"):
                await provider.generate_video(prompt="test", poll_interval=0.01)

    @pytest.mark.asyncio
    async def test_submit_402(self):
        submit_resp = _make_response(402, {"error": "no credits"})

        session = FakeSession([submit_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RuntimeError, match="Insufficient credits"):
                await provider.generate_video(prompt="test", poll_interval=0.01)

    @pytest.mark.asyncio
    async def test_submit_429(self):
        submit_resp = _make_response(429, {"error": "rate limited"})

        session = FakeSession([submit_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RuntimeError, match="Rate limited"):
                await provider.generate_video(prompt="test", poll_interval=0.01)


class TestOpenRouterVideoModelPrefix:
    @pytest.mark.asyncio
    async def test_strips_openrouter_prefix(self):
        """Model name should have openrouter/ prefix stripped before sending to API."""
        submit_resp = _make_response(202, {"id": JOB_ID, "status": "pending"})
        poll_resp = _make_response(
            200, {"status": "completed", "unsigned_urls": [UNSIGNED_URL], "usage": {}}
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        captured_body = {}

        class CapturingSession(FakeSession):
            def post(self, url, **kwargs):
                captured_body.update(kwargs.get("json", {}))
                return super().post(url, **kwargs)

        session = CapturingSession([submit_resp, poll_resp, download_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            await provider.generate_video(
                prompt="test",
                model="openrouter/google/veo-2.0-generate-001",
                poll_interval=0.01,
                timeout=5.0,
            )

        assert captured_body["model"] == "google/veo-2.0-generate-001"

    @pytest.mark.asyncio
    async def test_no_prefix_passthrough(self):
        """Model without openrouter/ prefix should pass through as-is."""
        submit_resp = _make_response(202, {"id": JOB_ID, "status": "pending"})
        poll_resp = _make_response(
            200, {"status": "completed", "unsigned_urls": [UNSIGNED_URL], "usage": {}}
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        captured_body = {}

        class CapturingSession(FakeSession):
            def post(self, url, **kwargs):
                captured_body.update(kwargs.get("json", {}))
                return super().post(url, **kwargs)

        session = CapturingSession([submit_resp, poll_resp, download_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            await provider.generate_video(
                prompt="test",
                model="google/veo-2.0-generate-001",
                poll_interval=0.01,
                timeout=5.0,
            )

        assert captured_body["model"] == "google/veo-2.0-generate-001"


class TestOpenRouterVideoImageUrl:
    @pytest.mark.asyncio
    async def test_image_url_included_in_body(self):
        """image_url must be sent in the API request body (CR-01 regression)."""
        submit_resp = _make_response(202, {"id": JOB_ID, "status": "pending"})
        poll_resp = _make_response(
            200, {"status": "completed", "unsigned_urls": [UNSIGNED_URL], "usage": {}}
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        captured_body = {}

        class CapturingSession(FakeSession):
            def post(self, url, **kwargs):
                captured_body.update(kwargs.get("json", {}))
                return super().post(url, **kwargs)

        session = CapturingSession([submit_resp, poll_resp, download_resp])
        provider = OpenRouterProvider(api_key="sk-test-key")

        with patch("aiohttp.ClientSession", return_value=session):
            await provider.generate_video(
                prompt="animate this image",
                model="openrouter/google/veo-2.0-generate-001",
                image_url="https://example.com/input.jpg",
                poll_interval=0.01,
                timeout=5.0,
            )

        assert "image_url" in captured_body
        assert captured_body["image_url"] == "https://example.com/input.jpg"


class TestOpenRouterSupportedModalities:
    def test_includes_video(self):
        provider = OpenRouterProvider(api_key="sk-test")
        assert "video" in provider.supported_modalities
        assert "image" in provider.supported_modalities


class TestOpenRouterVideoNoApiKey:
    @pytest.mark.asyncio
    async def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        provider = OpenRouterProvider()

        with pytest.raises(ValueError, match="API key required"):
            await provider.generate_video(prompt="test")


class TestVideoOutput:
    def test_video_output_fields(self):
        v = VideoOutput(
            url="https://example.com/video.mp4",
            mime_type="video/mp4",
            filename="test.mp4",
            cost_usd=0.25,
        )
        assert v.url == "https://example.com/video.mp4"
        assert v.mime_type == "video/mp4"
        assert v.cost_usd == 0.25

    def test_video_output_get_bytes_from_data(self):
        data = base64.b64encode(b"fake-video").decode()
        v = VideoOutput(data=data)
        assert v.get_bytes() == b"fake-video"

    def test_video_output_no_data_raises(self):
        v = VideoOutput()
        with pytest.raises(ValueError, match="No video data"):
            v.get_bytes()


class TestMultimodalResponseVideos:
    def test_has_videos_property(self):
        resp = MultimodalResponse(
            text="test",
            videos=[VideoOutput(url="https://example.com/v.mp4")],
        )
        assert resp.has_videos
        assert len(resp.videos) == 1
        assert resp.is_multimodal

    def test_no_videos(self):
        resp = MultimodalResponse(text="test")
        assert not resp.has_videos
        assert resp.videos == []

    def test_repr_includes_videos(self):
        resp = MultimodalResponse(
            text="test",
            videos=[VideoOutput(url="https://example.com/v.mp4")],
        )
        assert "videos=1" in repr(resp)
