"""Regression test for issue #589 — OpenRouter video download auth headers.

Issue #589: OpenRouterProvider.generate_video downloaded from unsigned_urls[0]
without passing the Authorization header, causing 401 on openrouter.ai-hosted videos.

The fix (PR #579) gates the auth header on the hostname: openrouter.ai URLs get
the Bearer token; CDN-hosted URLs get none.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agentfield.media_providers import OpenRouterProvider
from agentfield.multimodal_response import MultimodalResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VIDEO_BYTES = b"\x00\x00\x00\x1cftypmp4" + b"\x00" * 100

JOB_ID = "job-video-auth-589"
OPENROUTER_VIDEO_URL = "https://cdn.openrouter.ai/videos/job-video-auth-589/0.mp4"
CDN_VIDEO_URL = "https://storage.googleapis.com/third-party-cdn/video-589.mp4"


def _make_response(status, json_data=None, body=None, headers=None):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.headers = headers or {}
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    if body is not None:
        resp.read = AsyncMock(return_value=body)
    resp.text = AsyncMock(return_value=str(json_data or ""))
    return resp


class CapturingSession:
    """Mock aiohttp.ClientSession that captures GET request kwargs."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_index = 0
        self.get_calls = []  # (url, kwargs) tuples

    def post(self, url, **kwargs):
        resp = self._responses[self._call_index]
        self._call_index += 1
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        resp = self._responses[self._call_index]
        self._call_index += 1
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


class TestOpenRouterVideoDownloadAuth:
    """Verify that download requests carry the correct auth headers."""

    @pytest.mark.asyncio
    async def test_openrouter_url_gets_auth_header(self, monkeypatch):
        """openrouter.ai-hosted URLs should receive the Bearer Authorization header."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-auth-589")

        submit_resp = _make_response(
            202,
            {"id": JOB_ID, "status": "pending"},
        )
        poll_resp = _make_response(
            200,
            {
                "status": "completed",
                "unsigned_urls": [OPENROUTER_VIDEO_URL],
                "usage": {"cost": 0.15},
            },
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        session = CapturingSession([submit_resp, poll_resp, download_resp])
        provider = OpenRouterProvider()

        with patch("aiohttp.ClientSession", return_value=session):
            result = await provider.generate_video(
                prompt="test auth",
                model="openrouter/google/veo-2.0-generate-001",
                poll_interval=0.01,
                timeout=5.0,
            )

        # The download GET should be the last call
        assert len(session.get_calls) >= 1
        download_url, download_kwargs = session.get_calls[-1]
        assert download_url == OPENROUTER_VIDEO_URL

        # Must carry Authorization header for openrouter.ai URLs
        headers = download_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-test-auth-589", (
            "openrouter.ai download URL must carry the Bearer auth header"
        )

        assert isinstance(result, MultimodalResponse)
        assert result.has_videos

    @pytest.mark.asyncio
    async def test_cdn_url_does_not_get_auth_header(self, monkeypatch):
        """CDN-hosted URLs (non-openrouter.ai) should NOT receive auth headers."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-auth-589")

        submit_resp = _make_response(
            202,
            {"id": JOB_ID, "status": "pending"},
        )
        poll_resp = _make_response(
            200,
            {
                "status": "completed",
                "unsigned_urls": [CDN_VIDEO_URL],
                "usage": {"cost": 0.15},
            },
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        session = CapturingSession([submit_resp, poll_resp, download_resp])
        provider = OpenRouterProvider()

        with patch("aiohttp.ClientSession", return_value=session):
            result = await provider.generate_video(
                prompt="test cdn auth",
                model="openrouter/google/veo-2.0-generate-001",
                poll_interval=0.01,
                timeout=5.0,
            )

        # The download GET should be the last call
        assert len(session.get_calls) >= 1
        download_url, download_kwargs = session.get_calls[-1]
        assert download_url == CDN_VIDEO_URL

        # Must NOT carry auth headers for non-openrouter.ai URLs
        headers = download_kwargs.get("headers", {})
        assert "Authorization" not in headers, (
            "CDN download URL must NOT carry the Bearer auth header"
        )
        assert headers == {}, "Expected empty headers for non-openrouter.ai download"

        assert isinstance(result, MultimodalResponse)
        assert result.has_videos

    @pytest.mark.asyncio
    async def test_openrouter_subdomain_gets_auth_header(self, monkeypatch):
        """Subdomains of openrouter.ai should also get the auth header."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-subdomain")

        subdomain_url = "https://storage.openrouter.ai/videos/job-abc/0.mp4"

        submit_resp = _make_response(
            202,
            {"id": JOB_ID, "status": "pending"},
        )
        poll_resp = _make_response(
            200,
            {
                "status": "completed",
                "unsigned_urls": [subdomain_url],
                "usage": {"cost": 0.10},
            },
        )
        download_resp = _make_response(200, body=FAKE_VIDEO_BYTES)

        session = CapturingSession([submit_resp, poll_resp, download_resp])
        provider = OpenRouterProvider()

        with patch("aiohttp.ClientSession", return_value=session):
            result = await provider.generate_video(
                prompt="test subdomain",
                model="openrouter/google/veo-2.0-generate-001",
                poll_interval=0.01,
                timeout=5.0,
            )

        # The download GET should be the last call
        assert len(session.get_calls) >= 1
        download_url, download_kwargs = session.get_calls[-1]
        assert download_url == subdomain_url

        headers = download_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-test-subdomain", (
            "openrouter.ai subdomain should carry auth header"
        )

        assert isinstance(result, MultimodalResponse)
        assert result.has_videos