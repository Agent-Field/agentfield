"""``data:`` URL handling for media outputs (issue #587).

Gemini image models reached through OpenRouter return their output inline as a
``data:<mime>;base64,<payload>`` URL rather than an http(s) URL. Handing such a
URL to ``requests.get`` raises ``InvalidSchema``, so ``ImageOutput`` has to
decode inline payloads locally instead of downloading them.
"""

import base64

import pytest
import requests

from agentfield.multimodal_response import ImageOutput


PAYLOAD = b"foo"
B64 = base64.b64encode(PAYLOAD).decode()  # "Zm9v"

IMAGE_DATA_URL = f"data:image/png;base64,{B64}"

HTTP_URL = "https://example.test/media.bin"


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything tries to download while handling a data: URL."""

    def _boom(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args!r}")

    monkeypatch.setattr(requests, "get", _boom)


class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


@pytest.fixture
def recorded_download(monkeypatch):
    """Record http(s) downloads instead of performing them."""
    calls = []

    def _get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(PAYLOAD)

    monkeypatch.setattr(requests, "get", _get)
    return calls


class TestImageOutputDataUrl:
    def test_get_bytes_decodes_inline_payload(self, no_network):
        assert ImageOutput(url=IMAGE_DATA_URL).get_bytes() == PAYLOAD

    def test_save_writes_inline_payload(self, tmp_path, no_network):
        path = tmp_path / "out.png"
        ImageOutput(url=IMAGE_DATA_URL).save(path)
        assert path.read_bytes() == PAYLOAD

    def test_http_url_is_still_downloaded(self, tmp_path, recorded_download):
        path = tmp_path / "out.png"
        ImageOutput(url=HTTP_URL).save(path)
        assert path.read_bytes() == PAYLOAD
        assert [url for url, _ in recorded_download] == [HTTP_URL]
