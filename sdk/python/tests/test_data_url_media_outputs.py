"""``data:`` URL handling for media outputs (issue #587 and its siblings).

Gemini image models reached through OpenRouter return their output inline as a
``data:<mime>;base64,<payload>`` URL rather than an http(s) URL. Handing such a
URL to ``requests.get`` raises ``InvalidSchema``, so every media output that can
carry a URL has to decode inline payloads locally instead of downloading them.
"""

import base64

import pytest
import requests

from agentfield.multimodal import Audio
from agentfield.multimodal_response import FileOutput, ImageOutput, VideoOutput


PAYLOAD = b"foo"
B64 = base64.b64encode(PAYLOAD).decode()  # "Zm9v"

IMAGE_DATA_URL = f"data:image/png;base64,{B64}"
FILE_DATA_URL = f"data:application/pdf;base64,{B64}"
VIDEO_DATA_URL = f"data:video/mp4;base64,{B64}"
AUDIO_DATA_URL = f"data:audio/wav;base64,{B64}"

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


class TestFileOutputDataUrl:
    def test_get_bytes_decodes_inline_payload(self, no_network):
        assert FileOutput(url=FILE_DATA_URL).get_bytes() == PAYLOAD

    def test_save_writes_inline_payload(self, tmp_path, no_network):
        path = tmp_path / "out.pdf"
        FileOutput(url=FILE_DATA_URL).save(path)
        assert path.read_bytes() == PAYLOAD

    def test_base64_data_field_still_wins_over_url(self, no_network):
        out = FileOutput(url=FILE_DATA_URL, data=base64.b64encode(b"bar").decode())
        assert out.get_bytes() == b"bar"

    def test_http_url_is_still_downloaded(self, tmp_path, recorded_download):
        path = tmp_path / "out.pdf"
        FileOutput(url=HTTP_URL).save(path)
        assert path.read_bytes() == PAYLOAD
        assert FileOutput(url=HTTP_URL).get_bytes() == PAYLOAD
        assert [url for url, _ in recorded_download] == [HTTP_URL, HTTP_URL]


class TestVideoOutputDataUrl:
    def test_get_bytes_decodes_inline_payload(self, no_network):
        assert VideoOutput(url=VIDEO_DATA_URL).get_bytes() == PAYLOAD

    def test_save_writes_inline_payload(self, tmp_path, no_network):
        path = tmp_path / "out.mp4"
        VideoOutput(url=VIDEO_DATA_URL).save(path)
        assert path.read_bytes() == PAYLOAD

    def test_http_url_is_still_downloaded_with_timeout(
        self, tmp_path, recorded_download
    ):
        path = tmp_path / "out.mp4"
        VideoOutput(url=HTTP_URL).save(path)
        assert path.read_bytes() == PAYLOAD
        url, kwargs = recorded_download[0]
        assert url == HTTP_URL
        assert kwargs["timeout"] == 120


class TestAudioFromDataUrl:
    def test_from_url_decodes_inline_payload(self, no_network):
        audio = Audio.from_url(AUDIO_DATA_URL, format="wav")
        assert base64.b64decode(audio.input_audio["data"]) == PAYLOAD
        assert audio.input_audio["format"] == "wav"

    def test_http_url_is_still_downloaded(self, recorded_download):
        audio = Audio.from_url(HTTP_URL, format="mp3")
        assert base64.b64decode(audio.input_audio["data"]) == PAYLOAD
        assert [url for url, _ in recorded_download] == [HTTP_URL]
