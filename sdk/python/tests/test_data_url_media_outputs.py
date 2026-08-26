"""``data:`` URL handling for media outputs (issue #587 and its siblings).

Gemini image models reached through OpenRouter return their output inline as a
``data:<mime>;base64,<payload>`` URL rather than an http(s) URL. Handing such a
URL to ``requests.get`` raises ``InvalidSchema``, so every media output that can
carry a URL has to decode inline payloads locally instead of downloading them.

Only the base64 form is decodable. A ``data:`` URL that carries no payload or
declares a non-base64 encoding must be rejected loudly rather than decoded into
wrong-but-plausible bytes — before these helpers existed such a URL reached
``requests`` and raised ``InvalidSchema``, so silently yielding ``b""`` would be
a regression in disguise.
"""

import base64
import binascii

import pytest
import requests

from agentfield.data_url import decode_data_url, is_data_url
from agentfield.multimodal import Audio
from agentfield.multimodal_response import FileOutput, ImageOutput, VideoOutput


PAYLOAD = b"foo"
B64 = base64.b64encode(PAYLOAD).decode()  # "Zm9v"

IMAGE_DATA_URL = f"data:image/png;base64,{B64}"
FILE_DATA_URL = f"data:application/pdf;base64,{B64}"
VIDEO_DATA_URL = f"data:video/mp4;base64,{B64}"
AUDIO_DATA_URL = f"data:audio/wav;base64,{B64}"

HTTP_URL = "https://example.test/media.bin"

# Scheme casing is not significant (RFC 3986 3.1) - this must still decode.
UPPERCASE_DATA_URL = f"DATA:image/png;base64,{B64}"

# Neither of these is decodable: the first carries no payload at all, the second
# declares a plain-text (percent-encoded) payload rather than base64.
NO_COMMA_DATA_URL = "data:image/png"
NOT_BASE64_DATA_URL = "data:text/plain,hello"


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

    def test_uppercase_scheme_is_still_decoded(self, no_network):
        assert ImageOutput(url=UPPERCASE_DATA_URL).get_bytes() == PAYLOAD

    def test_data_url_without_a_payload_is_rejected(self, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            ImageOutput(url=NO_COMMA_DATA_URL).get_bytes()

    def test_non_base64_data_url_is_rejected(self, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            ImageOutput(url=NOT_BASE64_DATA_URL).get_bytes()


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

    def test_uppercase_scheme_is_still_decoded(self, tmp_path, no_network):
        assert FileOutput(url=UPPERCASE_DATA_URL).get_bytes() == PAYLOAD
        path = tmp_path / "upper.png"
        FileOutput(url=UPPERCASE_DATA_URL).save(path)
        assert path.read_bytes() == PAYLOAD

    def test_data_url_without_a_payload_is_rejected(self, tmp_path, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            FileOutput(url=NO_COMMA_DATA_URL).get_bytes()
        with pytest.raises(ValueError, match="expected base64 payload"):
            FileOutput(url=NO_COMMA_DATA_URL).save(tmp_path / "nope.bin")

    def test_non_base64_data_url_is_rejected(self, tmp_path, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            FileOutput(url=NOT_BASE64_DATA_URL).get_bytes()
        with pytest.raises(ValueError, match="expected base64 payload"):
            FileOutput(url=NOT_BASE64_DATA_URL).save(tmp_path / "nope.bin")


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

    def test_uppercase_scheme_is_still_decoded(self, tmp_path, no_network):
        assert VideoOutput(url=UPPERCASE_DATA_URL).get_bytes() == PAYLOAD
        path = tmp_path / "upper.mp4"
        VideoOutput(url=UPPERCASE_DATA_URL).save(path)
        assert path.read_bytes() == PAYLOAD

    def test_data_url_without_a_payload_is_rejected(self, tmp_path, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            VideoOutput(url=NO_COMMA_DATA_URL).get_bytes()
        with pytest.raises(ValueError, match="expected base64 payload"):
            VideoOutput(url=NO_COMMA_DATA_URL).save(tmp_path / "nope.mp4")

    def test_non_base64_data_url_is_rejected(self, tmp_path, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            VideoOutput(url=NOT_BASE64_DATA_URL).get_bytes()
        with pytest.raises(ValueError, match="expected base64 payload"):
            VideoOutput(url=NOT_BASE64_DATA_URL).save(tmp_path / "nope.mp4")


class TestAudioFromDataUrl:
    def test_from_url_decodes_inline_payload(self, no_network):
        audio = Audio.from_url(AUDIO_DATA_URL, format="wav")
        assert base64.b64decode(audio.input_audio["data"]) == PAYLOAD
        assert audio.input_audio["format"] == "wav"

    def test_http_url_is_still_downloaded(self, recorded_download):
        audio = Audio.from_url(HTTP_URL, format="mp3")
        assert base64.b64decode(audio.input_audio["data"]) == PAYLOAD
        assert [url for url, _ in recorded_download] == [HTTP_URL]

    def test_uppercase_scheme_is_still_decoded(self, no_network):
        audio = Audio.from_url(UPPERCASE_DATA_URL, format="wav")
        assert base64.b64decode(audio.input_audio["data"]) == PAYLOAD

    def test_data_url_without_a_payload_is_rejected(self, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            Audio.from_url(NO_COMMA_DATA_URL)

    def test_non_base64_data_url_is_rejected(self, no_network):
        with pytest.raises(ValueError, match="expected base64 payload"):
            Audio.from_url(NOT_BASE64_DATA_URL)


class TestDataUrlHelper:
    """Direct coverage of the shared helper the four classes above go through."""

    def test_scheme_match_ignores_case(self):
        assert is_data_url("data:image/png;base64,Zm9v")
        assert is_data_url("DaTa:image/png;base64,Zm9v")

    def test_non_data_urls_and_non_strings_are_not_matched(self):
        assert not is_data_url(HTTP_URL)
        assert not is_data_url(None)
        assert not is_data_url(b"data:image/png;base64,Zm9v")

    def test_corrupt_base64_payload_raises_instead_of_truncating(self):
        # "Zm9v!!" is declared base64 but is not - the permissive decoder used
        # to drop the stray characters and hand back short, wrong bytes.
        with pytest.raises(binascii.Error):
            decode_data_url("data:image/png;base64,Zm9v!!")

    def test_rejection_message_never_echoes_the_payload(self):
        # binascii.Error subclasses ValueError, so the marker is what pins the
        # *rejection* rather than an incidental decode failure.
        with pytest.raises(ValueError, match="expected base64 payload") as excinfo:
            decode_data_url("data:text/plain,sup3r-s3cret")
        assert "sup3r-s3cret" not in str(excinfo.value)
