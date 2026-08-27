"""``data:`` URL handling for media outputs (issue #587 and its siblings).

Gemini image models reached through OpenRouter return their output inline as a
``data:<mime>;base64,<payload>`` URL rather than an http(s) URL. Handing such a
URL to ``requests.get`` raises ``InvalidSchema``, so every media output that can
carry a URL has to decode inline payloads locally instead of downloading them.

Only the base64 form is decodable. A ``data:`` URL that carries no payload, an
empty payload, or declares a non-base64 encoding must be rejected loudly rather
than decoded into wrong-but-plausible bytes — before these helpers existed such
a URL reached ``requests`` and raised ``InvalidSchema``, so silently yielding
``b""`` would be a regression in disguise. Strictness stops at the base64
alphabet: whitespace inside the payload (RFC 2045 line wrapping, a trailing
newline, a space after the comma) is layout, not corruption, and must still
decode.

Rejecting a payload must also never cost the caller data: ``save()`` resolves
the bytes before it opens the destination, so a file already at that path
survives a failed save intact.
"""

import base64
import binascii
import tracemalloc

import pytest
import requests

from agentfield.data_url import data_url_mime_type, decode_data_url, is_data_url
from agentfield.multimodal import Audio, audio_from_url
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

# None of these is decodable: no payload at all, an empty payload, a plain-text
# (percent-encoded) payload rather than base64, and a payload declared base64
# that is not.
NO_COMMA_DATA_URL = "data:image/png"
EMPTY_PAYLOAD_DATA_URL = "data:image/png;base64,"
NOT_BASE64_DATA_URL = "data:text/plain,hello"
CORRUPT_DATA_URL = f"data:image/png;base64,{B64}!!"

UNDECODABLE_DATA_URLS = [
    NO_COMMA_DATA_URL,
    EMPTY_PAYLOAD_DATA_URL,
    NOT_BASE64_DATA_URL,
    CORRUPT_DATA_URL,
]

# Whitespace inside a payload is layout, not corruption: RFC 2045 encoders wrap
# at 76 columns, and a stray leading space or trailing newline is common.
BIG_PAYLOAD = bytes(range(256)) * 4
WRAPPED_DATA_URL = "data:image/png;base64," + base64.encodebytes(BIG_PAYLOAD).decode()
TRAILING_NEWLINE_DATA_URL = f"data:image/png;base64,{B64}\n"
LEADING_SPACE_DATA_URL = f"data:image/png;base64, {B64}"
SPLIT_DATA_URL = f"data:image/png;base64,{B64[:2]}\r\n\t{B64[2:]}"

# What an existing file at the save destination holds, so a truncating save is
# visible as a length change rather than only a content change.
EXISTING_BYTES = b"x" * 820

# (label, factory) for the three outputs whose save() writes a URL payload.
URL_OUTPUTS = [
    ("ImageOutput", ImageOutput),
    ("FileOutput", FileOutput),
    ("VideoOutput", VideoOutput),
]


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


@pytest.mark.parametrize(
    "make", [cls for _, cls in URL_OUTPUTS], ids=[label for label, _ in URL_OUTPUTS]
)
class TestSaveDoesNotDestroyTheDestination:
    """``save()`` must resolve its bytes before it truncates anything.

    ``open(path, "wb")`` truncates on open, so deciding the payload is
    undecodable *after* opening turns "this URL is broken" into "your file is
    gone" — the caller loses data they already had.
    """

    @pytest.mark.parametrize("bad_url", UNDECODABLE_DATA_URLS)
    def test_rejected_payload_leaves_an_existing_file_intact(
        self, tmp_path, no_network, make, bad_url
    ):
        path = tmp_path / "existing.bin"
        path.write_bytes(EXISTING_BYTES)
        with pytest.raises(ValueError):
            make(url=bad_url).save(path)
        assert path.read_bytes() == EXISTING_BYTES

    @pytest.mark.parametrize("bad_url", UNDECODABLE_DATA_URLS)
    def test_rejected_payload_leaves_no_stub_file_behind(
        self, tmp_path, no_network, make, bad_url
    ):
        path = tmp_path / "new.bin"
        with pytest.raises(ValueError):
            make(url=bad_url).save(path)
        assert not path.exists()

    def test_failed_download_leaves_an_existing_file_intact(
        self, tmp_path, monkeypatch, make
    ):
        def _boom(url, **kwargs):
            raise requests.exceptions.ConnectionError("network down")

        monkeypatch.setattr(requests, "get", _boom)
        path = tmp_path / "existing.bin"
        path.write_bytes(EXISTING_BYTES)
        with pytest.raises(requests.exceptions.ConnectionError):
            make(url=HTTP_URL).save(path)
        assert path.read_bytes() == EXISTING_BYTES

    def test_a_good_payload_still_overwrites_an_existing_file(
        self, tmp_path, no_network, make
    ):
        path = tmp_path / "existing.bin"
        path.write_bytes(EXISTING_BYTES)
        make(url=IMAGE_DATA_URL).save(path)
        assert path.read_bytes() == PAYLOAD


class TestWhitespaceInPayloads:
    """Layout whitespace decodes; corruption still raises.

    The permissive decoder these helpers replaced accepted line-wrapped
    payloads, so rejecting them would break callers that work today.
    """

    @pytest.mark.parametrize(
        "url,expected",
        [
            (WRAPPED_DATA_URL, BIG_PAYLOAD),
            (TRAILING_NEWLINE_DATA_URL, PAYLOAD),
            (LEADING_SPACE_DATA_URL, PAYLOAD),
            (SPLIT_DATA_URL, PAYLOAD),
        ],
        ids=["rfc2045-wrapped", "trailing-newline", "space-after-comma", "cr-lf-tab"],
    )
    def test_whitespace_is_not_corruption(self, url, expected):
        assert decode_data_url(url) == expected

    def test_the_wrapped_fixture_really_is_wrapped(self):
        # Guards the test above from silently becoming a no-op.
        assert "\n" in WRAPPED_DATA_URL.partition(",")[2]

    @pytest.mark.parametrize(
        "make", [cls for _, cls in URL_OUTPUTS], ids=[label for label, _ in URL_OUTPUTS]
    )
    def test_media_outputs_accept_wrapped_payloads(self, make, no_network):
        assert make(url=WRAPPED_DATA_URL).get_bytes() == BIG_PAYLOAD

    def test_audio_accepts_wrapped_payloads(self, no_network):
        audio = Audio.from_url(WRAPPED_DATA_URL)
        assert base64.b64decode(audio.input_audio["data"]) == BIG_PAYLOAD

    def test_stripping_whitespace_does_not_rescue_a_corrupt_payload(self):
        with pytest.raises(binascii.Error):
            decode_data_url(f"data:image/png;base64,{B64} !!")

    def test_a_whitespace_only_payload_is_empty_not_valid(self):
        with pytest.raises(ValueError, match="Empty data: URL payload"):
            decode_data_url("data:image/png;base64, \n ")


class TestEmptyPayloadIsRejected:
    """``data:image/png;base64,`` must not decode to ``b""``.

    Yielding empty bytes is the exact silent-wrong-answer failure the helper
    exists to prevent; it has to raise like the other undecodable forms.
    """

    def test_helper_rejects_an_empty_payload(self):
        with pytest.raises(ValueError, match="Empty data: URL payload"):
            decode_data_url(EMPTY_PAYLOAD_DATA_URL)

    @pytest.mark.parametrize(
        "make", [cls for _, cls in URL_OUTPUTS], ids=[label for label, _ in URL_OUTPUTS]
    )
    def test_media_outputs_reject_an_empty_payload(self, make, no_network):
        with pytest.raises(ValueError, match="Empty data: URL payload"):
            make(url=EMPTY_PAYLOAD_DATA_URL).get_bytes()

    def test_audio_rejects_an_empty_payload(self, no_network):
        with pytest.raises(ValueError, match="Empty data: URL payload"):
            Audio.from_url(EMPTY_PAYLOAD_DATA_URL)


class TestAudioFormatFollowsTheDataUrl:
    """``Audio.from_url`` must not label every inline payload ``wav``.

    The format is handed to the model as the declared encoding of the bytes;
    calling an MP3 a WAV is a wrong answer, not a cosmetic default.
    """

    @pytest.mark.parametrize(
        "mime,expected",
        [
            ("audio/mpeg", "mp3"),
            ("audio/mp3", "mp3"),
            ("audio/wav", "wav"),
            ("audio/x-wav", "wav"),
            ("audio/wave", "wav"),
            ("audio/flac", "flac"),
            ("audio/ogg", "ogg"),
            ("audio/aac", "wav"),  # unmapped -> the historical default
            ("", "wav"),  # RFC 2397 allows an absent MIME type
        ],
    )
    def test_format_is_derived_from_the_declared_mime_type(
        self, no_network, mime, expected
    ):
        audio = Audio.from_url(f"data:{mime};base64,{B64}")
        assert audio.input_audio["format"] == expected
        assert base64.b64decode(audio.input_audio["data"]) == PAYLOAD

    def test_mime_match_ignores_case_and_extra_parameters(self, no_network):
        assert (
            Audio.from_url(f"DATA:AUDIO/MPEG;BASE64,{B64}").input_audio["format"]
            == "mp3"
        )
        assert (
            Audio.from_url(f"data:audio/mpeg;charset=binary;base64,{B64}").input_audio[
                "format"
            ]
            == "mp3"
        )

    def test_an_explicit_format_always_wins(self, no_network):
        audio = Audio.from_url(f"data:audio/mpeg;base64,{B64}", format="flac")
        assert audio.input_audio["format"] == "flac"

    def test_http_urls_keep_the_wav_default(self, recorded_download):
        assert Audio.from_url(HTTP_URL).input_audio["format"] == "wav"
        assert Audio.from_url(HTTP_URL, format="mp3").input_audio["format"] == "mp3"

    def test_module_level_helper_derives_the_format_too(self, no_network):
        assert (
            audio_from_url(f"data:audio/ogg;base64,{B64}").input_audio["format"]
            == "ogg"
        )


class TestIsDataUrlCost:
    def test_detection_does_not_copy_the_url(self):
        """``is_data_url`` runs on every get_bytes()/save() call.

        An inline image is routinely megabytes, so lower-casing the whole URL
        to test a 5-character scheme allocates a full copy of the payload for
        nothing. Measured in allocations rather than wall time so the bound is
        deterministic.
        """
        big = "data:image/png;base64," + "A" * (10 * 1024 * 1024)
        tracemalloc.start()
        try:
            baseline = tracemalloc.get_traced_memory()[0]
            assert is_data_url(big) is True
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # A whole-URL .lower() would add ~10 MB here.
        assert peak - baseline < 1024 * 1024

    def test_short_strings_do_not_trip_the_scheme_slice(self):
        assert not is_data_url("")
        assert not is_data_url("data")
        assert is_data_url("data:")


class TestDataUrlMimeType:
    @pytest.mark.parametrize(
        "url,expected",
        [
            (f"data:audio/mpeg;base64,{B64}", "audio/mpeg"),
            (f"DATA:Audio/MPEG;BASE64,{B64}", "audio/mpeg"),
            (f"data:;base64,{B64}", ""),
            (f"data:image/png;charset=binary;base64,{B64}", "image/png"),
            ("data:image/png", "image/png"),
        ],
    )
    def test_mime_type_is_the_head_before_the_first_parameter(self, url, expected):
        assert data_url_mime_type(url) == expected
