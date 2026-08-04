import pytest

from agentfield.media_providers import (
    MINIMAX_CN_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    MiniMaxProvider,
)


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
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _form_fields(form):
    fields = {}
    for name_md, _headers, value in form._fields:
        fields[name_md.get("name")] = value
    return fields


@pytest.mark.asyncio
async def test_minimax_voice_clone_full_flow_uses_global_endpoint(monkeypatch):
    audio_bytes = b"fake-reference-audio"
    session = CaptureSession(
        [
            FakeResponse(
                {
                    "file": {"file_id": "file-ref-123"},
                    "base_resp": {"status_code": 0},
                }
            ),
            FakeResponse(
                {
                    "voice_id": "voice-cloned-1",
                    "base_resp": {"status_code": 0},
                }
            ),
        ]
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value")

    result = await provider.clone_voice(
        audio=audio_bytes,
        voice_id="voice-cloned-1",
        model="minimax/speech-2.8-hd",
        filename="reference.wav",
        content_type="audio/wav",
    )

    assert len(session.calls) == 2
    upload_url, upload_kwargs = session.calls[0][1], session.calls[0][2]
    assert upload_url == f"{MINIMAX_GLOBAL_BASE_URL}/files/upload"
    assert upload_kwargs["headers"]["Authorization"] == "Bearer unit-value"
    upload_fields = _form_fields(upload_kwargs["data"])
    assert upload_fields["purpose"] == "voice_clone"
    assert upload_fields["file"] == audio_bytes

    clone_url, clone_kwargs = session.calls[1][1], session.calls[1][2]
    assert clone_url == f"{MINIMAX_GLOBAL_BASE_URL}/voice_clone"
    assert _form_fields(clone_kwargs["data"]) == {
        "file_id": "file-ref-123",
        "voice_id": "voice-cloned-1",
        "model": "speech-2.8-hd",
    }
    assert result.text == "voice-cloned-1"
    assert result.raw_response["upload"]["file"]["file_id"] == "file-ref-123"
    assert result.raw_response["clone"]["voice_id"] == "voice-cloned-1"


@pytest.mark.asyncio
async def test_minimax_voice_clone_with_file_path_uses_cn_endpoint(
    monkeypatch, tmp_path
):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"reference-bytes")
    session = CaptureSession(
        [
            FakeResponse(
                {
                    "data": {"file_id": "file-cn-9"},
                    "base_resp": {"status_code": 0},
                }
            ),
            FakeResponse(
                {
                    "voice_id": "voice-cn-1",
                    "base_resp": {"status_code": 0},
                }
            ),
        ]
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    provider = MiniMaxProvider(api_key="unit-value", base_url=MINIMAX_CN_BASE_URL)

    result = await provider.clone_voice(
        audio=audio_path,
        voice_id="voice-cn-1",
        model="speech-2.6-hd",
    )

    assert session.calls[0][1] == f"{MINIMAX_CN_BASE_URL}/files/upload"
    upload_fields = _form_fields(session.calls[0][2]["data"])
    assert upload_fields["file"] == b"reference-bytes"
    assert upload_fields["purpose"] == "voice_clone"
    assert session.calls[1][1] == f"{MINIMAX_CN_BASE_URL}/voice_clone"
    assert _form_fields(session.calls[1][2]["data"])["model"] == "speech-2.6-hd"
    assert result.text == "voice-cn-1"


@pytest.mark.asyncio
async def test_minimax_voice_clone_validates_inputs(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    provider = MiniMaxProvider()
    with pytest.raises(ValueError, match="API key required"):
        await provider.clone_voice(audio=b"x", voice_id="v", filename="a.wav")

    provider = MiniMaxProvider(api_key="unit-value")
    with pytest.raises(ValueError, match="voice_id"):
        await provider.clone_voice(audio=b"x", filename="a.wav")
    with pytest.raises(ValueError, match="purpose"):
        await provider.clone_voice(
            audio=b"x", voice_id="v", purpose="other", filename="a.wav"
        )
    with pytest.raises(ValueError, match="requires a model"):
        await provider.clone_voice(
            audio=b"x", voice_id="v", model="minimax/", filename="a.wav"
        )
    with pytest.raises(ValueError, match="file path or bytes"):
        await provider.clone_voice(audio=12345, voice_id="v", filename="a.wav")
    with pytest.raises(ValueError, match="filename"):
        await provider.clone_voice(audio=b"x", voice_id="v")


@pytest.mark.asyncio
async def test_minimax_voice_clone_checks_api_errors(monkeypatch):
    provider = MiniMaxProvider(api_key="unit-value")

    session = CaptureSession(
        [
            FakeResponse(
                {
                    "base_resp": {
                        "status_code": 1004,
                        "status_msg": "authentication failed",
                    }
                }
            )
        ]
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="authentication failed"):
        await provider.clone_voice(audio=b"x", voice_id="v", filename="a.wav")

    session = CaptureSession(
        [FakeResponse({"file": {}, "base_resp": {"status_code": 0}})]
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="no file_id"):
        await provider.clone_voice(audio=b"x", voice_id="v", filename="a.wav")

    session = CaptureSession(
        [
            FakeResponse({"file": {"file_id": "f1"}, "base_resp": {"status_code": 0}}),
            FakeResponse({"base_resp": {"status_code": 0}}),
        ]
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="no voice_id"):
        await provider.clone_voice(audio=b"x", voice_id="v", filename="a.wav")

    session = CaptureSession(
        [
            FakeResponse({"file": {"file_id": "f1"}, "base_resp": {"status_code": 0}}),
            FakeResponse({"error": "boom"}, status=500),
        ]
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="voice cloning clone failed"):
        await provider.clone_voice(audio=b"x", voice_id="v", filename="a.wav")
