import pytest

from agentfield.media_providers import MINIMAX_CN_BASE_URL, MINIMAX_GLOBAL_BASE_URL, MiniMaxProvider


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
@pytest.mark.parametrize("base_url", [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CN_BASE_URL])
async def test_design_voice_posts_prompt_and_voice_id(monkeypatch, base_url):
    session = CaptureSession(
        FakeResponse({"voice_id": "designed-voice", "base_resp": {"status_code": 0}})
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)

    result = await MiniMaxProvider(api_key="unit-value", base_url=base_url).design_voice(
        prompt="A calm, warm narrator",
        voice_id="narrator-1",
    )

    assert session.calls[0][0] == f"{base_url}/voice_design"
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer unit-value"
    assert session.calls[0][1]["json"] == {
        "prompt": "A calm, warm narrator",
        "voice_id": "narrator-1",
    }
    assert result.text == "designed-voice"
    assert result.raw_response["base_resp"]["status_code"] == 0


@pytest.mark.asyncio
async def test_design_voice_validates_inputs_and_status(monkeypatch):
    provider = MiniMaxProvider(api_key="unit-value")
    with pytest.raises(ValueError, match="prompt"):
        await provider.design_voice(" ", "voice")
    with pytest.raises(ValueError, match="voice_id"):
        await provider.design_voice("A voice", "")

    session = CaptureSession(
        FakeResponse(
            {"base_resp": {"status_code": 1004, "status_msg": "invalid prompt"}}
        )
    )
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
    with pytest.raises(RuntimeError, match="invalid prompt"):
        await provider.design_voice("A voice", "voice")
