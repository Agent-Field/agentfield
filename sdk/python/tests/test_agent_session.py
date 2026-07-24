import pytest

from agentfield import Agent, SessionTransportError


def test_app_session_registers_voice_metadata():
    app = Agent("support", auto_register=False)

    @app.session(
        "voice",
        provider="openai",
        model="gpt-realtime-2",
        transport="webrtc",
        modalities=["audio", "text"],
        voice="marin",
        tools=["launch_support_workflow"],
        tags=["voice", "pii"],
    )
    async def voice(session):
        return session

    assert app.sessions == [
        {
            "name": "voice",
            "provider": "openai",
            "transport": "webrtc",
            "model": "gpt-realtime-2",
            "modalities": ["audio", "text"],
            "voice": "marin",
            "tools": ["launch_support_workflow"],
            "tags": ["voice", "pii"],
            "proposed_tags": ["voice", "pii"],
            "approved_tags": [],
            "metadata": {},
        }
    ]
    assert app._build_agent_metadata()["sessions"] == app.sessions


def test_app_session_rejects_unsupported_provider_transport_pair():
    app = Agent("support", auto_register=False)

    with pytest.raises(SessionTransportError):
        app.session("voice", provider="openrouter", transport="webrtc")
