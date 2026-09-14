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
            "turn_detection": {
                "type": "server_vad", "threshold": 0.5,
                "prefix_padding_ms": 300, "silence_duration_ms": 500,
                "create_response": True, "interrupt_response": True,
            },
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


@pytest.mark.parametrize("transport", ["webrtc", "websocket"])
@pytest.mark.parametrize("config,expected", [
    ({"type": "server_vad", "threshold": 0, "prefix_padding_ms": 0,
      "silence_duration_ms": 750, "create_response": False, "interrupt_response": False},
     {"type": "server_vad", "threshold": 0, "prefix_padding_ms": 0,
      "silence_duration_ms": 750, "create_response": False, "interrupt_response": False}),
    ({"type": "semantic_vad", "eagerness": "low"},
     {"type": "semantic_vad", "eagerness": "low", "create_response": True, "interrupt_response": True}),
])
def test_session_turn_detection_registration(transport, config, expected):
    app = Agent("support", auto_register=False)
    config = dict(config)
    async def handler(session):
        return session
    app.session("voice", provider="openai", transport=transport, turn_detection=config)(handler)
    config["type"] = "changed-after-registration"
    assert app.sessions[0]["turn_detection"] == expected
    assert app._build_agent_metadata()["sessions"][0]["turn_detection"] == expected


@pytest.mark.parametrize("config", [
    {}, {"type": "client_vad"}, {"type": "server_vad", "threshold": 2},
    {"type": "server_vad", "threshold": float("nan")},
    {"type": "server_vad", "threshold": True},
    {"type": "server_vad", "silence_duration_ms": -1},
    {"type": "server_vad", "prefix_padding_ms": 1.5},
    {"type": "server_vad", "create_response": "false"},
    {"type": "server_vad", "interrupt_response": None},
    {"type": "semantic_vad", "threshold": 0.5},
    {"type": "server_vad", "eagerness": "low"},
    {"type": "semantic_vad", "eagerness": "urgent"},
    {"type": "server_vad", "unknown": True}, [],
])
def test_session_rejects_invalid_turn_detection(config):
    app = Agent("support", auto_register=False)
    with pytest.raises(ValueError, match="turn_detection"):
        app.session("voice", provider="openai", transport="webrtc", turn_detection=config)
    assert app.sessions == []


def test_turn_detection_requires_openai():
    app = Agent("support", auto_register=False)
    with pytest.raises(ValueError, match="turn_detection requires"):
        app.session("voice", provider="openrouter", transport="audio_turns",
                    turn_detection={"type": "server_vad"})
    async def handler(session):
        return session
    app.session("voice", provider="openrouter", transport="audio_turns")(handler)
    assert "turn_detection" not in app.sessions[0]
