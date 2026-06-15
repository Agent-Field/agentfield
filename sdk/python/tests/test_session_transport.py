import pytest

from agentfield.session_transport import (
    SessionTransportError,
    validate_session_transport,
)


def test_validate_session_transport_accepts_explicit_supported_pairs():
    capability = validate_session_transport("OpenAI", "WebRTC")

    assert capability.provider == "openai"
    assert capability.transport == "webrtc"


def test_validate_session_transport_rejects_provider_transport_mismatch():
    with pytest.raises(SessionTransportError) as exc:
        validate_session_transport("openrouter", "webrtc")

    assert exc.value.provider == "openrouter"
    assert exc.value.transport == "webrtc"
    assert "Supported transports: audio_turns" in str(exc.value)
    assert "does not infer or switch providers" in str(exc.value)


def test_validate_session_transport_requires_explicit_transport():
    with pytest.raises(ValueError, match="transport is required"):
        validate_session_transport("openai", "")


def test_validate_session_transport_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown session provider 'custom'"):
        validate_session_transport("custom", "webrtc")
